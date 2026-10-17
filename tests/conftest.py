"""Shared fixtures for hermes-prospecta tests.

The plugin module lives at the repo root as ``__init__.py`` (loaded by
Hermes as ``plugins.memory.prospecta``). For test isolation we import it
as a synthetic ``hermes_prospecta`` module via importlib so we don't
collide with the tests package.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Optional checkouts (CI and fresh clones have neither: prospecta comes from
# the pinned dependency, hermes' MemoryProvider from the stub below).
for _env in ("PROSPECTA_ROOT", "HERMES_ROOT"):
    _p = os.environ.get(_env)
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


def _ensure_memory_provider():
    """Provide ``agent.memory_provider.MemoryProvider`` when hermes is absent."""
    try:
        import agent.memory_provider  # noqa: F401
        return
    except ImportError:
        pass
    import abc
    import types

    class MemoryProvider(abc.ABC):
        """Minimal stand-in for hermes' MemoryProvider ABC."""

    agent = types.ModuleType("agent")
    mod = types.ModuleType("agent.memory_provider")
    mod.MemoryProvider = MemoryProvider
    agent.memory_provider = mod
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = mod


_ensure_memory_provider()


def _load_plugin_module():
    """Load the plugin's __init__.py as 'hermes_prospecta'."""
    if "hermes_prospecta" in sys.modules:
        return sys.modules["hermes_prospecta"]
    spec = importlib.util.spec_from_file_location(
        "hermes_prospecta", str(REPO_ROOT / "__init__.py")
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_prospecta"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def plugin_module():
    return _load_plugin_module()


@pytest.fixture(autouse=True)
def _restore_timezone_env():
    """``initialize`` hands the profile's zone to the library as PROSPECTA_TIMEZONE (process-wide):
    leave no test's value behind for the next."""
    before = os.environ.get("PROSPECTA_TIMEZONE")
    yield
    if before is None:
        os.environ.pop("PROSPECTA_TIMEZONE", None)
    else:
        os.environ["PROSPECTA_TIMEZONE"] = before


# ---------------------------------------------------------------------------
# Stub embedder + ctx (no provider deps; reuses prospecta's stub embedder)
# ---------------------------------------------------------------------------

import hashlib
import math
import re

EMBED_DIM = 32


def stub_embed(texts):
    out = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for tok in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = h[0] % EMBED_DIM
            sign = 1.0 if (h[1] & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        out.append(vec)
    return out




class _StubCtx:
    def __init__(self):
        self.registered = []

    def register_memory_provider(self, provider):
        self.registered.append(provider)


def stub_llm(messages, *, json_mode: bool = False) -> str:
    """Deterministic stub LLMCallable — no real provider needed."""
    if json_mode:
        return '{"queries": ["what is this about?"]}'
    return "Synthesized answer based on memory."


@pytest.fixture
def mock_ctx():
    return _StubCtx()


# ---------------------------------------------------------------------------
# Postgres testcontainer (session-scoped, shared with prospecta tests style)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pg_container():
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers not installed")
    import psycopg
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    try:
        url = container.get_connection_url().replace("+psycopg2", "").replace("+psycopg", "")
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        yield container
    finally:
        container.stop()


@pytest.fixture
def pg_url(pg_container):
    """Fresh database per test."""
    import psycopg
    base = pg_container.get_connection_url().replace("+psycopg2", "").replace("+psycopg", "")
    db_name = f"test_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')
    parsed = urlparse(base)
    new_url = urlunparse(parsed._replace(path=f"/{db_name}"))
    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    yield new_url


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def provider_with_stubs(plugin_module, mock_ctx, pg_url, hermes_home, monkeypatch):
    """An initialized ProspectaProvider with stub embedder + stub LLM callable."""
    monkeypatch.setenv("DATABASE_URL", pg_url)

    provider = plugin_module.ProspectaProvider()
    # Patch _build_embedder to use the stub (avoid LiteLLM deps).
    monkeypatch.setattr(provider, "_build_embedder", lambda: stub_embed)
    # Patch _build_llm to use the stub callable (no provider needed).
    monkeypatch.setattr(provider, "_build_llm", lambda: stub_llm)

    # Set embedding_dim=32 to match stub.
    cfg_path = hermes_home / "prospecta.json"
    cfg_path.write_text('{"bank_id": "test", "embedding_dim": "32"}')

    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))
    yield provider
    provider.shutdown()


# ---------------------------------------------------------------------------
# A real bank on the library's hybrid search (Stage 1 + Stage 2), offline
# ---------------------------------------------------------------------------

class FakeRerank:
    """OpenRouter's rerank endpoint, offline: the wire shape the library sends and reads
    (``{model, query, documents, top_n}`` -> ``{results: [{index, relevance_score}], usage}``). A
    document's score is the share of the query's words it holds (0..1)."""

    def __init__(self, fail=None, delay=0.0):
        self.requests, self.fail, self.delay = [], fail, delay

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        query = set(re.findall(r"\w+", request["query"].split("Query:", 1)[-1].lower()))
        results = []
        for i, doc in enumerate(request["documents"]):
            words = set(re.findall(r"\w+", doc.lower()))
            results.append({"index": i, "relevance_score": round(len(query & words) / max(len(query), 1), 3)})
        return {"model": request["model"], "results": results, "usage": {"total_tokens": 7, "cost": 0.0001}}


class CountingLLM:
    """A stub text model that counts its calls (retain must make none)."""

    def __init__(self, queries=("q",)):
        self.calls, self.queries = 0, queries

    def __call__(self, messages, *, json_mode=False, **kw):
        self.calls += 1
        if json_mode:
            return json.dumps({"queries": [{"text": q} for q in self.queries]})
        return "A synthesised answer."


@pytest.fixture
def hybrid_bank(plugin_module, pg_url, hermes_home, monkeypatch):
    """Factory: an initialised provider on a fresh Postgres whose bank has the ``hybrid_chunk`` channel
    and the ``cross_encoder`` rerank over a FakeRerank (``p.rerank``). Shut down at the end."""
    made = []

    def build(*, config=None, rerank=None, hybrid=True, bank="hyb", llm=None):
        for k in ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_DECISION_MODEL",
                  "PROSPECTA_RERANKER_MODEL", "PROSPECTA_QUERY_WRITER_MODEL", "PROSPECTA_RECALL_STAGE3",
                  "PROSPECTA_TIMEZONE", "PROSPECTA_LINKED_WRITES", "PROSPECTA_DUAL_WRITE_BANK",
                  "PROSPECTA_SHADOW_BANK"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
        monkeypatch.setenv("PROSPECTA_BANK_ID", bank)
        cfg = {"embedding_dim": "32"}
        cfg.update(config or {})
        (hermes_home / "prospecta.json").write_text(json.dumps(cfg), encoding="utf-8")
        p = plugin_module.ProspectaProvider()
        monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
        monkeypatch.setattr(p, "_build_llm", lambda: llm)
        p.initialize(session_id="s1", hermes_home=str(hermes_home))
        if hybrid:
            p._memory.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
            p._memory.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
        else:   # a bank created fresh is hybrid since library PR 86 (bank_defaults): "without the channel" must be made legacy
            p._memory.set_channel_config([])
            p._memory.set_recall_config({})
        p.rerank = rerank if rerank is not None else FakeRerank()
        p._memory._rerank_transport = p.rerank
        made.append(p)
        return p

    yield build
    for p in made:
        p.shutdown()
