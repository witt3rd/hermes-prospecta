"""Shared fixtures for hermes-prospecta tests.

The plugin module lives at the repo root as ``__init__.py`` (loaded by
Hermes as ``plugins.memory.prospecta``). For test isolation we import it
as a synthetic ``hermes_prospecta`` module via importlib so we don't
collide with the tests package.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROSPECTA_ROOT = Path("/home/dt/src/witt3rd/prospecta")
HERMES_ROOT = Path("/home/dt/src/ext/hermes-agent")

# Make prospecta + agent.memory_provider importable.
for p in (str(PROSPECTA_ROOT), str(HERMES_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


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


class _StubResult:
    def __init__(self, text: str):
        self.text = text


class _StubLLM:
    def __init__(self):
        self.complete_calls = []
        self.complete_structured_calls = []

    def complete(self, *, messages, purpose=None, **_kw):
        self.complete_calls.append({"messages": messages, "purpose": purpose})
        # Deterministic-enough synthesis text
        return _StubResult("Synthesized answer based on memory.")

    def complete_structured(self, *, instructions, input, json_mode=False,
                            purpose=None, **_kw):
        self.complete_structured_calls.append({
            "instructions": instructions, "input": input,
            "json_mode": json_mode, "purpose": purpose,
        })
        # Return JSON object shaped like prospecta's formulate parser expects:
        # a list of query strings under "queries". Prospecta is tolerant on
        # parse failure (falls back to single-query).
        return _StubResult('{"queries": ["what is this about?"]}')


class _StubCtx:
    def __init__(self):
        self.llm = _StubLLM()
        self.registered = []

    def register_memory_provider(self, provider):
        self.registered.append(provider)


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
    """An initialized ProspectaProvider wired with stub embedder + ctx.llm."""
    monkeypatch.setenv("DATABASE_URL", pg_url)

    provider = plugin_module.ProspectaProvider()
    provider._ctx = mock_ctx
    # Patch _build_embedder to use the stub (avoid LiteLLM deps).
    monkeypatch.setattr(provider, "_build_embedder", lambda: stub_embed)

    # Set embedding_dim=32 to match stub.
    cfg_path = hermes_home / "prospecta.json"
    cfg_path.write_text('{"bank_id": "test", "embedding_dim": "32"}')

    provider.initialize(session_id="test-session", hermes_home=str(hermes_home))
    yield provider
    provider.shutdown()
