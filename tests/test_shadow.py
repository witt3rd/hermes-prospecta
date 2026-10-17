"""Shadow reads: old/new pairs for search and ask (retrieval only on the shadow side), off the old path."""
from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tests.conftest import check_call, stub_embed, stub_llm

OLD, NEW = "old_bank", "new_bank"


def _rm(source, bank):
    from prospecta._types import RecalledMemory
    return RecalledMemory(
        content=f"c {source}", original_chunk=f"chunk {source}", source=source, score=0.5,
        scores={"hybrid_chunk": 0.5, "rrf": 0.5},
        metadata={}, bank_id=bank, document_id=f"{bank}-{source}",
    )


class OldMemory:
    """Stands in for the old-bank Memory: serves old answers, records events."""

    def __init__(self):
        self.items = [_rm("a", OLD), _rm("b", OLD)]
        self.asked = SimpleNamespace(
            answer="old answer [1]", rendered="old answer [1]\n\nSources", citations=[], items=self.items,
            sufficient=None, retried=False, gate=None)
        self.events = []
        self.calls = []

    def search(self, q, **kw):
        check_call("search", q, **kw)
        self.calls.append(("search", q))
        return self.items

    def ask(self, q, **kw):
        check_call("ask", q, **kw)
        self.calls.append(("ask", q))
        return self.asked

    def _tracer(self, kind, payload):
        self.events.append((kind, payload))


class NewMemory:
    def __init__(self, *, delay=0.0, fail=None):
        self.delay, self.fail = delay, fail
        self.searches, self.closed = [], False

    def search(self, q, **kw):
        check_call("search", q, **kw)
        self.searches.append((q, kw))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return [_rm("a", NEW), _rm("c", NEW)]

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def origin_ids(monkeypatch, request):
    if "pg_url" in request.fixturenames:  # the real-database test uses the real lookup
        return
    # document id "<bank>-<source>" -> original id "<source>" (the DB lookup is a library detail)
    import prospecta._shadow as lib
    monkeypatch.setattr(
        lib, "_origin_ids", lambda memory, bank, results: [r.document_id.split("-", 1)[1] for r in results])


def _provider(plugin_module, *, shadow=True, new=None):
    p = plugin_module.ProspectaProvider()
    p._memory = OldMemory()
    p._llm = object()
    p._new = new or NewMemory()
    if shadow:
        p._shadow = plugin_module._ShadowReader(
            p._memory, p._new, bank_id=OLD, shadow_bank_id=NEW)
    return p


def _drain(p):
    """Stop the worker (processes everything queued first)."""
    shadow, p._shadow = p._shadow, None
    shadow.stop()


def _pairs(p):
    return [e[1] for e in p._memory.events]


def test_off_by_default_changes_nothing(plugin_module, monkeypatch):
    monkeypatch.delenv("PROSPECTA_SHADOW_BANK", raising=False)
    p = plugin_module.ProspectaProvider()
    assert p._shadow is None and p._resolve_shadow_bank_id() == ""
    p._memory, p._llm = OldMemory(), object()
    out = json.loads(p.handle_tool_call("prospecta_search", {"query": "x"}))
    assert [r["source"] for r in out["results"]] == ["a", "b"]
    assert p._memory.events == []
    p._start_shadow(OLD)  # nothing configured -> still off
    assert p._shadow is None


def test_env_overrides_config_and_empty_forces_off(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {"shadow_bank_id": "from_cfg"}
    monkeypatch.delenv("PROSPECTA_SHADOW_BANK", raising=False)
    assert p._resolve_shadow_bank_id() == "from_cfg"
    monkeypatch.setenv("PROSPECTA_SHADOW_BANK", "from_env")
    assert p._resolve_shadow_bank_id() == "from_env"
    monkeypatch.setenv("PROSPECTA_SHADOW_BANK", "")
    assert p._resolve_shadow_bank_id() == ""


def test_result_is_exactly_the_old_one(plugin_module):
    off = _provider(plugin_module, shadow=False)
    on = _provider(plugin_module)
    for tool, args in (("prospecta_search", {"query": "x"}), ("prospecta_recall", {"query": "x"})):
        assert on.handle_tool_call(tool, args) == off.handle_tool_call(tool, args)
    assert on._memory.calls == off._memory.calls == [("search", "x"), ("ask", "x")]
    assert on._memory.items == off._memory.items
    _drain(on)
    # search old/new, recall old/new: the shadow side never carries an answer
    assert [e["synthesis"] for e in _pairs(on)] == [None, None, "old answer [1]", None]


def test_search_pair_recorded(plugin_module):
    p = _provider(plugin_module)
    p.handle_tool_call("prospecta_search", {"query": "x", "limit": 3, "depth": "deep"})
    _drain(p)
    old, new = _pairs(p)
    assert (old["bank_id"], new["bank_id"]) == (OLD, NEW)
    assert p._new.searches == [("x", {"limit": 3, "depth": "deep"})]       # the library's current signature: no mode
    so, sn = old["trace"]["shadow"], new["trace"]["shadow"]
    assert so["id"] == sn["id"] and (so["role"], sn["role"]) == ("old", "new")
    assert (so["peer_bank"], sn["peer_bank"]) == (NEW, OLD)
    assert so["tool"] == "prospecta_search" and so["shadow_mode"] == "search" and "kw_n" not in so
    assert so["old_origin_document_ids"] == ["a", "b"] and so["new_origin_document_ids"] == ["a", "c"]
    assert so["jaccard"] == pytest.approx(1 / 3)
    assert isinstance(so["old_ms"], int) and isinstance(so["new_ms"], int)
    assert old["duration_ms"] == so["old_ms"] and new["duration_ms"] == so["new_ms"]
    assert old["mode"] == new["mode"] == "hybrid" and old["depth"] == "deep"      # recall_events.mode is NOT NULL: always hybrid
    assert [r["source"] for r in new["results"]] == ["a", "c"]
    assert p._new.closed


def test_recall_pair_runs_the_same_question_on_the_shadow_bank_retrieval_only(plugin_module):
    p = _provider(plugin_module)
    p.handle_tool_call("prospecta_recall", {"query": "what?"})
    _drain(p)
    old, new = _pairs(p)
    assert old["synthesis"] == "old answer [1]" and new["synthesis"] is None     # no model call on the shadow side
    assert old["trace"]["shadow"]["tool"] == "prospecta_recall"
    assert [q for q, _ in p._new.searches] == ["what?"]
    assert [r["source"] for r in old["results"]] == ["a", "b"] and [r["source"] for r in new["results"]] == ["a", "c"]


def test_the_shadow_has_no_full_mode_and_nothing_is_shadowed_but_search_and_recall(plugin_module, monkeypatch):
    """``shadow_mode: full`` re-ran recall_synth on the shadow bank; the library has no recall_synth now."""
    assert "shadow_mode" not in {f["key"] for f in plugin_module.ProspectaProvider().get_config_schema()}
    assert not hasattr(plugin_module.ProspectaProvider, "_prefetch_synth")
    import inspect
    assert "mode" not in inspect.signature(plugin_module._ShadowReader.__init__).parameters
    monkeypatch.setenv("PROSPECTA_PREFETCH", "1")
    p = _provider(plugin_module)
    assert p.prefetch("hi") == "" and p._memory.calls == []         # the old synthesised prefetch is gone
    _drain(p)
    assert _pairs(p) == []


def test_a_stale_shadow_mode_full_starts_the_search_shadow_with_one_warning(plugin_module, monkeypatch, caplog):
    monkeypatch.delenv("PROSPECTA_SHADOW_BANK", raising=False)
    p = plugin_module.ProspectaProvider()
    p._config = {"shadow_bank_id": NEW, "shadow_mode": "full"}
    p._memory, p._llm, p._database_url = OldMemory(), object(), "postgresql://x"
    monkeypatch.setattr(p, "_build_shadow_embedder", lambda m, d: stub_embed)
    new = NewMemory()
    monkeypatch.setattr(plugin_module, "_build_memory", lambda *a, **k: new)
    with caplog.at_level("WARNING"):
        p._start_shadow(OLD)
    assert p._shadow is not None and p._shadow._shadow is new
    warned = [r.getMessage() for r in caplog.records if r.levelno >= 30]
    assert len(warned) == 1 and "shadow_mode" in warned[0] and "full" in warned[0]
    p._shadow.stop()
    p._config = {"shadow_bank_id": NEW, "shadow_mode": "search"}      # the other value is read silently
    caplog.clear()
    with caplog.at_level("WARNING"):
        p._start_shadow(OLD)
    assert p._shadow is not None and not [r for r in caplog.records if r.levelno >= 30]
    p._shadow.stop()


def test_shadow_failure_does_not_affect_result_and_is_recorded(plugin_module, caplog):
    off = _provider(plugin_module, shadow=False)
    on = _provider(plugin_module, new=NewMemory(fail=RuntimeError("embedder exploded: details " * 50)))
    with caplog.at_level("ERROR"):
        got = on.handle_tool_call("prospecta_search", {"query": "x"})
        _drain(on)
    assert got == off.handle_tool_call("prospecta_search", {"query": "x"})
    _, new = _pairs(on)
    assert new["trace"]["shadow"]["error"].startswith("RuntimeError: embedder exploded")
    assert new["trace"]["shadow"]["error"].endswith("details ")  # full text, not truncated
    assert "embedder exploded" in caplog.text


def test_tracer_failure_is_contained(plugin_module, caplog):
    p = _provider(plugin_module)
    p._memory._tracer = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down"))
    got = p.handle_tool_call("prospecta_search", {"query": "x"})
    _drain(p)
    assert json.loads(got)["results"]
    assert "db down" in caplog.text


def test_old_path_does_not_wait_for_the_shadow(plugin_module):
    p = _provider(plugin_module, new=NewMemory(delay=1.5))
    t0 = time.monotonic()
    p.handle_tool_call("prospecta_search", {"query": "x"})
    p.handle_tool_call("prospecta_recall", {"query": "x"})
    assert time.monotonic() - t0 < 0.5
    assert p._shadow._thread.is_alive() and p._shadow._thread.daemon
    assert p._memory.events == []  # shadow still running in the background
    _drain(p)
    assert len(p._memory.events) == 4


def test_full_queue_drops_loudly_without_blocking(plugin_module, caplog):
    gate = threading.Event()
    new = NewMemory()
    new.search = lambda q, **kw: gate.wait(10) and []
    p = _provider(plugin_module, new=new)
    t0 = time.monotonic()
    with caplog.at_level("ERROR"):
        for _ in range(plugin_module._SHADOW_QUEUE_MAX + 5):
            p.handle_tool_call("prospecta_search", {"query": "x"})
    assert time.monotonic() - t0 < 0.5
    assert "queue full" in caplog.text
    gate.set()
    _drain(p)


def test_shadow_is_created_once_and_closed_with_the_provider(plugin_module):
    p = _provider(plugin_module)
    shadow = p._shadow
    for _ in range(3):
        p.handle_tool_call("prospecta_search", {"query": "x"})
    assert p._shadow is shadow
    old = p._memory
    old.shutdown = lambda: None
    p.shutdown()
    assert p._new.closed and p._shadow is None and not shadow._thread.is_alive()
    assert len(old.events) == 6  # drained before closing


def test_shadow_start_failure_leaves_it_off(plugin_module, monkeypatch, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"shadow_bank_id": "new"}
    p._database_url = "postgresql://x"
    monkeypatch.delenv("PROSPECTA_SHADOW_BANK", raising=False)
    monkeypatch.setattr(p, "_build_shadow_embedder", lambda m, d: (_ for _ in ()).throw(RuntimeError("no key")))
    p._start_shadow("old")
    assert p._shadow is None and "no key" in caplog.text
    p._config = {"shadow_bank_id": "old"}
    p._start_shadow("old")  # same as the live bank
    assert p._shadow is None


def test_embedder_defaults(plugin_module, monkeypatch):
    got = {}
    import prospecta.defaults as d
    monkeypatch.setattr(d, "make_default_embedder", lambda m, dimensions=None: got.update(m=m, d=dimensions))
    plugin_module.ProspectaProvider()._build_shadow_embedder(
        plugin_module._SHADOW_DEFAULT_EMBED_MODEL, 1536)
    assert got == {"m": "openrouter/openai/text-embedding-3-large", "d": 1536}


# ---- real Postgres: the pair lands in recall_events and the library can compare it ----

def test_pair_is_stored_in_recall_events(plugin_module, pg_url, hermes_home, monkeypatch):
    import psycopg
    from prospecta.memory import Memory

    monkeypatch.setenv("DATABASE_URL", pg_url)
    monkeypatch.delenv("PROSPECTA_SHADOW_BANK", raising=False)
    (hermes_home / "prospecta.json").write_text(json.dumps({
        "bank_id": "live", "embedding_dim": "32",
        "shadow_bank_id": "live_v2", "shadow_embedding_dim": "32"}))
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: stub_llm)
    monkeypatch.setattr(p, "_build_shadow_embedder", lambda m, d: stub_embed)
    # the shadow bank must exist before the provider starts
    from prospecta.db.migrate import run_migrations
    run_migrations(database_url=pg_url)
    new_mem = Memory(database_url=pg_url, embed=stub_embed, bank_id="live_v2")
    new_mem.create_bank("live_v2", embedding_dim=32)
    new_mem.retain("the quick brown fox", source="fox.md")
    p.initialize("s", hermes_home=str(hermes_home))
    p._retain("the quick brown fox", source="fox.md")

    with psycopg.connect(pg_url, autocommit=True) as conn, conn.cursor() as cur:  # as a migrated bank is
        cur.execute(
            "UPDATE documents SET document_metadata = jsonb_build_object('migrated_from_document_id', "
            "(SELECT id::text FROM documents WHERE bank_id = 'live')) WHERE bank_id = 'live_v2'")

    got = json.loads(p.handle_tool_call("prospecta_search", {"query": "quick brown fox"}))
    assert got["results"][0]["source"] == "fox.md"
    p.shutdown()  # drains the shadow worker before closing
    new_mem.shutdown()

    with psycopg.connect(pg_url) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT bank_id, trace -> 'shadow' ->> 'role', (trace -> 'shadow' ->> 'jaccard')::float, "
            "trace -> 'shadow' ->> 'tool', duration_ms, trace -> 'shadow' ->> 'id' "
            "FROM recall_events WHERE trace -> 'shadow' IS NOT NULL ORDER BY bank_id")
        rows = cur.fetchall()
    assert [(r[0], r[1], r[3]) for r in rows] == [
        ("live", "old", "prospecta_search"), ("live_v2", "new", "prospecta_search")]
    assert rows[0][5] == rows[1][5] and rows[0][2] == 1.0
