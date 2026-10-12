"""Linked writes and dual write: a second bank kept current off the turn's path."""
from __future__ import annotations

import hashlib
import json
import threading
import time

import pytest

from tests.conftest import stub_embed, stub_llm

MAIN, SECOND = "main_bank", "second_bank"


class _Conn:
    def __init__(self, hashes):
        self._hashes = hashes
        self._row = None

    def execute(self, sql, params):
        self._row = (1,) if params[1] in self._hashes else None
        return self

    def fetchone(self):
        return self._row


class _Pool:
    def __init__(self, hashes):
        self._hashes = hashes

    def connection(self):
        pool = self

        class _Ctx:
            def __enter__(self):
                return _Conn(pool._hashes)

            def __exit__(self, *a):
                return False
        return _Ctx()


class FakeMemory:
    """Stands in for a Memory: records retains, knows which content hashes it holds."""

    def __init__(self, *, fail=None, delay=0.0):
        self.calls, self.hashes, self.closed = [], set(), False
        self.fail, self.delay = fail, delay
        self._pool = _Pool(self.hashes)

    def retain(self, content, **kw):
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        self.calls.append((content, kw))
        self.hashes.add(hashlib.sha256(content.encode("utf-8")).hexdigest())
        return f"doc-{len(self.calls)}"

    def close(self):
        self.closed = True


def _provider(plugin_module, tmp_path, *, dual=True, linked=False, second=None, main=None):
    p = plugin_module.ProspectaProvider()
    p._memory = main or FakeMemory()
    p._llm = None  # no-LLM mode: sync_turn supplies its own index_text
    p._hermes_home = tmp_path
    p._linked_writes = linked
    p.retry = tmp_path / "retry.jsonl"
    p.second = second or FakeMemory()
    if dual:
        p._dual = plugin_module._DualWriter(p.second, bank_id=SECOND, retry_path=p.retry)
    return p


def _turn(p, user="u", assistant="a"):
    p.sync_turn(user, assistant, session_id="s")
    p._sync_thread.join(timeout=5)


def _drain(p):
    dual, p._dual = p._dual, None
    dual.stop()


def _retry_lines(p):
    return [json.loads(x) for x in p.retry.read_text().splitlines()] if p.retry.exists() else []


# ---- off by default ----

def test_both_off_by_default_changes_nothing(plugin_module, tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DUAL_WRITE_BANK", raising=False)
    monkeypatch.delenv("PROSPECTA_LINKED_WRITES", raising=False)
    import prospecta.memory
    monkeypatch.setattr(prospecta.memory, "Memory",
                        lambda **kw: pytest.fail("a second Memory was built"))
    before = threading.active_count()
    p = _provider(plugin_module, tmp_path, dual=False)
    p._config = {}
    assert p._resolve_linked_writes() is False and p._resolve_dual_write_bank_id() == ""
    p._start_dual_write(MAIN)
    assert p._dual is None and threading.active_count() == before
    _turn(p)
    (content, kw), = p._memory.calls
    assert "child_chunks" not in kw and kw["tags"] == ["conversation"]
    assert not p.retry.exists()


def test_settings_resolve_from_config_and_env(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {"linked_writes": "true", "dual_write_bank_id": "from_cfg"}
    monkeypatch.delenv("PROSPECTA_DUAL_WRITE_BANK", raising=False)
    monkeypatch.delenv("PROSPECTA_LINKED_WRITES", raising=False)
    assert p._resolve_linked_writes() and p._resolve_dual_write_bank_id() == "from_cfg"
    p._config = {"linked_writes": False}
    assert not p._resolve_linked_writes()
    p._config = {"linked_writes": "false", "dual_write_bank_id": "from_cfg"}
    monkeypatch.setenv("PROSPECTA_LINKED_WRITES", "1")
    monkeypatch.setenv("PROSPECTA_DUAL_WRITE_BANK", "from_env")
    assert p._resolve_linked_writes() and p._resolve_dual_write_bank_id() == "from_env"
    monkeypatch.setenv("PROSPECTA_DUAL_WRITE_BANK", "")
    assert p._resolve_dual_write_bank_id() == ""
    keys = {f["key"] for f in p.get_config_schema()}
    assert {"linked_writes", "dual_write_bank_id", "dual_write_embed_model",
            "dual_write_embedding_dim", "dual_write_retry_path"} <= keys


# ---- on ----

def test_second_bank_receives_the_same_note(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path)
    _turn(p, "hello", "world")
    out = json.loads(p.handle_tool_call(
        "prospecta_retain", {"content": "a fact", "source": "fact:1", "tags": ["x"], "index_text": ["q?"]}))
    assert out["document_id"] == "doc-2"
    _drain(p)
    assert [(c, {k: v for k, v in kw.items() if k != "source"}) for c, kw in p.second.calls] == [
        (c, {**{k: v for k, v in kw.items() if k != "source"}, "child_chunks": True})
        for c, kw in p._memory.calls]
    assert [kw["source"] for _, kw in p.second.calls] == [kw["source"] for _, kw in p._memory.calls]
    assert p.second.closed and not p.retry.exists()


def test_the_turn_does_not_wait_for_the_second_write(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path, second=FakeMemory(delay=1.5))
    t0 = time.monotonic()
    _turn(p)
    p.handle_tool_call("prospecta_retain", {"content": "c", "source": "s"})
    assert time.monotonic() - t0 < 0.5
    assert p._dual._thread.daemon
    _drain(p)
    assert len(p.second.calls) == 2


def test_failed_second_write_leaves_the_turn_alone_and_lands_in_the_retry_file(
        plugin_module, tmp_path, caplog):
    off = _provider(plugin_module, tmp_path, dual=False)
    p = _provider(plugin_module, tmp_path, second=FakeMemory(fail=RuntimeError("bank down: details " * 40)))
    with caplog.at_level("ERROR"):
        got = p.handle_tool_call("prospecta_retain", {"content": "c", "source": "s"})
        _drain(p)
    assert got == off.handle_tool_call("prospecta_retain", {"content": "c", "source": "s"})
    (line,) = _retry_lines(p)
    assert (line["content"], line["source"]) == ("c", "s")
    assert line["error"].startswith("RuntimeError: bank down") and line["error"].endswith("details ")
    assert "bank down" in caplog.text and "Traceback" in caplog.text


def test_retry_file_replays_once(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path, dual=False)
    p.retry.write_text("\n".join(json.dumps(j) for j in (
        {"content": "one", "source": "s1", "error": "x", "failed_at": 1},
        {"content": "two", "source": "s2", "tags": ["t"]})) + "\nnot json\n")
    w = plugin_module._DualWriter(p.second, bank_id=SECOND, retry_path=p.retry)
    w.stop()
    assert [(c, kw["source"]) for c, kw in p.second.calls] == [("one", "s1"), ("two", "s2")]
    assert not p.retry.exists() and not p.retry.with_name("retry.jsonl.replaying").exists()
    w = plugin_module._DualWriter(p.second, bank_id=SECOND, retry_path=p.retry)  # next initialize
    w.stop()
    assert len(p.second.calls) == 2


def test_replay_failure_stays_in_the_retry_file(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path, dual=False, second=FakeMemory(fail=RuntimeError("still down")))
    p.retry.write_text(json.dumps({"content": "one", "source": "s1"}) + "\n")
    plugin_module._DualWriter(p.second, bank_id=SECOND, retry_path=p.retry).stop()
    assert [l["content"] for l in _retry_lines(p)] == ["one"]


def test_same_content_hash_is_a_no_op(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path)
    p.second.hashes.add(hashlib.sha256(b"already copied").hexdigest())  # the catch-up copy got there first
    for _ in range(2):
        p.handle_tool_call("prospecta_retain", {"content": "already copied", "source": "s"})
        p.handle_tool_call("prospecta_retain", {"content": "new note", "source": "s2"})
    _drain(p)
    assert [c for c, _ in p.second.calls] == ["new note"]
    assert not p.retry.exists()


def test_full_queue_goes_to_the_retry_file_without_blocking(plugin_module, tmp_path, caplog):
    p = _provider(plugin_module, tmp_path, second=FakeMemory(delay=0.4))
    t0 = time.monotonic()
    n = plugin_module._SHADOW_QUEUE_MAX + 5
    with caplog.at_level("ERROR"):
        for i in range(n):
            p.handle_tool_call("prospecta_retain", {"content": f"c{i}", "source": f"s{i}"})
    assert time.monotonic() - t0 < 0.5
    assert "queue full" in caplog.text
    overflow = len(_retry_lines(p))
    assert overflow >= 1
    p._dual._queue.queue.clear()  # skip the slow remainder
    _drain(p)


def test_shutdown_drains_then_closes(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path)
    dual = p._dual
    for i in range(3):
        p.handle_tool_call("prospecta_retain", {"content": f"c{i}", "source": f"s{i}"})
    p._memory.shutdown = lambda: None
    p.shutdown()
    assert len(p.second.calls) == 3 and p.second.closed and p._dual is None
    assert not dual._thread.is_alive()


def test_dual_start_failures_leave_it_off(plugin_module, tmp_path, monkeypatch, caplog):
    p = plugin_module.ProspectaProvider()
    p._hermes_home, p._database_url = tmp_path, "postgresql://x"
    monkeypatch.delenv("PROSPECTA_DUAL_WRITE_BANK", raising=False)
    p._config = {"dual_write_bank_id": "new"}
    monkeypatch.setattr(p, "_build_shadow_embedder", lambda m, d: (_ for _ in ()).throw(RuntimeError("no key")))
    p._start_dual_write("old")
    assert p._dual is None and "no key" in caplog.text
    p._config = {"dual_write_bank_id": "old"}
    p._start_dual_write("old")  # same as the live bank
    assert p._dual is None


# ---- linked writes ----

def test_linked_writes_asks_for_chunk_children(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path, dual=False, linked=True)
    _turn(p)
    assert p._memory.calls[0][1]["child_chunks"] is True


def test_stage_kwargs_come_from_the_library_cli(plugin_module, monkeypatch):
    import prospecta.cli._common as common
    monkeypatch.setattr(common, "_stage_kwargs", lambda: {"linker": "L", "jev": "J"})
    assert plugin_module.ProspectaProvider()._stage_kwargs() == {"linker": "L", "jev": "J"}


def test_linker_build_failure_is_logged_and_never_raised(plugin_module, monkeypatch, caplog):
    import prospecta.cli._common as common
    monkeypatch.setattr(common, "_stage_kwargs", lambda: (_ for _ in ()).throw(RuntimeError("no litellm")))
    assert plugin_module.ProspectaProvider()._stage_kwargs() == {}
    assert "no litellm" in caplog.text and "Traceback" in caplog.text


def test_failing_main_write_neither_raises_nor_queues(plugin_module, tmp_path, caplog):
    p = _provider(plugin_module, tmp_path, linked=True, main=FakeMemory(fail=RuntimeError("main down")))
    _turn(p)
    assert json.loads(p.handle_tool_call(
        "prospecta_retain", {"content": "c", "source": "s"}))["error"] == "main down"
    _drain(p)
    assert p.second.calls == [] and "main down" in caplog.text


# ---- real Postgres ----

class _FailingLinker:
    asynchronous = True

    def link_document(self, conn, bank_id, document_id, calls=None):
        raise RuntimeError("linker exploded")


def _real_provider(plugin_module, pg_url, hermes_home, monkeypatch, cfg):
    from prospecta.db.migrate import run_migrations
    from prospecta.memory import Memory
    monkeypatch.setenv("DATABASE_URL", pg_url)
    for var in ("PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_LINKED_WRITES"):
        monkeypatch.delenv(var, raising=False)
    (hermes_home / "prospecta.json").write_text(json.dumps(
        {"bank_id": "live", "embedding_dim": "32", **cfg}))
    run_migrations(database_url=pg_url)
    second = Memory(database_url=pg_url, embed=stub_embed, bank_id=SECOND)
    second.create_bank(SECOND, embedding_dim=32)
    second.close()
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: stub_llm)
    monkeypatch.setattr(p, "_build_shadow_embedder", lambda m, d: stub_embed)
    return p


def _count(pg_url, bank, kind=None):
    import psycopg
    with psycopg.connect(pg_url) as conn:
        docs = conn.execute("SELECT count(*) FROM documents WHERE bank_id = %s", (bank,)).fetchone()[0]
        items = conn.execute(
            "SELECT count(*) FROM memory_items WHERE bank_id = %s AND kind = %s",
            (bank, kind or "question")).fetchone()[0]
    return docs, items


def test_real_banks_dual_write_with_chunks_and_is_idempotent(
        plugin_module, pg_url, hermes_home, monkeypatch):
    p = _real_provider(plugin_module, pg_url, hermes_home, monkeypatch, {
        "linked_writes": "true", "dual_write_bank_id": SECOND, "dual_write_embedding_dim": "32"})
    monkeypatch.setattr(p, "_stage_kwargs", lambda: {})
    p.initialize("s", hermes_home=str(hermes_home))
    p.handle_tool_call("prospecta_retain", {"content": "Paris is in France.\n\nIt is large.", "source": "fact:1"})
    p.handle_tool_call("prospecta_retain", {"content": "Paris is in France.\n\nIt is large.", "source": "fact:1"})
    p.shutdown()
    assert _count(pg_url, "live") == (1, _count(pg_url, "live")[1])
    assert _count(pg_url, "live", "chunk")[1] >= 1  # linked_writes: chunk children on the main bank
    assert _count(pg_url, SECOND) == (1, _count(pg_url, "live")[1])
    assert _count(pg_url, SECOND, "chunk") == _count(pg_url, "live", "chunk")


def test_real_linker_failure_never_fails_the_turn(plugin_module, pg_url, hermes_home, monkeypatch, caplog):
    p = _real_provider(plugin_module, pg_url, hermes_home, monkeypatch, {"linked_writes": True})
    monkeypatch.setattr(p, "_stage_kwargs", lambda: {"linker": _FailingLinker()})
    p.initialize("s", hermes_home=str(hermes_home))
    out = json.loads(p.handle_tool_call("prospecta_retain", {"content": "a note", "source": "n"}))
    p.shutdown()
    assert "document_id" in out and "linker exploded" in caplog.text
    assert _count(pg_url, "live")[0] == 1


def test_real_missing_dual_bank_leaves_it_off(plugin_module, pg_url, hermes_home, monkeypatch, caplog):
    p = _real_provider(plugin_module, pg_url, hermes_home, monkeypatch, {"dual_write_bank_id": "nope"})
    monkeypatch.setattr(p, "_stage_kwargs", lambda: {})
    p.initialize("s", hermes_home=str(hermes_home))
    assert p._dual is None and "No such bank" in caplog.text
    p.shutdown()
