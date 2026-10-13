"""A note the bank already holds is skipped before the LLM call."""
from __future__ import annotations

import hashlib
import logging

import pytest

from tests.test_dual_write import FakeMemory


class _Conn:
    def __init__(self, rows, boom):
        self._rows, self._boom, self._row = rows, boom, None

    def execute(self, sql, params):
        if self._boom:
            raise RuntimeError("db down")
        self._row = self._rows.get(params[1])
        return self

    def fetchone(self):
        return self._row


class _Pool:
    def __init__(self, rows, boom):
        self.rows, self.boom = rows, boom

    def connection(self):
        pool = self

        class _Ctx:
            def __enter__(self):
                return _Conn(pool.rows, pool.boom)

            def __exit__(self, *a):
                return False
        return _Ctx()


class CountingMemory(FakeMemory):
    """retain() stands for the LLM call: it is counted."""

    def __init__(self, rows=None, boom=False):
        super().__init__()
        self._pool = _Pool(rows or {}, boom)
        self._default_bank_id = "b"


def _h(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture
def prov(plugin_module, monkeypatch):
    monkeypatch.delenv("PROSPECTA_HASH_PRECHECK", raising=False)

    def make(**kw):
        p = plugin_module.ProspectaProvider()
        p._memory = CountingMemory(**kw)
        p._config = {}
        p._linked_writes = False
        p._dual = None
        return p
    return make


def test_existing_note_skips_the_llm(prov):
    p = prov(rows={_h("x"): ("id-1", "src")})
    assert p._retain("x", source="src") == "id-1"
    assert p._memory.calls == []


def test_new_note_proceeds(prov):
    p = prov()
    assert p._retain("x", source="src") == "doc-1"
    assert len(p._memory.calls) == 1


def test_different_source_skipped_quietly(prov, caplog):
    p = prov(rows={_h("x"): ("id-1", "other")})
    with caplog.at_level(logging.INFO):
        assert p._retain("x", source="src") == ""
    assert p._memory.calls == []
    assert "other" in caplog.text


def test_lookup_failure_falls_through(prov):
    p = prov(boom=True)
    assert p._retain("x", source="src") == "doc-1"
    assert len(p._memory.calls) == 1


def test_switch_off_by_config_and_env(prov, monkeypatch):
    p = prov(rows={_h("x"): ("id-1", "src")})
    p._config = {"hash_precheck": "off"}
    p._retain("x", source="src")
    assert len(p._memory.calls) == 1
    p._config = {}
    monkeypatch.setenv("PROSPECTA_HASH_PRECHECK", "0")
    p._retain("x", source="src")
    assert len(p._memory.calls) == 2
