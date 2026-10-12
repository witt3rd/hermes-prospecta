"""The note's date is saved with new notes (documents.created_on)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import psycopg
import pytest

from tests.conftest import stub_embed
from tests.test_dual_write import FakeMemory, SECOND, _drain, _provider, _turn

TZ_LA = "America/Los_Angeles"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_NOTE_DATE", "PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_LINKED_WRITES"):
        monkeypatch.delenv(k, raising=False)


# ---- the helper ----

def test_local_date_is_the_local_day_not_the_utc_day(plugin_module):
    # 03:30 UTC on 15 July is still the evening of 14 July in Los Angeles.
    now = datetime(2026, 7, 15, 3, 30, tzinfo=timezone.utc)
    assert plugin_module.local_date_iso(TZ_LA, now) == "2026-07-14"
    assert plugin_module.local_date_iso("UTC", now) == "2026-07-15"
    assert plugin_module.local_date_iso("Pacific/Auckland", now) == "2026-07-15"


def test_local_date_follows_daylight_saving(plugin_module):
    # 07:30 UTC is 23:30 the day before in winter (UTC-8) but 00:30 the same day in summer (UTC-7).
    assert plugin_module.local_date_iso(TZ_LA, datetime(2026, 1, 15, 7, 30, tzinfo=timezone.utc)) == "2026-01-14"
    assert plugin_module.local_date_iso(TZ_LA, datetime(2026, 7, 15, 7, 30, tzinfo=timezone.utc)) == "2026-07-15"


def test_local_date_defaults_and_never_raises(plugin_module, caplog):
    now = datetime(2026, 7, 15, 3, 30, tzinfo=timezone.utc)
    assert plugin_module.local_date_iso("", now) == "2026-07-14"
    assert plugin_module.local_date_iso(None, now) == "2026-07-14"
    with caplog.at_level("WARNING"):
        assert plugin_module.local_date_iso("Not/AZone", now) == "2026-07-14"
    assert "Not/AZone" in caplog.text
    assert len(plugin_module.local_date_iso("Nope")) == 10  # the current time, no ``now``


# ---- switch ----

def test_switch_default_config_and_env(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    assert p._note_dates_enabled() is True                       # default: on
    p._linked_writes = True
    assert p._note_dates_enabled() is False                      # ...except with linked writes
    p._config = {"note_date": "true"}
    assert p._note_dates_enabled() is True                       # explicit choice wins
    p._linked_writes = False
    p._config = {"note_date": "false"}
    assert p._note_dates_enabled() is False
    monkeypatch.setenv("PROSPECTA_NOTE_DATE", "on")
    assert p._note_dates_enabled() is True                       # env wins over config
    assert {"timezone", "note_date"} <= {f["key"] for f in p.get_config_schema()}


def test_metadata_uses_the_configured_timezone(plugin_module, monkeypatch):
    seen = []
    monkeypatch.setattr(plugin_module, "local_date_iso", lambda tz=None, now=None: seen.append(tz) or "2026-07-14")
    p = plugin_module.ProspectaProvider()
    assert p._note_date_metadata() == {"created_on": "2026-07-14"}
    p._config = {"timezone": "Europe/Paris"}
    p._note_date_metadata()
    assert seen == ["", "Europe/Paris"]


def test_helper_failure_never_breaks_the_note(plugin_module, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("clock broke")
    monkeypatch.setattr(plugin_module, "local_date_iso", boom)
    p = plugin_module.ProspectaProvider()
    assert p._note_date_metadata() == {}
    assert p._with_note_date({"source": "s"}) == {"source": "s"}


# ---- main write and dual write ----

def _fix_date(plugin_module, monkeypatch, day="2026-07-14"):
    monkeypatch.setattr(plugin_module, "local_date_iso", lambda tz=None, now=None: day)


def test_turn_and_tool_note_carry_the_date_to_both_banks(plugin_module, tmp_path, monkeypatch):
    _fix_date(plugin_module, monkeypatch)
    p = _provider(plugin_module, tmp_path)
    _turn(p)
    p.handle_tool_call("prospecta_retain", {"content": "a fact", "source": "fact:1", "tags": ["x"]})
    _drain(p)
    assert len(p._memory.calls) == 2 and len(p.second.calls) == 2
    for bank in (p._memory, p.second):
        for _, kw in bank.calls:
            assert kw["metadata"] == {"created_on": "2026-07-14"}
    # nothing else about the note changed
    (_, turn_kw), (_, tool_kw) = p._memory.calls
    assert turn_kw["tags"] == ["conversation"] and turn_kw["index_text"] == ["u", "a"]
    assert tool_kw["tags"] == ["x"] and set(tool_kw) == {"source", "tags", "metadata"}


def test_turn_date_is_fixed_when_the_turn_ends(plugin_module, tmp_path, monkeypatch):
    days = iter(["2026-07-14", "2026-07-15"])
    monkeypatch.setattr(plugin_module, "local_date_iso", lambda tz=None, now=None: next(days))
    p = _provider(plugin_module, tmp_path, dual=False)
    _turn(p)
    (_, kw), = p._memory.calls
    assert kw["metadata"] == {"created_on": "2026-07-14"}  # computed once, not again in the thread


def test_off_changes_nothing(plugin_module, tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPECTA_NOTE_DATE", "off")
    p = _provider(plugin_module, tmp_path)
    _turn(p)
    _drain(p)
    for bank in (p._memory, p.second):
        assert all("metadata" not in kw for _, kw in bank.calls)


def test_linked_writes_leave_it_off_unless_asked(plugin_module, tmp_path, monkeypatch):
    p = _provider(plugin_module, tmp_path, dual=False, linked=True)
    _turn(p)
    assert "metadata" not in p._memory.calls[0][1]
    p._config = {"note_date": "true"}
    _turn(p)
    assert p._memory.calls[1][1]["metadata"]["created_on"]
    assert p._memory.calls[1][1]["child_chunks"] is True


def test_existing_metadata_is_kept_and_a_date_in_it_wins(plugin_module, tmp_path, monkeypatch):
    _fix_date(plugin_module, monkeypatch)
    p = _provider(plugin_module, tmp_path, dual=False)
    p._retain("c1", source="s1", metadata={"person": "Kelly"})
    p._retain("c2", source="s2", metadata={"date": "2020-01-02", "person": "Kelly"})
    assert p._memory.calls[0][1]["metadata"] == {"person": "Kelly", "created_on": "2026-07-14"}
    assert p._memory.calls[1][1]["metadata"] == {"date": "2020-01-02", "person": "Kelly"}


def test_a_failed_second_write_keeps_the_date_in_the_retry_file(plugin_module, tmp_path, monkeypatch):
    _fix_date(plugin_module, monkeypatch)
    p = _provider(plugin_module, tmp_path, second=FakeMemory(fail=RuntimeError("down")))
    _turn(p)
    _drain(p)
    (line,) = [json.loads(x) for x in p.retry.read_text().splitlines()]
    assert line["metadata"] == {"created_on": "2026-07-14"}
    p2 = _provider(plugin_module, tmp_path, dual=False)  # replay once the bank is back
    plugin_module._DualWriter(p2.second, bank_id=SECOND, retry_path=p.retry).stop()
    (_, kw), = p2.second.calls
    assert kw["metadata"] == {"created_on": "2026-07-14"}


# ---- the library really stores it ----

def test_library_stores_created_on_and_nothing_else_changes(
        plugin_module, pg_url, hermes_home, monkeypatch):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", "dated")
    (hermes_home / "prospecta.json").write_text(
        json.dumps({"embedding_dim": "32", "timezone": "America/Los_Angeles"}), encoding="utf-8")
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: None)
    monkeypatch.setattr(plugin_module, "local_date_iso", lambda tz=None, now=None: "2026-07-14")
    p.initialize(session_id="dated", hermes_home=str(hermes_home))
    try:
        p.sync_turn("when is the party", "in July", session_id="dated")
        p._sync_thread.join(timeout=10)
        p.handle_tool_call("prospecta_retain",
                           {"content": "the party is on the 20th", "source": "fact:party", "index_text": ["party?"]})
        with psycopg.connect(pg_url) as c:
            rows = c.execute(
                "SELECT source, created_on::text, person, source_kind, tags, document_metadata "
                "FROM documents WHERE bank_id = 'dated' ORDER BY source").fetchall()
    finally:
        p.shutdown()
    assert [r[0].split(":")[0] for r in rows] == ["fact", "turn"]
    for _, created_on, person, kind, tags, meta in rows:
        assert created_on == "2026-07-14" and person is None and kind is None
        assert meta == {"created": "2026-07-14", "created_on": "2026-07-14"}
    assert rows[1][4] == ["conversation"]
