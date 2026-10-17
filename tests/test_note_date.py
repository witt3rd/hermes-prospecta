"""The note's date: the library stamps it (documents.created_on) in the profile's timezone."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import psycopg
import pytest

from tests.conftest import CountingLLM, stub_embed
from tests.test_dual_write import _drain, _provider, _turn

TZ_LA = "America/Los_Angeles"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_NOTE_DATE", "PROSPECTA_TIMEZONE", "PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_LINKED_WRITES"):
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


# ---- the plugin no longer stamps; it hands the library its timezone ----

def test_notes_are_passed_on_unstamped_the_library_owns_the_date(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path)
    _turn(p)
    p.handle_tool_call("prospecta_retain", {"content": "a fact", "source": "fact:1", "tags": ["x"]})
    _drain(p)
    assert len(p._memory.calls) == 2 and len(p.second.calls) == 2
    for bank in (p._memory, p.second):
        for _, kw in bank.calls:
            assert "metadata" not in kw                      # no created_on from the plugin
    (_, turn_kw), (_, tool_kw) = p._memory.calls
    assert set(turn_kw) == {"source", "tags"} and turn_kw["tags"] == ["conversation"]
    assert set(tool_kw) == {"source", "tags"} and tool_kw["tags"] == ["x"]
    for gone in ("_note_dates_enabled", "_note_date_metadata", "_with_note_date"):
        assert not hasattr(p, gone)
    assert "note_date" not in {f["key"] for f in p.get_config_schema()}


def test_a_date_the_caller_gives_is_left_alone(plugin_module, tmp_path):
    p = _provider(plugin_module, tmp_path, dual=False)
    p._retain("c2", source="s2", metadata={"date": "2020-01-02", "person": "Kelly"})
    assert p._memory.calls[0][1]["metadata"] == {"date": "2020-01-02", "person": "Kelly"}


def test_the_profiles_timezone_is_handed_to_the_library(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {"timezone": "Europe/Paris"}
    p._apply_timezone()
    assert os.environ["PROSPECTA_TIMEZONE"] == "Europe/Paris"


def test_without_a_profile_zone_the_environments_zone_stands_else_los_angeles(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._apply_timezone()
    assert os.environ["PROSPECTA_TIMEZONE"] == TZ_LA                  # the plugin's default, not the library's UTC
    monkeypatch.setenv("PROSPECTA_TIMEZONE", "Asia/Tokyo")
    p._apply_timezone()
    assert os.environ["PROSPECTA_TIMEZONE"] == "Asia/Tokyo"
    p._config = {"timezone": "Europe/Paris"}                         # the profile wins over the environment
    p._apply_timezone()
    assert os.environ["PROSPECTA_TIMEZONE"] == "Europe/Paris"


def test_an_unknown_profile_zone_falls_back_loudly(plugin_module, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"timezone": "Not/AZone"}
    with caplog.at_level(logging.WARNING):
        p._apply_timezone()
    assert os.environ["PROSPECTA_TIMEZONE"] == TZ_LA and "Not/AZone" in caplog.text


def test_today_is_the_calendar_day_in_the_profiles_zone(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    seen = []
    monkeypatch.setattr(plugin_module, "local_date_iso", lambda tz=None, now=None: seen.append(tz) or "2026-07-14")
    p._config = {"timezone": "Europe/Paris"}
    assert str(p._today()) == "2026-07-14" and seen == ["Europe/Paris"]


# ---- the library really stamps it, and writes the turn as chunks with no model call ----

def _provider_on_pg(plugin_module, pg_url, hermes_home, monkeypatch, llm, **cfg):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", "dated")
    (hermes_home / "prospecta.json").write_text(json.dumps({"embedding_dim": "32", **cfg}), encoding="utf-8")
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: llm)
    p.initialize(session_id="dated", hermes_home=str(hermes_home))
    return p


def test_library_stamps_created_on_in_the_profile_zone_and_a_turn_costs_no_model_call(
        plugin_module, pg_url, hermes_home, monkeypatch):
    llm = CountingLLM()
    p = _provider_on_pg(plugin_module, pg_url, hermes_home, monkeypatch, llm, timezone="Pacific/Auckland")
    try:
        before = plugin_module.local_date_iso("Pacific/Auckland")
        p.sync_turn("when is the party", "in July", session_id="dated")
        p._sync_thread.join(timeout=10)
        p.handle_tool_call("prospecta_retain", {"content": "the party is on the 20th", "source": "fact:party"})
        p.handle_tool_call("prospecta_retain", {"content": "an old fact", "source": "fact:old",
                                                 "tags": ["x"]})
        p._retain("a dated fact", source="fact:dated", metadata={"created_on": "2020-01-02"})
        after = plugin_module.local_date_iso("Pacific/Auckland")
        with psycopg.connect(pg_url) as c:
            docs = c.execute("SELECT source, created_on::text, person, source_kind, tags FROM documents "
                             "WHERE bank_id = 'dated' ORDER BY source").fetchall()
            kinds = c.execute("SELECT DISTINCT kind FROM memory_items WHERE bank_id = 'dated'").fetchall()
    finally:
        p.shutdown()
    assert llm.calls == 0                                              # zero model calls at write time
    assert kinds == [("chunk",)]                                       # chunks only: no question items
    turn = next(d for d in docs if d[0].startswith("turn:"))
    party = next(d for d in docs if d[0] == "fact:party")
    dated = next(d for d in docs if d[0] == "fact:dated")
    assert turn[1] in {before, after} and turn[4] == ["conversation"]  # the day in Auckland, set by the library
    assert party[1] in {before, after}
    assert dated[1] == "2020-01-02"                                    # a date the caller gave wins
    assert all(d[2] is None and d[3] is None for d in docs)


def test_sync_turn_without_a_model_writes_the_same_chunks(plugin_module, pg_url, hermes_home, monkeypatch):
    p = _provider_on_pg(plugin_module, pg_url, hermes_home, monkeypatch, None)
    try:
        p.sync_turn("Donald: Are you live on Prospecta?", "Augur: Yes, I am live on Prospecta.", session_id="x")
        p._sync_thread.join(timeout=10)
        with psycopg.connect(pg_url) as c:
            rows = c.execute("SELECT kind, original_chunk FROM memory_items WHERE bank_id = 'dated'").fetchall()
    finally:
        p.shutdown()
    assert [r[0] for r in rows] == ["chunk"] and "live on Prospecta" in rows[0][1]
