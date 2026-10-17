"""Mode 1A (the contract prefetch) end to end on the library's hybrid search: a real Postgres bank with
the ``hybrid_chunk`` channel (Stage 1) and the ``cross_encoder`` rerank (Stage 2, over a fake OpenRouter
rerank transport), the plugin's real provider, a fake decision model and a fake question writer. Invented
text only; no network and no key."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date

import psycopg
import pytest

from tests.conftest import FakeRerank
from tests.test_prefetch_contract import FakeJev, FakeWriter, jev_decision, questions_reply

TODAY = date(2026, 10, 17)
ROOF_Q = "What did we decide about the Henley branch roof?"
NEW_ROOF = "The Henley branch roof will be replaced in June, the board agreed."
OLD_ROOF = "The Henley branch roof leaks; the board will look at it in July."


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_DECISION_MODEL", "PROSPECTA_RERANKER_MODEL",
              "PROSPECTA_QUERY_WRITER_MODEL", "PROSPECTA_RECALL_STAGE3", "PROSPECTA_TIMEZONE", "PROSPECTA_LINKED_WRITES",
              "PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_SHADOW_BANK"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def make(hybrid_bank):
    """A provider with the contract prefetch on, a fake decision model and a fake question writer."""

    def build(*, config=None, decision=None, writer=None, **kw):
        cfg = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract"}}}
        cfg.update(config or {})
        p = hybrid_bank(config=cfg, **kw)
        p._today = lambda: TODAY
        p.jev = FakeJev(decision=decision or jev_decision("new_ask", 0.9, stands=0.95))
        p.writer = writer or FakeWriter(questions_reply(ROOF_Q))
        p._jev_transport, p._writer_llm = p.jev, p.writer
        return p

    return build


def note(p, text, source, day=None, **kw):
    return p._retain(text, source=source, **({"metadata": {"created_on": day}} if day else {}), **kw)


def rows(pg_url, sql, *args):
    with psycopg.connect(pg_url) as c:
        return c.execute(sql, args).fetchall()


def turn(p, user, assistant, sid="s1"):
    p.sync_turn(user, assistant, session_id=sid)
    p._sync_thread.join(timeout=10)
    time.sleep(0.003)


# ---------------------------------------------------------------------------- the gate

def test_gate_false_searches_nothing_and_injects_nothing(make):
    p = make(decision=jev_decision("answer", 0.05))
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    assert p.prefetch("no, the second option please") == ""
    assert len(p.jev.requests) == 1 and not p.rerank.requests and not p.writer.prompts


# ---------------------------------------------------------------------------- Stage 1 + Stage 2 + the cited block

def test_the_block_is_the_librarys_cited_evidence_from_stage_1_and_stage_2(make):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    note(p, "The village library opens at nine on weekdays.", "note:hours", "2026-10-11")
    out = p.prefetch(ROOF_Q)
    assert out.startswith("Background from memory, gathered for this turn.")
    assert f"Asked: {ROOF_Q}" in out
    header = [ln for ln in out.split("\n") if ln.startswith("[1] ")]       # kind is left out: these notes have none
    assert len(header) == 1 and re.fullmatch(r"\[1\] source=note:roof \| date=2026-10-12 \| match=semantic", header[0])
    assert f"\n    {NEW_ROOF}" in out                                       # the whole chunk under its header
    assert "village library" not in out                                      # under Stage 2's floor: cut
    assert p.rerank.requests and p.rerank.requests[0]["model"] == "qwen/qwen3-reranker-8b"
    assert p._last_brief.ranked is True and p._last_brief.injected >= 1


def test_the_exact_phrase_match_carries_its_snippet_in_the_block(make):
    p = make(decision=jev_decision("new_ask", 0.9, stands=0.95))
    note(p, "We watched Electric Dreams at the party and everyone cheered.", "note:party", "2026-10-12")
    note(p, "Dreams are electric when the lights go out on the hill.", "note:scattered", "2026-10-12")
    out = p.prefetch('What did we say about "electric dreams" at the party?')
    assert 'Asked: "electric dreams"' in out                                  # the phrase is asked on its own too
    assert 'Exact match snippet: "' in out and "<<Electric>> <<Dreams>>" in out
    party = [ln for ln in out.split("\n") if ln.startswith("[") and "source=note:party" in ln]
    assert len(party) == 1 and "match=exact" in party[0]


def test_a_missing_reranker_key_falls_back_to_stage_1_order_with_a_note_and_the_first_page(make):
    p = make(rerank=FakeRerank(fail=RuntimeError("HTTP 503")))
    for i in range(8):
        note(p, f"The Henley branch roof update number {i}: the board discussed it.", f"note:r{i}", "2026-10-12")
    out = p.prefetch(ROOF_Q)
    assert "The reranker could not rank these; they are in the search's own order." in out
    assert p._last_brief.ranked is False and p._last_brief.injected == 5      # the first page, in Stage 1's order
    assert p.rerank.requests                                                   # it was tried


# ---------------------------------------------------------------------------- excluded_sources and the date window reach SQL

def test_the_threads_last_two_turns_are_excluded_in_sql_and_an_older_turn_is_not(make):
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1),
             writer=FakeWriter(questions_reply("What was said about the Henley roof?")))
    turn(p, "about the Henley roof tell me one", "Reply one: the Henley roof leak was reported.")
    turn(p, "about the Henley roof tell me two", "Reply two: the Henley roof contractor was named.")
    turn(p, "about the Henley roof tell me three", "Reply three: the Henley roof budget was set.")
    out = p.prefetch("and the roof?")
    assert "Reply one" in out                                       # older: still eligible
    assert "Reply two" not in out and "Reply three" not in out     # the last two: never returned
    # and the library was asked in SQL, not filtered after the fact: the reranker never saw them
    docs = " ".join(d for r in p.rerank.requests for d in r["documents"])
    assert "Reply two" not in docs and "Reply three" not in docs


def test_the_date_window_of_a_relative_time_scope_is_a_stage_1_filter(make):
    p = make(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"))
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    note(p, OLD_ROOF, "note:old", "2026-07-14")
    out = p.prefetch(ROOF_Q)
    assert "source=note:new" in out and "note:old" not in out
    assert "Only notes dated 2026-10-10 to 2026-10-17 were searched." in out
    assert not any("leaks" in d for r in p.rerank.requests for d in r["documents"])


def test_a_named_period_uses_the_writers_window(make):
    writer = FakeWriter(json.dumps({"questions": [ROOF_Q], "date_from": "2026-07-01", "date_to": "2026-07-31"}))
    p = make(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="named"), writer=writer)
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    note(p, OLD_ROOF, "note:old", "2026-07-14")
    out = p.prefetch("What did we decide in July about the Henley branch roof?")
    assert "source=note:old" in out and "note:new" not in out


def test_a_quoted_phrase_the_writer_dropped_still_matches_exactly(make):
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1),
             writer=FakeWriter(questions_reply("What was said at the party?")))   # the writer dropped the phrase
    note(p, "We watched Electric Dreams at the party and everyone cheered.", "note:party", "2026-10-12")
    out = p.prefetch('and what about "electric dreams"?')
    assert 'Asked: What was said at the party? "electric dreams"' in out      # put back by the guard
    assert "<<Electric>> <<Dreams>>" in out


def test_a_follow_up_is_resolved_from_the_agents_last_line(make):
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1),
             writer=FakeWriter(questions_reply("Who did I see the Tuesday cafe visit with?")))
    note(p, "On the Tuesday cafe visit I met Ana and her brother.", "note:cafe", "2026-10-12")
    turn(p, "tell me about my week", "We met Ana at the Tuesday cafe visit.")
    out = p.prefetch("do you remember who i saw it with?")
    assert "The agent's last line: We met Ana at the Tuesday cafe visit." in p.writer.prompts[0]
    assert "Asked: Who did I see the Tuesday cafe visit with?" in out and "met Ana and her brother" in out


# ---------------------------------------------------------------------------- never fails a turn, logs nothing

def test_a_bank_without_the_hybrid_channel_gives_the_not_consulted_line_never_an_error(make):
    p = make(hybrid=False, decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"))
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    out = p.prefetch(ROOF_Q)                       # the window needs the hybrid_chunk channel: the library refuses
    assert out == "(memory was not consulted: the search failed)"


def test_a_successful_turn_logs_nothing_and_records_no_recall_event(make, pg_url, caplog):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    before = rows(pg_url, "SELECT count(*) FROM recall_events")[0][0]
    caplog.clear()                      # setup logged the bank check (the bank was configured after it started)
    with caplog.at_level(logging.DEBUG):
        out = p.prefetch(ROOF_Q)
    assert NEW_ROOF in out
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert not [r for r in caplog.records if "Henley" in r.getMessage()]       # no question or note text, ever
    assert rows(pg_url, "SELECT count(*) FROM recall_events")[0][0] == before   # no recall logging


def test_prefetch_never_raises_even_when_the_library_does(make, monkeypatch, caplog):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")

    def boom(*a, **k):
        raise RuntimeError("secret words: the Panama deal")
    monkeypatch.setattr(p._memory, "search", boom)
    with caplog.at_level(logging.INFO):
        out = p.prefetch(ROOF_Q)
    assert out == "(memory was not consulted: the search failed)" and "Panama" not in caplog.text


def test_the_models_reach_the_library_and_the_transport(make):
    p = make(config={"reranker_model": "x/reranker-cfg", "decision_model": "typesafe/jev-9"})
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    p.prefetch(ROOF_Q)
    assert p.rerank.requests[0]["model"] == "x/reranker-cfg"
    assert p.jev.decisions()[0]["model"] == "typesafe/jev-9"
    assert p._memory._reranker_model == "x/reranker-cfg" and p._memory._decision_model == "typesafe/jev-9"


def test_after_a_restart_the_last_two_turns_are_read_from_the_bank(make, plugin_module, pg_url, hermes_home, monkeypatch):
    p = make()
    for i in (1, 2, 3):
        turn(p, f"user line {i}", f"assistant reply {i}")
    sources = [r[0] for r in rows(pg_url, "SELECT source FROM documents ORDER BY created_at, source")]
    assert len(sources) == 3
    fresh = plugin_module.ProspectaProvider()
    fresh._memory = p._memory
    assert fresh._read_turn_sources("s1") == sources[-2:]          # oldest first, the last two
    assert fresh._read_turn_sources("s1x") == [] and fresh._read_turn_sources("s_") == []   # an exact prefix, no wildcard
