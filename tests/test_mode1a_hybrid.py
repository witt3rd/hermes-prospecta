"""Mode 1 (the contract prefetch) end to end on the real library, offline: a real Postgres bank with the
``hybrid_chunk`` channel (Stage 1) and the ``cross_encoder`` rerank (Stage 2, over a fake OpenRouter rerank
transport), the plugin's real provider and the library's real ``Memory.expand_and_recall`` (pinned at the
merged PR 85 commit), a fake decision model (Jev) and a fake query-writer model. Invented text only; no network
and no key."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import date

import psycopg
import pytest

from tests.conftest import FakeRerank
from tests.test_prefetch_contract import FakeJev, jev_decision

TODAY = date(2026, 10, 17)
ROOF_Q = "What did we decide about the Henley branch roof?"
NEW_ROOF = "The Henley branch roof will be replaced in June, the board agreed."
OLD_ROOF = "The Henley branch roof leaks; the board will look at it in July."
ROOF_SEARCH = "Henley branch roof decision"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_DECISION_MODEL", "PROSPECTA_RERANKER_MODEL",
              "PROSPECTA_QUERY_WRITER_MODEL", "PROSPECTA_RECALL_STAGE3", "PROSPECTA_TIMEZONE", "PROSPECTA_LINKED_WRITES",
              "PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_SHADOW_BANK", "PROSPECTA_JEV_MODEL"):
        monkeypatch.delenv(k, raising=False)


class FakeWriterLLM:
    """The library's query-writer model (``Memory(llm=...)``): called as ``llm(messages=[...], json_mode=True,
    model=<writer model>)``; it records the prompt and the model, and answers a fixed JSON reply."""

    def __init__(self, reply=None, fail=None, delay=0.0):
        self.reply = reply if reply is not None else searches_reply(ROOF_SEARCH)
        self.fail, self.delay = fail, delay
        self.prompts, self.models = [], []

    def __call__(self, messages, *, json_mode=False, model=None):
        self.prompts.append(messages[0]["content"])
        self.models.append(model)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return self.reply


def searches_reply(*searches):
    """The writer's JSON: each search a query (a string) or a dict of query / terms / date_from / date_to."""
    out = []
    for s in searches:
        d = {"query": s} if isinstance(s, str) else dict(s)
        out.append({"terms": [], "date_from": None, "date_to": None, **d})
    return json.dumps({"searches": out})


@pytest.fixture
def make(hybrid_bank):
    """A provider with the contract prefetch on, a fake decision model and a fake query writer."""

    def build(*, config=None, decision=None, writer=None, **kw):
        cfg = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract"}}}
        cfg.update(config or {})
        writer = writer if writer is not None else FakeWriterLLM()
        p = hybrid_bank(config=cfg, llm=writer, **kw)
        p._today = lambda: TODAY
        p.jev = FakeJev(decision=decision or jev_decision("new_ask", 0.9, stands=0.95))
        p.writer = writer
        p._jev_transport = p.jev
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


def reranked_docs(p):
    return " ".join(d for r in p.rerank.requests for d in r["documents"])


# ---------------------------------------------------------------------------- the gate: false means no expansion

def test_gate_false_makes_no_expansion_call_searches_nothing_and_injects_nothing(make):
    p = make(decision=jev_decision("answer", 0.05))
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    assert p.prefetch("no, the second option please") == ""
    assert len(p.jev.requests) == 1                                     # the decision, and nothing else
    assert not p.writer.prompts and not p.rerank.requests               # no writer call, no rerank, no search


def test_the_library_gate_is_not_run_a_second_time(make, monkeypatch):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    seen = {}
    real = p._memory.expand_and_recall

    def spy(prompt, kind="chat", **kw):
        seen.update(kw)
        return real(prompt, kind, **kw)
    monkeypatch.setattr(p._memory, "expand_and_recall", spy)
    p.prefetch(ROOF_Q)
    assert seen["gate"] is False and len(p.jev.requests) == 1


# ---------------------------------------------------------------------------- expansion: the cited block

def test_the_block_lists_the_searches_tells_the_agent_to_cite_and_carries_the_librarys_entries(make):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    note(p, "The village library opens at nine on weekdays.", "note:hours", "2026-10-11")
    out = p.prefetch(ROOF_Q)
    assert out.startswith("Background from memory, gathered for this turn from the searches listed below.")
    assert f"Searches run (an item's header names the searches that found it):\n(1) {ROOF_SEARCH}" in out
    assert "When you use an item below in your response, cite it as [n], its number." in out
    header = [ln for ln in out.split("\n") if ln.startswith("[1] ")]       # kind is left out: these notes have none
    assert len(header) == 1 and re.fullmatch(
        r"\[1\] source=note:roof \| date=2026-10-12 \| match=exact\+semantic \| found_by=1", header[0])
    assert f"\n    {NEW_ROOF}" in out                                       # the whole chunk under its header
    assert "village library" not in out                                      # under Stage 2's floor: cut
    assert "Asked:" not in out
    assert p.rerank.requests and p.rerank.requests[0]["model"] == "qwen/qwen3-reranker-8b"
    assert p._last_brief.ranked is True and p._last_brief.injected >= 1 and p._last_brief.searches == [ROOF_SEARCH]


def test_several_searches_run_in_parallel_each_reranked_on_its_own_query_one_record_once(make):
    writer = FakeWriterLLM(searches_reply(ROOF_SEARCH, "Henley branch roof replacement June",
                                          "village library opening hours"))
    p = make(writer=writer)
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    note(p, "The village library opens at nine on weekdays.", "note:hours", "2026-10-11")
    out = p.prefetch(ROOF_Q)
    queries = sorted(r["query"].split("Query:", 1)[-1].strip() for r in p.rerank.requests)
    assert queries == sorted([ROOF_SEARCH, "Henley branch roof replacement June", "village library opening hours"])
    assert out.count(f"\n    {NEW_ROOF}") == 1                            # found by two searches: once
    roof = [ln for ln in out.split("\n") if "source=note:roof" in ln]
    assert len(roof) == 1 and "found_by=1,2" in roof[0]
    hours = [ln for ln in out.split("\n") if "source=note:hours" in ln]
    assert len(hours) == 1 and "found_by=3" in hours[0]
    for n, q in enumerate([ROOF_SEARCH, "Henley branch roof replacement June", "village library opening hours"], 1):
        assert f"({n}) {q}" in out
    assert len(writer.prompts) == 1 and len(p.jev.requests) == 1          # one writer call, one decision


def test_the_exact_phrase_match_carries_its_snippet_and_the_quoted_phrase_the_writer_dropped_is_a_term(make):
    p = make(writer=FakeWriterLLM(searches_reply("party discussion")))      # the writer dropped the phrase
    note(p, "We watched Electric Dreams at the party and everyone cheered.", "note:party", "2026-10-12")
    note(p, "Dreams are electric when the lights go out on the hill.", "note:scattered", "2026-10-12")
    out = p.prefetch('What did we say about "electric dreams" at the party?')
    assert '(1) party discussion | terms: "electric dreams"' in out        # the library's verbatim guard
    assert 'Exact match snippet: "' in out and "<<Electric>> <<Dreams>>" in out
    party = [ln for ln in out.split("\n") if ln.startswith("[") and "source=note:party" in ln]
    assert len(party) == 1 and "match=exact" in party[0]


def test_nothing_answers_says_which_searches_were_asked(make):
    p = make()
    note(p, "The village library opens at nine on weekdays.", "note:hours", "2026-10-11")
    out = p.prefetch(ROOF_Q)
    assert out == f"Memory was asked: {ROOF_SEARCH}. Nothing in it answers this."
    assert p._last_brief.injected == 0


def test_a_failing_reranker_falls_back_to_stage_1_order_with_a_note(make):
    p = make(rerank=FakeRerank(fail=RuntimeError("HTTP 503")))
    for i in range(3):
        note(p, f"The Henley branch roof update number {i}: the board discussed it.", f"note:r{i}", "2026-10-12")
    out = p.prefetch(ROOF_Q)
    assert "The reranker could not rank these; they are in the search's own order." in out
    assert p._last_brief.ranked is False and p._last_brief.injected >= 1
    assert p.rerank.requests                                                   # it was tried


# ---------------------------------------------------------------------------- follow-ups: agent_last

def test_a_follow_up_carries_the_agents_last_answer_to_the_writer_and_the_decision(make):
    writer = FakeWriterLLM(searches_reply("Tuesday cafe visit companions"))
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1), writer=writer)
    note(p, "On the Tuesday cafe visit I met Ana and her brother.", "note:cafe", "2026-10-12")
    turn(p, "tell me about my week", "We met Ana at the Tuesday cafe visit.")
    out = p.prefetch("do you remember who i saw it with?")
    assert "We met Ana at the Tuesday cafe visit." in writer.prompts[0].split("## The message")[0]
    assert "do you remember who i saw it with?" in writer.prompts[0].split("## The message")[1]
    assert p.jev.decisions()[0]["state"]["agent_last"] == "We met Ana at the Tuesday cafe visit."
    assert "met Ana and her brother" in out


def test_the_first_message_of_a_session_has_no_agent_last_for_the_writer(make):
    writer = FakeWriterLLM()
    p = make(writer=writer)
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    p.prefetch(ROOF_Q)
    assert "The agent's last turn" not in writer.prompts[0]


def test_after_a_restart_the_last_answer_is_recovered_from_the_session_store_for_the_library(make, hermes_home):
    import sqlite3
    c = sqlite3.connect(hermes_home / "state.db")
    c.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL)")
    c.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES "
              "('s1', 'assistant', 'We met Ana at the Tuesday cafe visit.', 1)")
    c.commit()
    c.close()
    writer = FakeWriterLLM(searches_reply("Tuesday cafe visit companions"))
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1), writer=writer)
    note(p, "On the Tuesday cafe visit I met Ana and her brother.", "note:cafe", "2026-10-12")
    out = p.prefetch("do you remember who i saw it with?")                 # no sync_turn yet: a fresh process
    assert "We met Ana at the Tuesday cafe visit." in writer.prompts[0]
    assert "met Ana and her brother" in out


# ---------------------------------------------------------------------------- filters reach every search

def test_the_threads_last_two_turns_are_excluded_in_sql_for_every_search_and_an_older_turn_is_not(make):
    writer = FakeWriterLLM(searches_reply("Henley roof leak report", "Henley roof contractor", "Henley roof budget"))
    p = make(decision=jev_decision("follow_up", 0.9, stands=0.1), writer=writer)
    turn(p, "about the Henley roof tell me one", "Reply one: the Henley roof leak was reported.")
    turn(p, "about the Henley roof tell me two", "Reply two: the Henley roof contractor was named.")
    turn(p, "about the Henley roof tell me three", "Reply three: the Henley roof budget was set.")
    out = p.prefetch("and the roof?")
    assert "Reply one" in out                                       # older: still eligible
    assert "Reply two" not in out and "Reply three" not in out     # the last two: never returned
    assert len(p.rerank.requests) == 3                              # every one of the three searches ran Stage 2
    docs = reranked_docs(p)                                         # and none of them ever saw the last two turns
    assert "Reply two" not in docs and "Reply three" not in docs


def test_the_date_window_of_a_relative_time_scope_is_a_stage_1_filter_of_every_search(make):
    writer = FakeWriterLLM(searches_reply(ROOF_SEARCH, "Henley branch roof leak report", "Henley branch board"))
    p = make(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"), writer=writer)
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    note(p, OLD_ROOF, "note:old", "2026-07-14")
    out = p.prefetch(ROOF_Q)
    assert "source=note:new" in out and "note:old" not in out
    assert "Only notes dated 2026-10-10 to 2026-10-17 were searched." in out
    assert len(p.rerank.requests) == 3 and "leaks" not in reranked_docs(p)    # no search read the old note


def test_a_named_period_is_a_search_s_own_window_from_the_librarys_writer(make):
    """The plugin passes no window for 'named'; the library's writer dates the search that names the period,
    and the other search is unrestricted."""
    writer = FakeWriterLLM(searches_reply(
        {"query": "Henley branch roof board July decision", "date_from": "2026-07-01", "date_to": "2026-07-31"},
        "Henley branch roof replaced in June"))
    p = make(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="named"), writer=writer)
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    note(p, OLD_ROOF, "note:old", "2026-07-14")
    out = p.prefetch("What did we decide in July about the Henley branch roof?")
    old = [ln for ln in out.split("\n") if "source=note:old" in ln]
    assert len(old) == 1 and "found_by=1" in old[0]                          # the dated search found the July note
    assert "dates: 2026-07-01 to 2026-07-31" in out and "were searched" not in out


# ---------------------------------------------------------------------------- never fails, logs nothing, records nothing

def test_a_bank_without_the_hybrid_channel_gives_the_not_consulted_line_never_an_error(make):
    p = make(hybrid=False, decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"))
    note(p, NEW_ROOF, "note:new", "2026-10-12")
    assert p.prefetch(ROOF_Q) == "(memory was not consulted: the search failed)"


def test_a_successful_turn_logs_nothing_and_records_no_recall_event(make, pg_url, caplog):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    before = rows(pg_url, "SELECT count(*) FROM recall_events")[0][0]
    caplog.clear()                      # setup logged the bank check (the bank was configured after it started)
    with caplog.at_level(logging.DEBUG):
        out = p.prefetch(ROOF_Q)
    assert NEW_ROOF in out
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert not [r for r in caplog.records if "Henley" in r.getMessage()]       # no search or note text, ever
    assert rows(pg_url, "SELECT count(*) FROM recall_events")[0][0] == before   # no recall logging
    assert rows(pg_url, "SELECT count(*) FROM llm_calls")[0][0] == 0      # not even the writer's call is recorded


def test_prefetch_never_raises_even_when_the_library_does(make, monkeypatch, caplog):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")

    def boom(*a, **k):
        raise RuntimeError("secret words: the Panama deal")
    monkeypatch.setattr(p._memory, "expand_and_recall", boom)
    with caplog.at_level(logging.INFO):
        out = p.prefetch(ROOF_Q)
    assert out == "(memory was not consulted: the search failed)" and "Panama" not in caplog.text


def test_a_failing_writer_falls_back_to_one_search_with_the_prompt_and_the_turn_still_gets_memory(make):
    p = make(writer=FakeWriterLLM(fail=RuntimeError("provider down")))
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    out = p.prefetch(ROOF_Q)
    assert f"(1) {ROOF_Q}" in out and NEW_ROOF in out                        # the library's own fallback


def test_a_bank_with_no_writer_model_at_all_still_searches_the_prompt(make):
    p = make()
    p._memory._llm = None
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    assert NEW_ROOF in p.prefetch(ROOF_Q)


# ---------------------------------------------------------------------------- the deadline

def test_a_reranker_slower_than_the_deadline_drops_the_late_searches_and_the_turn_goes_on(make):
    p = make(config={"recall": {"prefetch": {"mode": "contract", "deadline_seconds": 1.0}}},
             rerank=FakeRerank(delay=3.0))
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    t0 = time.monotonic()
    out = p.prefetch(ROOF_Q)
    assert time.monotonic() - t0 < 2.0
    assert out == "(memory was not consulted: the search did not answer in time)"
    assert p._last_brief.injected == 0


def test_a_writer_slower_than_its_share_falls_back_inside_the_deadline_and_still_returns_memory(make):
    p = make(config={"recall": {"prefetch": {"mode": "contract", "deadline_seconds": 2.0}}},
             writer=FakeWriterLLM(delay=5.0))
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    t0 = time.monotonic()
    out = p.prefetch(ROOF_Q)
    assert time.monotonic() - t0 < 3.0
    assert f"(1) {ROOF_Q}" in out and NEW_ROOF in out


def test_the_deadline_the_library_gets_is_the_plugins_remainder_never_above_the_cap(make, monkeypatch):
    p = make(config={"recall.prefetch.deadline_seconds": "30"})
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    seen = {}
    real = p._memory.expand_and_recall

    def spy(prompt, kind="chat", **kw):
        seen.update(kw)
        return real(prompt, kind, **kw)
    monkeypatch.setattr(p._memory, "expand_and_recall", spy)
    p.prefetch(ROOF_Q)
    assert 6.0 < seen["deadline"] <= 7.0


# ---------------------------------------------------------------------------- models and settings reach the library

def test_the_models_reach_the_library_and_the_transport(make):
    p = make(config={"reranker_model": "x/reranker-cfg", "decision_model": "typesafe/jev-9",
                     "query_writer_model": "openrouter/x/writer-cfg"})
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    p.prefetch(ROOF_Q)
    assert p.rerank.requests[0]["model"] == "x/reranker-cfg"
    assert p.jev.decisions()[0]["model"] == "typesafe/jev-9"
    assert p.writer.models == ["openrouter/x/writer-cfg"]
    assert p._memory._reranker_model == "x/reranker-cfg" and p._memory._decision_model == "typesafe/jev-9"


def test_with_no_profile_models_the_library_resolves_environment_then_default(make, monkeypatch):
    from prospecta import stages
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    assert p._memory._reranker_model is None and p._memory._decision_model is None      # nothing duplicated here
    p.prefetch(ROOF_Q)
    assert p.writer.models == ["openrouter/google/gemini-2.5-flash-lite"]               # the library's default
    assert p.rerank.requests[0]["model"] == "qwen/qwen3-reranker-8b"
    assert p.jev.decisions()[0]["model"] == stages.JEV_MODEL
    monkeypatch.setenv("PROSPECTA_QUERY_WRITER_MODEL", "openrouter/x/writer-env")
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "x/reranker-env")
    p.prefetch(ROOF_Q)
    assert p.writer.models[-1] == "openrouter/x/writer-env"
    assert p.rerank.requests[-1]["model"] == "x/reranker-env"


def test_the_deprecated_question_model_key_still_sets_the_writer_model(make, caplog):
    p = make(config={"recall": {"prefetch": {"mode": "contract", "question_model": "openrouter/x/writer-old"}}})
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        p._warn_renamed_settings()
    assert len(caplog.records) == 1 and "recall.prefetch.question_model is deprecated" in caplog.text
    p.prefetch(ROOF_Q)
    assert p.writer.models == ["openrouter/x/writer-old"]


def test_the_token_valve_is_the_librarys(make):
    p = make(config={"recall": {"prefetch": {"mode": "contract", "max_tokens": 60}}})
    for name in ("tiles", "gutters", "scaffolding", "insurance", "contractor", "timeline"):
        note(p, f"The Henley branch roof decision on {name}: " + f"the board agreed about {name} and recorded it. " * 6,
             f"note:{name}", "2026-10-12")
    out = p.prefetch(ROOF_Q)
    assert p._last_brief.injected == 1                                   # best first, at least one, then the valve
    assert re.search(r"\(\d+ more records matched less well and not shown\.\)", out)


# ---------------------------------------------------------------------------- the tools are untouched

def test_the_search_tool_still_runs_the_librarys_stage_1_and_stage_2_without_the_writer_or_the_gate(make):
    p = make()
    note(p, NEW_ROOF, "note:roof", "2026-10-12")
    out = json.loads(p.handle_tool_call("prospecta_search", {"query": "Henley branch roof"}))
    assert [r["source"] for r in out["results"]] == ["note:roof"] and out["reranked"] is True
    assert not p.writer.prompts and not p.jev.requests                         # no expansion, no decision


# ---------------------------------------------------------------------------- the bank check at initialise

def test_the_stage_check_says_what_the_bank_lacks_for_the_expansion(make, caplog):
    p = make(hybrid=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        p._check_recall_stages("hyb")
    assert "no hybrid_chunk channel" in caplog.text and "cross_encoder" in caplog.text
    assert "every turn goes without memory" in caplog.text
    assert [r.levelno for r in caplog.records] == [logging.WARNING, logging.WARNING]


def test_the_stage_check_says_so_when_the_library_has_no_expansion_entry_point(make, caplog):
    from types import SimpleNamespace
    p = make()
    p._memory = SimpleNamespace()                         # the library before PR 85
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        p._check_recall_stages("hyb")
    assert len(caplog.records) == 1 and "expand_and_recall" in caplog.text and "736bd28" in caplog.text


def test_after_a_restart_the_last_two_turns_are_read_from_the_bank(make, plugin_module, pg_url):
    p = make()
    for i in (1, 2, 3):
        turn(p, f"user line {i}", f"assistant reply {i}")
    sources = [r[0] for r in rows(pg_url, "SELECT source FROM documents ORDER BY created_at, source")]
    assert len(sources) == 3
    fresh = plugin_module.ProspectaProvider()
    fresh._memory = p._memory
    assert fresh._read_turn_sources("s1") == sources[-2:]          # oldest first, the last two
    assert fresh._read_turn_sources("s1x") == [] and fresh._read_turn_sources("s_") == []   # an exact prefix, no wildcard
