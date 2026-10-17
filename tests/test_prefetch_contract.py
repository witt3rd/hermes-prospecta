"""The pre-turn recall contract (recall.prefetch.mode: contract), offline. The plugin gates and decides (a fake
Jev transport), then hands the message to the library's query expansion (``Memory.expand_and_recall``, replaced
here by a fake that records its arguments and returns the library's own result types, block and all). Invented
text only. The real library behind it is in ``test_mode1a_hybrid.py``."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date

import pytest
from prospecta._types import ExpandedRecall, RecalledMemory, SearchRun, SearchSpec
from prospecta.cited import CITE_LINE, HEADING, format_cited_rag_block

import prefetch_contract as pc

OPENER_AGENT_LAST = ("I can draft the one-page explainer on the village library now. "
                     "Shall I write it in our explainer house style?")
ROOF_Q = "What did we decide about the Henley branch roof?"
TODAY = date(2026, 10, 17)


# ---------------------------------------------------------------------------- fakes

def jev_decision(kind, needs, *, stands=0.9, scope="none", **need):
    """A System One answer in the real shape (every question of the contract answered); ``need`` names the
    need_* at or above 0.5; ``scope`` is the time_scope choice."""
    def noul(v):
        return {"type": "noul", "noul": v}
    answers = {
        "line_kind": {"type": "choice", "choice": kind, "probabilities": {kind: 0.9}, "confidence": 0.9},
        "stands_alone": noul(stands),
        "needs_memory": noul(needs),
        "time_scope": {"type": "choice", "choice": scope, "probabilities": {scope: 0.9}, "confidence": 0.9},
    }
    for k in ("need_decisions", "need_library", "need_conversations", "need_work", "need_person"):
        answers[k] = noul(0.9 if need.get(k) else 0.1)
    return {"model": "typesafe/jev-1.14-20261014", "answers": answers,
            "usage": {"input_tokens": 900, "output_tokens": 12, "cost": 0.00004}}


class FakeJev:
    """The decision (System One) transport: the plugin's only Jev call."""

    def __init__(self, decision=None, fail_decision=None, delay=0.0):
        self.decision, self.fail_decision, self.delay = decision, fail_decision, delay
        self.requests = []
        self.lock = threading.Lock()

    def decisions(self):
        return [r for r in self.requests if "line_kind" in r["questions"]]

    def __call__(self, request, timeout):
        with self.lock:
            self.requests.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.fail_decision:
            raise self.fail_decision
        return self.decision


def item(name, text=None, *, date="2026-03-02", source=None, doc=None, kind="note", match="semantic",
         snippet=None, score=0.8, found_by=(1,)):
    """A recalled record as the library's expansion hands it back (its own type)."""
    return RecalledMemory(
        content=f"stored text of {name}", original_chunk=text or f"stored text of {name}",
        source=source or f"note:{name}", score=0.5, scores={"rrf": 0.5}, metadata={}, bank_id="b",
        document_id=doc or f"doc-{name}", created_on=date, kind=kind, match_kind=match,
        exact_snippet=snippet, relevance_score=score, found_by=found_by)


def expanded(searches, items=(), *, status=None, reranked=True, deadline_hit=False, error=None, cost=0.0003,
             fallbacks=()):
    """What ``expand_and_recall`` returns, built with the library's own types and its own block formatter.
    ``searches`` are queries (or SearchSpecs); ``status`` maps a search number to ok | late | error."""
    specs = [s if isinstance(s, SearchSpec) else SearchSpec(query=s) for s in searches]
    runs = [SearchRun(index=i, spec=sp, status=(status or {}).get(i, "ok"), n_candidates=len(items),
                      n_kept=len(items), reranked=reranked,
                      fallback_reason=None if reranked else "stage 2: HTTP 503")
            for i, sp in enumerate(specs, start=1)]
    ok = [r for r in runs if r.status == "ok"]
    block = format_cited_rag_block(list(items), heading=HEADING, searches=[(r.index, r.spec) for r in ok],
                                   cite=True)
    return ExpandedRecall(kind="chat", block=block, items=list(items), searches=specs, runs=runs,
                          fallbacks=list(fallbacks), timings_ms={"writer": 11, "stage1": 7, "total": 40},
                          accounting={"n_llm_calls": 1, "cost_usd": cost}, deadline_hit=deadline_hit, error=error)


class FakeExpansion:
    """``Memory.expand_and_recall``, replaced: it records every call (prompt, kind, keyword arguments) and
    returns ``result`` (a result, or a function of the call's prompt and keyword arguments)."""

    def __init__(self, result=None, *, fail=None, delay=0.0, gate=None):
        self.result, self.fail, self.delay, self.gate = result, fail, delay, gate
        self.calls = []

    @property
    def kwargs(self):
        return [c[2] for c in self.calls]

    def __call__(self, prompt, kind="chat", **kw):
        self.calls.append((prompt, kind, kw))
        if self.gate is not None:
            self.gate.wait()
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        result = self.result
        if callable(result):
            result = result(prompt, kw)
        return result if result is not None else expanded(["Henley branch roof decision"], [item("roof")])


def make_deps(expansion, jev=None):
    return pc.Deps(expand=expansion, transport=jev, today=lambda: TODAY)


SETTINGS = pc.Settings(mode="contract", decision_timeout_s=0.5, deadline_s=3.0)


def run(line, deps, *, agent_last=OPENER_AGENT_LAST, settings=SETTINGS, **kw):
    return pc.run_contract(line, agent_last=agent_last, turns=4, deps=deps, settings=settings,
                           contract=pc.load_contract(), **kw)


def ask(line="What did we decide about the Henley branch roof?", *, decision=None, expansion=None, **kw):
    """One turn on a decision that asks (new_ask, 0.9) unless another is given: (brief, expansion, jev)."""
    jev = FakeJev(decision=decision or jev_decision("new_ask", 0.9, stands=0.95))
    expansion = expansion or FakeExpansion()
    return run(line, make_deps(expansion, jev), **kw), expansion, jev


# ---------------------------------------------------------------------------- the contract file

def test_bundled_contract_loads_with_the_designs_words():
    c = pc.load_contract()
    assert list(c.questions) == ["line_kind", "stands_alone", "needs_memory", "need_decisions", "need_library",
                                 "need_conversations", "need_work", "need_person", "time_scope"]
    assert list(c.questions["time_scope"]["criteria"]) == list(pc.TIME_SCOPES)
    kinds = c.questions["line_kind"]["criteria"]
    assert list(kinds) == ["answer", "go_ahead", "follow_up", "new_ask", "material", "social"]
    assert kinds["social"] == "Thanks, a greeting or an acknowledgement; nothing to do."
    assert c.questions["line_kind"]["instructions"] == "What is `new_lines`, read as a reply to `agent_last`?"
    assert c.questions["needs_memory"]["instructions"].startswith(
        "To do what `new_lines` asks (or what it agrees to), does the agent need something that is NOT in")
    assert c.questions["needs_memory"]["criteria"]["true"] == "Something it needs is outside what it has."


def test_the_time_scope_is_the_memorys_period_not_the_requests():
    """The question asks for a window ONLY when the line limits the MEMORY it asks about to a past period; a
    relative period that only modifies the request ("this week's theme") is no window; doubt is no window. The
    five choices and the set of questions are unchanged (the single owner of the wording is the contract file)."""
    c = pc.load_contract()
    q = c.questions["time_scope"]
    assert q["type"] == "choice" and tuple(q["criteria"]) == pc.TIME_SCOPES == (
        "none", "recent", "past_week", "past_month", "past_year", "named")
    text = (q["instructions"] + " " + " ".join(q["criteria"].values())).lower()
    assert "modifies the request" in text                  # the rule: a period of the request is not the memory's
    assert "this week's" in text and "tonight" in text and "next week" in text     # what that looks like
    assert "worthless" in text                             # a window only when older notes would be useless
    assert "older notes could also answer" in text         # ... and none when they could still answer
    assert "when in doubt: none" in text                   # the plain rule
    assert "when in doubt: none" in q["criteria"]["none"].lower()   # ... also where the model picks its answer
    for scope in ("recent", "past_week", "past_month", "past_year"):
        assert "happened" in q["criteria"][scope].lower()  # a window states what the MEMORY is about
    assert "limit what it asks about" not in q["instructions"]     # the old, misleading question


def test_the_wording_of_the_time_scope_lives_in_the_contract_only():
    """No second copy of the question's words in the code (the code only reads the answer)."""
    code = (pc.DEFAULT_CONTRACT_PATH.parent.parent / "prefetch_contract.py").read_text()
    assert "modifies the request" not in code and "when in doubt" not in code.lower()


def test_the_contract_has_no_question_writer_prompt_any_more():
    """The library's writer owns the prompt for the searches (prospecta/prompts/expand-searches.md)."""
    assert not hasattr(pc.load_contract(), "writer_prompt")
    assert "## question-writer-prompt" not in pc.DEFAULT_CONTRACT_PATH.read_text()


def test_the_fixed_sentences_speak_of_searches_not_questions():
    c = pc.load_contract()
    assert c.line("nothing-answers", searches="a / b") == "Memory was asked: a / b. Nothing in it answers this."
    assert c.line("not-consulted", reason="x") == "(memory was not consulted: x)"
    assert c.line("no-decision") == "Searches were made without the decision model."
    assert c.line("window", date_from="2026-10-10", date_to="2026-10-17") == (
        "Only notes dated 2026-10-10 to 2026-10-17 were searched.")
    assert c.line("search-order") == "The reranker could not rank these; they are in the search's own order."
    assert "question" not in c.intro.lower().replace("short question", "")
    assert "prospecta_search" in c.intro and "stale" in c.intro
    assert "asked" not in c.lines and "more" not in c.lines            # the library writes those parts of the block


def test_swapping_the_contract_file_is_enough(tmp_path):
    text = pc.DEFAULT_CONTRACT_PATH.read_text()
    path = tmp_path / "other.md"
    path.write_text(text.replace("Background from memory,", "Swapped intro,"))
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.1))
    b = pc.run_contract("and that branch?", agent_last="", turns=1, deps=make_deps(FakeExpansion(), jev),
                        settings=SETTINGS, contract=pc.load_contract(path))
    assert b.text.startswith("Swapped intro,")


def test_a_contract_file_that_still_has_the_old_writer_prompt_loads(tmp_path):
    text = pc.DEFAULT_CONTRACT_PATH.read_text()
    old = text + '\n## question-writer-prompt\n\n```text\nWrite the questions.\n```\n'
    path = tmp_path / "old.md"
    path.write_text(old)
    assert pc.load_contract(path).intro == pc.load_contract().intro      # the extra section is not read


def test_bad_contract_files_are_refused(tmp_path):
    with pytest.raises(pc.ContractError):
        pc.load_contract(tmp_path / "missing.md")
    bad = tmp_path / "bad.md"
    bad.write_text("## jev-questions\n\n```json\n{\"line_kind\": {}}\n```\n")
    with pytest.raises(pc.ContractError):
        pc.load_contract(bad)


def test_the_decision_request_has_the_designs_shape():
    _, _, jev = ask()
    (req,) = jev.decisions()
    assert list(req["state"]) == ["agent", "turn", "thread", "agent_last", "new_lines", "in_view", "today"]
    assert req["state"]["today"] == "2026-10-17"
    assert req["state"]["agent_last"] == OPENER_AGENT_LAST
    assert req["state"]["new_lines"] == [f"The person (to you): {ROOF_Q}"]
    assert list(req["questions"]) == list(pc.load_contract().questions)


def test_the_decision_model_is_the_profiles_else_the_librarys_never_a_default_of_the_plugins():
    from prospecta import stages
    _, _, jev = ask()
    assert jev.decisions()[0]["model"] == stages.JEV_MODEL               # env PROSPECTA_DECISION_MODEL, else its default
    jev = FakeJev(decision=jev_decision("new_ask", 0.9))
    run(ROOF_Q, make_deps(FakeExpansion(), jev), settings=pc.Settings(mode="contract", decision_model="typesafe/jev-9"))
    assert jev.decisions()[0]["model"] == "typesafe/jev-9"
    assert not hasattr(pc, "DEFAULT_DECISION_MODEL") and not hasattr(pc, "DEFAULT_QUERY_WRITER_MODEL")
    assert not hasattr(pc, "DEFAULT_RERANKER_MODEL")


# ---------------------------------------------------------------------------- the gate stays in the plugin

@pytest.mark.parametrize("text", ["", "   \n ", "<memory-context>\n</memory-context>", "/new", "  /status now",
                                  "yes", "OK!", "thanks :)", "do it", None])
def test_gate_skips_without_any_call(text):
    jev, expansion = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeExpansion()
    b = run(text, make_deps(expansion, jev))
    assert b.text == "" and b.decision == "zero"
    assert not jev.requests and not expansion.calls


@pytest.mark.parametrize("kind,needs", [("social", 0.95), ("answer", 0.04), ("go_ahead", 0.1), ("new_ask", 0.49),
                                        ("follow_up", 0.0)])
def test_a_closed_gate_makes_no_expansion_call_and_injects_nothing(kind, needs):
    b, expansion, jev = ask("Thanks, that was helpful to know about it",
                            decision=jev_decision(kind, needs, need_decisions=1))
    assert b.text == "" and b.decision == "zero" and b.injected == 0
    assert expansion.calls == []                                        # not called at all
    assert len(jev.requests) == 1                                       # the turn pays the decision only
    assert b.cost_usd == pytest.approx(0.00004)


def test_the_threshold_is_one_half_and_the_gate_is_the_plugins_not_the_librarys():
    b, expansion, _ = ask(decision=jev_decision("new_ask", 0.5))
    assert b.decision == "ask" and len(expansion.calls) == 1
    assert expansion.kwargs[0]["gate"] is False                         # decided here: the library does not decide again


def test_social_is_zero_whatever_needs_memory_says():
    b, expansion, _ = ask("Thanks, that was helpful", decision=jev_decision("social", 0.95))
    assert b.text == "" and not expansion.calls


def test_ok_do_it_is_decided_by_jev_while_yes_is_core_filtered():
    jev = FakeJev(decision=jev_decision("go_ahead", 0.08, stands=0.1))
    expansion = FakeExpansion()
    deps = make_deps(expansion, jev)
    assert run("yes", deps).text == "" and not jev.requests
    b = run("ok do it", deps)
    assert len(jev.decisions()) == 1 and b.text == "" and b.line_kind == "go_ahead" and not expansion.calls


def test_a_go_ahead_that_needs_memory_is_expanded_with_the_agents_last_line():
    b, expansion, _ = ask("ok do it", decision=jev_decision("go_ahead", 0.85, stands=0.1, need_library=1))
    assert b.decision == "ask" and len(expansion.calls) == 1
    prompt, kind, kw = expansion.calls[0]
    assert (prompt, kind) == ("ok do it", "chat") and kw["agent_last"] == OPENER_AGENT_LAST


# ---------------------------------------------------------------------------- what reaches the library

def test_the_expansion_is_called_with_the_line_as_a_chat_and_the_arguments_the_plugin_owns():
    b, expansion, _ = ask(ROOF_Q)
    ((prompt, kind, kw),) = expansion.calls
    assert prompt == ROOF_Q and kind == "chat"
    assert kw["agent_last"] == OPENER_AGENT_LAST
    assert kw["gate"] is False and kw["record"] is False                # no second gate, no recall logging
    assert kw["now"] == TODAY                                           # the profile's calendar day
    assert kw["max_tokens"] is None and kw["query_writer_model"] is None   # not set: the library resolves both
    assert kw["date_from"] is None and kw["date_to"] is None and kw["excluded_sources"] is None
    assert 0 < kw["deadline"] <= SETTINGS.deadline_s
    assert set(kw) == {"agent_last", "date_from", "date_to", "excluded_sources", "deadline", "max_tokens",
                       "query_writer_model", "gate", "record", "now"}
    assert not any(k in kw for k in ("project", "metadata_filter", "max_searches", "reranker_model", "decision"))


def test_a_follow_up_carries_the_agents_last_line_and_the_first_message_carries_none():
    _, expansion, _ = ask("do you remember who i saw it with?", decision=jev_decision("follow_up", 0.9, stands=0.1),
                          agent_last="We met Ana at the cafe on Tuesday.")
    assert expansion.kwargs[0]["agent_last"] == "We met Ana at the cafe on Tuesday."
    assert expansion.calls[0][0] == "do you remember who i saw it with?"        # the line, whole: never rewritten here
    _, first, _ = ask(agent_last="")
    assert first.kwargs[0]["agent_last"] is None


def test_the_writer_model_the_valve_and_the_echo_exclusion_are_passed_on():
    s = pc.Settings(mode="contract", decision_timeout_s=0.5, deadline_s=3.0, query_writer_model="openrouter/x/writer",
                    max_tokens=900)
    _, expansion, _ = ask(settings=s, excluded_sources=("turn:s1:11", "turn:s1:12"))
    kw = expansion.kwargs[0]
    assert kw["query_writer_model"] == "openrouter/x/writer" and kw["max_tokens"] == 900
    assert kw["excluded_sources"] == ["turn:s1:11", "turn:s1:12"]
    _, off, _ = ask(settings=pc.Settings(mode="contract", drop_echoes=False), excluded_sources=("turn:s1:11",))
    assert off.kwargs[0]["excluded_sources"] is None


@pytest.mark.parametrize("scope,window", [
    ("recent", ("2026-10-16", "2026-10-17")),
    ("past_week", ("2026-10-10", "2026-10-17")),
    ("past_month", ("2026-09-16", "2026-10-17")),
    ("past_year", ("2025-10-16", "2026-10-17")),
])
def test_a_relative_time_scope_is_the_window_every_search_carries(scope, window):
    b, expansion, jev = ask(decision=jev_decision("new_ask", 0.9, stands=0.95, scope=scope))
    kw = expansion.kwargs[0]
    assert (kw["date_from"], kw["date_to"]) == window and (b.date_from, b.date_to) == window
    assert b.time_scope == scope
    assert f"Only notes dated {window[0]} to {window[1]} were searched." in b.text
    assert jev.decisions()[0]["questions"]["time_scope"]["type"] == "choice"


@pytest.mark.parametrize("scope", ["none", "named"])
def test_no_window_is_passed_for_none_and_for_a_named_period(scope):
    """A named period ("in July") is dated by the library's writer, search by search; the plugin passes none."""
    b, expansion, _ = ask(decision=jev_decision("new_ask", 0.9, stands=0.95, scope=scope))
    kw = expansion.kwargs[0]
    assert kw["date_from"] is None and kw["date_to"] is None and b.time_scope == scope
    assert "were searched" not in b.text


def test_an_unknown_or_missing_time_scope_answer_is_no_window_never_no_decision():
    for mutate in (lambda a: a.pop("time_scope"), lambda a: a["time_scope"].update(choice="next_century"),
                   lambda a: a.update(time_scope="garbage")):
        decision = jev_decision("new_ask", 0.9, stands=0.95, scope="past_week")
        mutate(decision["answers"])
        b, expansion, _ = ask(decision=decision)
        assert b.decision == "ask" and expansion.kwargs[0]["date_from"] is None


def test_a_contract_without_the_time_scope_question_still_works(tmp_path):
    import re
    text = pc.DEFAULT_CONTRACT_PATH.read_text()
    body = re.search(r"## jev-questions.*?```json\n(.*?)\n```", text, re.S).group(1)
    questions = json.loads(body)
    del questions["time_scope"]
    path = tmp_path / "old.md"
    path.write_text(text.replace(body, json.dumps(questions)))
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"))
    expansion = FakeExpansion()
    b = pc.run_contract(ROOF_Q, agent_last="", turns=1, deps=make_deps(expansion, jev), settings=SETTINGS,
                        contract=pc.load_contract(path))
    assert "time_scope" not in jev.decisions()[0]["questions"] and expansion.kwargs[0]["date_from"] is None
    assert b.injected == 1


def test_the_plugin_reads_only_what_it_uses_of_the_decision():
    """stands_alone and the need_* are still asked (the contract's words) but nothing reads them any more."""
    partial = jev_decision("new_ask", 0.9)
    for gone in ("stands_alone", "need_decisions", "need_library", "need_conversations", "need_work", "need_person"):
        del partial["answers"][gone]
    b, expansion, _ = ask(decision=partial)
    assert b.decision == "ask" and len(expansion.calls) == 1
    fields = pc.Decision.__dataclass_fields__
    assert "stands_alone" not in fields and "needs" not in fields and not hasattr(pc.Decision, "sources")


@pytest.mark.parametrize("broken", [
    lambda a: a.pop("needs_memory"), lambda a: a.pop("line_kind"),
    lambda a: a["needs_memory"].update(noul=1.5), lambda a: a["needs_memory"].update(noul="high"),
    lambda a: a["line_kind"].update(choice="whatever"),
])
def test_an_unreadable_decision_is_no_decision(broken):
    decision = jev_decision("new_ask", 0.9)
    broken(decision["answers"])
    b, expansion, _ = ask(decision=decision)
    assert b.decision == "fallback" and len(expansion.calls) == 1        # the rule: asked, with the note
    assert "Searches were made without the decision model." in b.text


def test_the_window_math_is_pure():
    for scope in ("recent", "past_week", "past_month", "past_year", "named", "none"):
        w = pc.relative_window(scope, date(2026, 3, 1))
        assert (w is None) == (scope in ("none", "named"))
        if w:
            assert w[1] == "2026-03-01" and w[0] < w[1]


# ---------------------------------------------------------------------------- a paste

def test_a_paste_is_marked_as_material_by_its_size_and_goes_whole_to_the_library():
    paste = "Minutes of the branch review. " * 2500                      # 75,000 characters
    line = paste + "\nCompare this with what we decided in March."
    jev = FakeJev(decision=jev_decision("material", 0.9))
    expansion = FakeExpansion()
    b = run(line, make_deps(expansion, jev))
    assert not jev.decisions()                                          # too long for Jev to read
    assert b.line_kind == "material" and b.decision == "ask"
    assert expansion.calls[0][0] == line                                # whole, never sliced
    assert b.injected == 1


def test_a_smaller_paste_goes_to_jev_first():
    line = "Notes from the call: the roof, the budget, the opening hours.\n" * 40 + "What did we decide in March?"
    b, expansion, jev = ask(line, decision=jev_decision("material", 0.9, stands=0.4))
    assert len(jev.decisions()) == 1 and expansion.calls[0][0] == line


def test_a_line_over_the_reading_limit_is_skipped():
    expansion = FakeExpansion()
    b = run("x " * (pc.MAX_LINE_CHARS // 2 + 10), make_deps(expansion, FakeJev(decision=jev_decision("new_ask", 0.9))))
    assert b.text == "" and b.reason == "too large to read" and not expansion.calls


# ---------------------------------------------------------------------------- fallbacks of the decision

def test_jev_refusal_decides_by_rule_and_the_block_says_so():
    expansion = FakeExpansion()
    deps = make_deps(expansion, FakeJev(fail_decision=RuntimeError("HTTP 400")))
    b = run("What are the Henley branch opening hours these days?", deps)
    assert b.decision == "fallback" and len(expansion.calls) == 1
    assert "Searches were made without the decision model." in b.text
    assert expansion.kwargs[0]["date_from"] is None                     # no decision: no window
    short = run("sounds good", deps)
    assert short.text == "" and short.decision == "fallback" and len(expansion.calls) == 1   # three words or fewer


def test_slow_jev_falls_back_inside_its_budget():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9), delay=2.0)
    t0 = time.monotonic()
    b = run(ROOF_Q, make_deps(FakeExpansion(), jev))
    assert time.monotonic() - t0 < 1.5 and b.decision == "fallback" and b.injected == 1


def test_jev_without_a_transport_or_key_still_gives_an_answer():
    b = run(ROOF_Q, make_deps(FakeExpansion()))
    assert b.decision == "fallback" and b.text and b.injected == 1


# ---------------------------------------------------------------------------- the block

def test_the_block_is_the_contracts_intro_then_the_librarys_searches_cite_line_and_entries():
    items = [item("a", date="2026-03-02", match="exact+semantic", snippet="the <<roof>> failed", found_by=(1, 2)),
             item("b", date="2026-04-01", kind="conversation", match="lexical", found_by=(2,)),
             item("c", date="2026-05-09", match="semantic", found_by=(1,))]
    result = expanded([SearchSpec(query="Henley branch roof decision", terms=("Henley",)),
                       SearchSpec(query="branch roof budget", date_from="2026-03-01", date_to="2026-05-31")], items)
    b, _, _ = ask(expansion=FakeExpansion(result))
    c = pc.load_contract()
    assert b.text.startswith(c.intro) and HEADING not in b.text          # the contract's paragraph, not the library's
    assert "Searches run (an item's header names the searches that found it):" in b.text
    assert "(1) Henley branch roof decision | terms: Henley" in b.text
    assert "(2) branch roof budget | dates: 2026-03-01 to 2026-05-31" in b.text
    assert CITE_LINE in b.text                                           # the agent is told to cite [n]
    assert ('[1] source=note:a | date=2026-03-02 | kind=note | match=exact+semantic | found_by=1,2\n'
            '    Exact match snippet: "the <<roof>> failed"\n    stored text of a') in b.text
    assert "[2] source=note:b | date=2026-04-01 | kind=conversation | match=lexical | found_by=2\n    stored text of b" in b.text
    assert b.text.count("Exact match snippet:") == 1 and "Asked:" not in b.text and "question" not in b.text.lower().replace(
        "short question", "")
    assert b.injected == 3 and b.searches == ["Henley branch roof decision", "branch roof budget"]
    # everything after the intro is the library's own block, entry for entry
    assert result.block[len(HEADING):].lstrip("\n") in b.text


def test_the_intro_replaces_the_librarys_heading_and_a_changed_heading_is_not_a_problem():
    result = expanded(["roof"], [item("a")])
    result = ExpandedRecall(**{**result.__dict__, "block": result.block.replace(HEADING, "A NEW LIBRARY HEADING:")})
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert b.text.startswith(pc.load_contract().intro) and "A NEW LIBRARY HEADING:" in b.text and "[1] source=note:a" in b.text


def test_nothing_answers_lists_the_searches_that_ran():
    result = expanded(["Henley branch roof decision", "roof repair budget"], [])
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert b.text == ("Memory was asked: Henley branch roof decision / roof repair budget. "
                      "Nothing in it answers this.")
    assert b.injected == 0 and b.reason == ""


def test_nothing_answers_inside_a_window_says_which_window():
    b, _, _ = ask(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"),
                  expansion=FakeExpansion(expanded(["roof"], [])))
    assert b.text == "Memory was asked: roof. Nothing in it answers this. Only notes dated 2026-10-10 to 2026-10-17 were searched."


def test_only_the_searches_that_ran_are_listed_in_nothing_answers():
    result = expanded(["first", "second", "third"], [], status={2: "error", 3: "late"})
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert b.text == "Memory was asked: first. Nothing in it answers this."


def test_a_window_note_and_the_search_order_note_and_the_no_decision_note():
    result = expanded(["roof"], [item("a")], reranked=False)
    jev = FakeJev(fail_decision=RuntimeError("down"))
    b = run(ROOF_Q, make_deps(FakeExpansion(result), jev))
    assert "The reranker could not rank these; they are in the search's own order." in b.text
    assert "Searches were made without the decision model." in b.text and b.ranked is False
    b2, _, _ = ask(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="recent"),
                   expansion=FakeExpansion(expanded(["roof"], [item("a")])))
    assert "Only notes dated 2026-10-16 to 2026-10-17 were searched." in b2.text and b2.ranked is True
    assert "own order" not in b2.text


def test_a_graded_block_has_no_search_order_note_and_no_count_cap():
    items = [item(f"n{i}", found_by=(1,)) for i in range(40)]
    b, _, _ = ask(expansion=FakeExpansion(expanded(["roof"], items)))
    assert b.injected == 40 and b.ranked is True and "own order" not in b.text and "more records" not in b.text


def test_the_librarys_token_valve_line_comes_through():
    result = expanded(["roof"], [item("a")])
    block = result.block + "\n\n(3 more records matched less well and not shown.)"
    result = ExpandedRecall(**{**result.__dict__, "block": block})
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert b.text.endswith("(3 more records matched less well and not shown.)")


# ---------------------------------------------------------------------------- failure: the not-consulted line

def test_a_library_that_raises_gives_the_not_consulted_line(caplog):
    with caplog.at_level(logging.DEBUG):
        b, expansion, _ = ask(expansion=FakeExpansion(fail=RuntimeError("secret words: the Panama deal")))
    assert b.text == "(memory was not consulted: the search failed)" and b.injected == 0 and len(expansion.calls) == 1
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == [] and "Panama" not in caplog.text


@pytest.mark.parametrize("err", [RuntimeError("x"), KeyError("k"), ValueError(), OSError(5, "io"), AttributeError("no such")])
def test_whatever_the_library_raises_is_the_not_consulted_line(err):
    b, _, _ = ask(expansion=FakeExpansion(fail=err))
    assert b.text == "(memory was not consulted: the search failed)"


def test_an_error_result_is_the_not_consulted_line():
    err = expanded([], [], error="the bank's channel_config has no hybrid_chunk channel")
    b, _, _ = ask(expansion=FakeExpansion(err))
    assert b.text == "(memory was not consulted: the search failed)" and "hybrid_chunk" not in b.text


def test_every_search_failing_is_the_not_consulted_line():
    b, _, _ = ask(expansion=FakeExpansion(expanded(["a", "b"], [], status={1: "error", 2: "error"})))
    assert b.text == "(memory was not consulted: the search failed)"


def test_every_search_late_is_the_not_consulted_line_in_time():
    b, _, _ = ask(expansion=FakeExpansion(expanded(["a", "b"], [], status={1: "late", 2: "late"}, deadline_hit=True)))
    assert b.text == "(memory was not consulted: the search did not answer in time)"


def test_a_late_search_is_dropped_the_others_stand():
    result = expanded(["a", "b"], [item("a")], status={2: "late"}, deadline_hit=True)
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert "(1) a" in b.text and "(2) b" not in b.text and b.injected == 1


def test_a_result_the_plugin_cannot_read_is_the_not_consulted_line():
    b, _, _ = ask(expansion=FakeExpansion(lambda prompt, kw: object()))
    assert b.text == "(memory was not consulted: the search failed)"


# ---------------------------------------------------------------------------- the deadline

def test_the_library_is_given_what_is_left_of_the_deadline():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9), delay=0.3)
    expansion = FakeExpansion()
    run(ROOF_Q, make_deps(expansion, jev), settings=pc.Settings(mode="contract", decision_timeout_s=1.0, deadline_s=2.0))
    assert 1.4 < expansion.kwargs[0]["deadline"] < 1.8                 # 2.0 s minus the decision's 0.3 s


def test_a_library_slower_than_the_deadline_is_the_not_consulted_line_in_time_and_its_late_answer_dropped():
    expansion = FakeExpansion(delay=1.5)
    s = pc.Settings(mode="contract", decision_timeout_s=0.5, deadline_s=0.4)
    t0 = time.monotonic()
    b, _, _ = ask(expansion=expansion, settings=s)
    assert time.monotonic() - t0 < 1.0
    assert b.text == "(memory was not consulted: the search did not answer in time)"
    time.sleep(1.3)                                                      # the late call finishes: nothing changes
    assert b.text == "(memory was not consulted: the search did not answer in time)" and b.injected == 0


def test_no_time_left_after_the_decision_means_no_library_call():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9), delay=1.0)
    expansion = FakeExpansion()
    s = pc.Settings(mode="contract", decision_timeout_s=1.0, deadline_s=0.3)
    t0 = time.monotonic()
    b = run(ROOF_Q, make_deps(expansion, jev), settings=s)
    assert time.monotonic() - t0 < 0.8 and not expansion.calls
    assert b.text == "(memory was not consulted: the search did not answer in time)"


def test_the_stage_budget_is_capped_by_what_is_left_of_the_deadline():
    now = [0.0]
    deps = pc.Deps(expand=FakeExpansion(), clock=lambda: now[0])
    r = pc._Run(deps, pc.Settings(deadline_s=7.0), pc.load_contract(), started=0.0)
    assert r.budget(1.0) == 1.0
    now[0] = 6.5
    assert r.budget(1.0) == 0.5
    now[0] = 8.0
    assert r.budget(1.0) == 0.0


# ---------------------------------------------------------------------------- calls run in parallel in the library: the plugin adds none

def test_one_decision_and_one_expansion_per_asked_turn_and_nothing_else():
    _, expansion, jev = ask()
    assert len(jev.requests) == 1 and len(expansion.calls) == 1


def test_cost_adds_the_decision_and_the_librarys_accounting():
    b, _, _ = ask(expansion=FakeExpansion(expanded(["roof"], [item("a")], cost=0.0007)))
    assert b.cost_usd == pytest.approx(0.00004 + 0.0007)
    assert b.stages_ms["expand"] >= 0 and "decide" in b.stages_ms and b.stages_ms["lib_writer"] == 11


def test_the_library_s_fallbacks_are_kept_on_the_brief_for_diagnosis_without_logging():
    result = expanded(["roof"], [item("a")], fallbacks=["writer: the writer failed (TimeoutError): one search with the prompt itself"])
    b, _, _ = ask(expansion=FakeExpansion(result))
    assert b.fallbacks == ["writer: the writer failed (TimeoutError): one search with the prompt itself"]


# ---------------------------------------------------------------------------- the dead code is gone

@pytest.mark.parametrize("name", [
    "write_questions", "Written", "fallback_question", "jev_score", "JEV_TASK", "parse_questions", "valid_window",
    "quoted_phrases", "keep_quoted", "with_phrase_queries", "is_plain_question", "default_writer_llm", "Scope", "Found",
    "allowed", "merge_lists", "build_block", "cited_entries", "estimate_tokens", "MAX_QUESTIONS", "MAX_PHRASE_QUERIES",
    "SEARCH_LIMIT", "FALLBACK_PAGE", "WRITER_TIMEOUT_S", "SEARCH_TIMEOUT_S", "PLAIN_QUESTION_CHARS", "SENTENCE_POOL",
    "NEED_SOURCES", "ALL_SOURCES"])
def test_the_plugins_own_question_writing_and_its_helpers_are_gone(name):
    assert not hasattr(pc, name), name


def test_settings_lost_the_writer_and_search_timeouts():
    s = pc.Settings()
    assert not hasattr(s, "writer_timeout_s") and not hasattr(s, "search_timeout_s")
    assert (s.decision_timeout_s, s.deadline_s) == (1.0, 5.0)


# ---------------------------------------------------------------------------- provider: switch, settings, never fails

class FakeMemory:
    """The provider's Memory: ``expand_and_recall`` is the fake expansion. It has nothing else: the library has no
    ``recall_synth`` (the old synthesised prefetch) and the prefetch never searches by itself."""

    def __init__(self, expansion):
        self.expansion = expansion

    def expand_and_recall(self, prompt, kind="chat", **kw):
        return self.expansion(prompt, kind, **kw)

    def search(self, *a, **k):
        raise AssertionError("the contract prefetch never calls search: it is one expansion")

    def _tracer(self, *a, **k):
        pass


ENV = ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_PREFETCH_DECISION_MODEL",
       "PROSPECTA_PREFETCH_QUESTION_MODEL", "PROSPECTA_PREFETCH_CONTRACT", "PROSPECTA_PREFETCH_MAX_TOKENS",
       "PROSPECTA_PREFETCH_DEADLINE", "PROSPECTA_PREFETCH_DECISION_TIMEOUT", "PROSPECTA_PREFETCH_WRITER_TIMEOUT",
       "PROSPECTA_PREFETCH_SEARCH_TIMEOUT", "PROSPECTA_DECISION_MODEL", "PROSPECTA_QUERY_WRITER_MODEL",
       "PROSPECTA_RERANKER_MODEL", "PROSPECTA_RECALL_STAGE3", "PROSPECTA_TIMEZONE", "PROSPECTA_JEV_MODEL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


def provider(plugin_module, *, config=None, expansion=None, jev=None):
    p = plugin_module.ProspectaProvider()
    p._config = dict(config or {})
    p._session_id = "s1"
    p.expansion = expansion if expansion is not None else FakeExpansion()
    p._memory = FakeMemory(p.expansion)
    p._jev_transport = jev
    p._today = lambda: TODAY
    p._prefetch_cfg = p._resolve_prefetch_settings()
    return p


CONTRACT_ON = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract"}}}


def test_defaults_are_contract_and_flat_nested_and_env_settings_read(plugin_module, monkeypatch):
    """``contract`` is the only prefetch there is; ``off`` is no prefetch. ``prefetch_enabled`` (default off) is
    the master switch, so the default mode costs nothing until a profile turns the prefetch on."""
    p = plugin_module.ProspectaProvider()
    s = p._resolve_prefetch_settings()
    assert (s.mode, s.decision_model, s.query_writer_model, s.max_tokens, s.deadline_s) == (
        "contract", None, None, 0, 5.0) and s.active
    assert pc.MODES == ("contract", "off") and pc.DEFAULT_MODE == "contract"
    assert p._prefetch_enabled() is False
    assert p._reranker_model() is None and p._decision_model() is None
    p._config = {"recall.prefetch.mode": "contract", "recall.prefetch.max_tokens": "800",
                 "recall.prefetch.contract_path": "/x/c.md"}
    s = p._resolve_prefetch_settings()
    assert (s.mode, s.max_tokens, s.contract_path) == ("contract", 800, "/x/c.md")
    p._config = {"recall": {"prefetch": {"mode": "contract"}}}
    assert p._resolve_prefetch_settings().mode == "contract"
    monkeypatch.setenv("PROSPECTA_PREFETCH_MODE", "off")
    s = p._resolve_prefetch_settings()
    assert s.mode == "off" and not s.active


def test_the_models_are_the_profiles_only_the_library_resolves_the_environment_and_the_defaults(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    monkeypatch.setenv("PROSPECTA_DECISION_MODEL", "typesafe/jev-env")
    monkeypatch.setenv("PROSPECTA_QUERY_WRITER_MODEL", "openrouter/x/writer-env")
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "x/reranker-env")
    s = p._resolve_prefetch_settings()
    assert (s.decision_model, s.query_writer_model, p._decision_model(), p._reranker_model()) == (None,) * 4
    p._config = {"decision_model": "typesafe/jev-9", "query_writer_model": "openrouter/x/writer-cfg",
                 "reranker_model": "x/reranker-cfg"}
    s = p._resolve_prefetch_settings()
    assert (s.decision_model, s.query_writer_model, p._reranker_model()) == (
        "typesafe/jev-9", "openrouter/x/writer-cfg", "x/reranker-cfg")
    p._config = {"decision_model": "  ", "reranker_model": 5}              # blank or not a string: not set
    assert (p._decision_model(), p._reranker_model()) == (None, None)


def test_the_writer_model_is_read_from_recall_prefetch_too_and_that_wins(plugin_module):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"query_writer_model": "openrouter/x/nested"}}}
    assert p._resolve_prefetch_settings().query_writer_model == "openrouter/x/nested"
    p._config = {"recall.prefetch.query_writer_model": "openrouter/x/flat"}
    assert p._resolve_prefetch_settings().query_writer_model == "openrouter/x/flat"
    p._config = {"query_writer_model": "openrouter/x/top", "recall.prefetch.query_writer_model": "openrouter/x/flat"}
    assert p._resolve_prefetch_settings().query_writer_model == "openrouter/x/flat"


def test_the_old_question_model_key_is_a_deprecated_alias_with_one_warning_line(plugin_module, caplog):
    for config in ({"recall": {"prefetch": {"question_model": "openrouter/x/old"}}},
                   {"recall.prefetch.question_model": "openrouter/x/old"}):
        p = plugin_module.ProspectaProvider()
        p._config = dict(config)
        assert p._resolve_prefetch_settings().query_writer_model == "openrouter/x/old"      # it still works
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            p._warn_renamed_settings()
        assert len(caplog.records) == 1                                                       # ONE warning
        msg = caplog.records[0].getMessage()
        assert "\n" not in msg and "recall.prefetch.question_model is deprecated" in msg
        assert "recall.prefetch.query_writer_model" in msg and "openrouter/x/old" not in msg


def test_the_new_key_wins_over_the_deprecated_alias_and_the_warning_still_names_it(plugin_module, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"question_model": "openrouter/x/old",
                                         "query_writer_model": "openrouter/x/new"}}}
    assert p._resolve_prefetch_settings().query_writer_model == "openrouter/x/new"
    with caplog.at_level(logging.WARNING):
        p._warn_renamed_settings()
    assert len(caplog.records) == 1 and "deprecated" in caplog.text


def test_the_alias_is_warned_once_at_initialise_not_on_every_resolve_or_turn(plugin_module, caplog):
    p = provider(plugin_module, config={**CONTRACT_ON, "recall": {"prefetch": {
        "mode": "contract", "question_model": "openrouter/x/old"}}})
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        p._prefetch_cfg = p._resolve_prefetch_settings()
        for _ in range(3):
            p.prefetch("What did we decide about the Henley branch roof?")
    assert not caplog.records                                          # resolving and prefetching are silent


def test_the_other_old_settings_stay_ignored_loudly_one_line_each(plugin_module, monkeypatch, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall.prefetch.decision_model": "typesafe/jev-old",
                 "recall": {"prefetch": {"writer_timeout_seconds": 2, "search_timeout_seconds": "3"}}}
    monkeypatch.setenv("PROSPECTA_PREFETCH_QUESTION_MODEL", "openrouter/x/old-env")
    monkeypatch.setenv("PROSPECTA_PREFETCH_SEARCH_TIMEOUT", "2")
    with caplog.at_level(logging.WARNING):
        p._warn_renamed_settings()
    assert p._resolve_prefetch_settings().decision_model is None                          # not applied
    for old in ("recall.prefetch.decision_model is ignored now; set decision_model",
                "PROSPECTA_PREFETCH_QUESTION_MODEL is ignored now; set PROSPECTA_QUERY_WRITER_MODEL",
                "recall.prefetch.writer_timeout_seconds is ignored now",
                "recall.prefetch.search_timeout_seconds is ignored now",
                "PROSPECTA_PREFETCH_SEARCH_TIMEOUT is ignored now"):
        assert old in caplog.text, old
    assert all("\n" not in r.getMessage() for r in caplog.records)
    caplog.clear()
    monkeypatch.delenv("PROSPECTA_PREFETCH_QUESTION_MODEL")
    monkeypatch.delenv("PROSPECTA_PREFETCH_SEARCH_TIMEOUT")
    plugin_module.ProspectaProvider()._warn_renamed_settings()                           # nothing set: silent
    assert not caplog.records


def test_bad_mode_is_the_default_with_a_warning(plugin_module, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"mode": "full", "max_tokens": "lots"}}}
    with caplog.at_level(logging.WARNING):
        s = p._resolve_prefetch_settings()
    assert s.mode == "contract" and s.max_tokens == 0 and "full" in caplog.text and "lots" in caplog.text
    assert "'contract'" in caplog.text and "'off'" in caplog.text        # it says what the two modes are


def test_mode_is_read_at_initialize_so_the_next_session_picks_it_up(plugin_module):
    p = plugin_module.ProspectaProvider()
    p._config = {"prefetch_enabled": "true"}
    assert p._resolve_prefetch_settings().mode == "contract"
    p._config = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "off"}}}
    assert p._resolve_prefetch_settings().mode == "off"


def test_prefetch_disabled_changes_nothing_even_with_mode_contract(plugin_module):
    jev, expansion = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeExpansion()
    p = provider(plugin_module, config={"recall": {"prefetch": {"mode": "contract"}}}, expansion=expansion, jev=jev)
    assert p.prefetch(ROOF_Q) == ""
    assert not jev.requests and not expansion.calls
    p._config = {"prefetch_enabled": "false", "recall": {"prefetch": {"mode": "contract"}}}
    assert p.prefetch(ROOF_Q) == "" and not jev.requests


def test_mode_off_is_no_prefetch_at_all_even_with_the_master_switch_on(plugin_module, caplog):
    """``off`` used to be the old synthesised answer (recall_synth: gone from the library). It is no prefetch now: no
    decision call, no search, nothing injected, nothing prepared ahead, no warning."""
    jev, expansion = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeExpansion()
    config = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "off"}}}
    p = provider(plugin_module, config=config, expansion=expansion, jev=jev)
    with caplog.at_level(logging.DEBUG):
        assert p.prefetch("What did we decide?") == ""
        p.queue_prefetch("anything")
        time.sleep(0.1)
    assert not jev.requests and not expansion.calls and not caplog.records


def test_a_profile_that_enables_the_prefetch_and_names_no_mode_gets_the_contract(plugin_module):
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    p = provider(plugin_module, config={"prefetch_enabled": "true"}, jev=jev,
                 expansion=FakeExpansion(expanded(["Henley roof"], [item("roof", "The roof is to be replaced in June.")])))
    assert "The roof is to be replaced in June." in p.prefetch(ROOF_Q) and len(jev.requests) == 1


def test_the_old_synthesised_prefetch_is_gone_from_the_provider(plugin_module):
    assert not hasattr(plugin_module.ProspectaProvider, "_prefetch_synth")


def test_mode_contract_never_synthesises_and_never_searches_itself(plugin_module):
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev,
                 expansion=FakeExpansion(expanded(["Henley roof"], [item("roof", "The roof is to be replaced in June.")])))
    out = p.prefetch(ROOF_Q)
    assert "The roof is to be replaced in June." in out
    assert p._last_brief.decision == "ask" and "<prospecta-recall>" not in out


def test_a_successful_turn_writes_no_recall_log_line(plugin_module, caplog):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev,
                 expansion=FakeExpansion(expanded(["roof plan"], [item("roof", "The roof is to be replaced in June.")])))
    with caplog.at_level(logging.DEBUG):
        assert "The roof is to be replaced" in p.prefetch("and the roof?")
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert not [r for r in caplog.records if "[recall]" in r.getMessage() or "brief" in r.getMessage()]
    assert p.expansion.kwargs[0]["record"] is False


def test_prefetch_never_raises_whatever_fails(plugin_module, monkeypatch, caplog):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("new_ask", 0.9)))

    class Weird(Exception):
        pass

    for err in (RuntimeError("secret words: the Panama deal"), Weird("x"), KeyError("k"), ValueError(), OSError(5, "io")):
        def boom(*a, **k):
            raise err
        monkeypatch.setattr(plugin_module._pc, "run_contract", boom)   # the plugin's own copy of the module
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert p.prefetch(ROOF_Q) == ""
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "Panama" not in caplog.text and "Henley" not in caplog.text    # no text in the log
    monkeypatch.undo()
    p._memory = None
    assert p.prefetch("anything at all here") == ""
    p2 = provider(plugin_module, config={**CONTRACT_ON, "recall": {"prefetch": {"mode": "contract",
                                                                              "contract_path": "/no/such.md"}}})
    p2._prefetch_cfg = p2._resolve_prefetch_settings()
    assert p2.prefetch(ROOF_Q) == ""


def test_a_library_without_the_entry_point_goes_without_memory_with_one_warning_line(plugin_module, caplog):
    from types import SimpleNamespace
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("new_ask", 0.9)))
    p._memory = SimpleNamespace(search=lambda *a, **k: [])          # the pinned library before PR 85: no entry point
    with caplog.at_level(logging.INFO):
        assert p.prefetch(ROOF_Q) == ""
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "AttributeError" in warnings[0].getMessage() and "Henley" not in caplog.text


def test_every_stage_failing_still_returns_a_string(plugin_module):
    p = provider(plugin_module, config=CONTRACT_ON, expansion=FakeExpansion(fail=RuntimeError("db")),
                 jev=FakeJev(fail_decision=RuntimeError("down")))
    out = p.prefetch(ROOF_Q)
    assert isinstance(out, str) and out == "(memory was not consulted: the search failed)"


def test_the_deadline_is_honoured_and_a_late_answer_is_dropped(plugin_module, caplog):
    expansion = FakeExpansion(delay=1.5)
    p = provider(plugin_module, config=CONTRACT_ON, expansion=expansion,
                 jev=FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95)))
    p._prefetch_cfg = pc.Settings(mode="contract", deadline_s=0.4)
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        out = p.prefetch(ROOF_Q)
    assert out == "(memory was not consulted: the search did not answer in time)"
    assert time.monotonic() - t0 < 1.0
    assert "Henley" not in caplog.text
    time.sleep(1.4)                                  # the late call finishes: nothing is injected afterwards
    assert p._last_brief.text == out


def test_a_hung_run_is_ended_by_the_provider_and_goes_without_memory(plugin_module, monkeypatch, caplog):
    p = provider(plugin_module, config=CONTRACT_ON)
    p._prefetch_cfg = pc.Settings(mode="contract", deadline_s=0.3)
    monkeypatch.setattr(plugin_module._pc, "run_contract", lambda *a, **k: time.sleep(2))
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        assert p.prefetch(ROOF_Q) == ""
    assert time.monotonic() - t0 < 1.0
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and "Henley" not in caplog.text


def test_the_deadline_is_five_seconds_by_default_and_never_above_seven(plugin_module):
    s = plugin_module.ProspectaProvider()._resolve_prefetch_settings()
    assert s.deadline_s == 5.0
    p = plugin_module.ProspectaProvider()
    p._config = {"recall.prefetch.deadline_seconds": "30"}
    assert p._resolve_prefetch_settings().deadline_s == 7.0
    p._config = {"recall.prefetch.deadline_seconds": "6.5"}
    assert p._resolve_prefetch_settings().deadline_s == 6.5
    assert pc.MAX_DEADLINE_S == 7.0


def test_the_library_never_gets_more_than_the_cap(plugin_module):
    p = provider(plugin_module, config={**CONTRACT_ON, "recall.prefetch.deadline_seconds": "30"},
                 jev=FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95)))
    p.prefetch(ROOF_Q)
    assert 0 < p.expansion.kwargs[0]["deadline"] <= 7.0


# ---------------------------------------------------------------------------- the stage limit the plugin still owns

def test_the_decision_limit_default_and_overrides(plugin_module, monkeypatch):
    assert plugin_module.ProspectaProvider()._resolve_prefetch_settings().decision_timeout_s == 1.0
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"decision_timeout_seconds": "3.25"}}}
    assert p._resolve_prefetch_settings().decision_timeout_s == 3.25
    p._config = {"recall.prefetch.decision_timeout_seconds": 2}
    assert p._resolve_prefetch_settings().decision_timeout_s == 2.0
    monkeypatch.setenv("PROSPECTA_PREFETCH_DECISION_TIMEOUT", "0.75")
    assert p._resolve_prefetch_settings().decision_timeout_s == 0.75


@pytest.mark.parametrize("bad", ["soon", "-1", "0", "nan", "inf"])
def test_a_bad_decision_limit_falls_back_with_one_warning(plugin_module, caplog, bad):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"decision_timeout_seconds": bad}}}
    with caplog.at_level(logging.WARNING):
        s = p._resolve_prefetch_settings()
    assert s.decision_timeout_s == 1.0
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and "decision_timeout_seconds" in caplog.text


# ---------------------------------------------------------------------------- the agent's last answer and ahead

def _sync(p, user, assistant, monkeypatch, sid="s1"):
    p._retained = getattr(p, "_retained", [])
    monkeypatch.setattr(p, "_retain", lambda content, **k: p._retained.append(k) or "doc")
    p.sync_turn(user, assistant, session_id=sid)
    p._sync_thread.join(2)
    time.sleep(0.003)             # a turn's source ends in the millisecond: keep two turns apart


def _wait(cond, secs=3.0):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_sync_turn_keeps_the_agents_last_answer_per_session_and_the_library_gets_it(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev)
    _sync(p, "hello there", OPENER_AGENT_LAST, monkeypatch)
    _sync(p, "other", "A different answer for another session.", monkeypatch, sid="s2")
    p.prefetch("ok do it please")
    assert jev.decisions()[0]["state"]["agent_last"] == OPENER_AGENT_LAST
    assert p.expansion.kwargs[0]["agent_last"] == OPENER_AGENT_LAST
    assert "1 earlier" in jev.decisions()[0]["state"]["in_view"]
    p.prefetch("and then", session_id="s2")
    assert jev.decisions()[1]["state"]["agent_last"] == "A different answer for another session."
    assert p.expansion.kwargs[1]["agent_last"] == "A different answer for another session."


def test_ahead_is_keyed_on_the_agents_answer_and_spares_the_next_turn(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1, need_library=1))
    expansion = FakeExpansion(expanded(["village library closure"],
                                       [item("closure", "The library closed on 4 June after the roof failed.")]))
    p = provider(plugin_module, config=CONTRACT_ON, expansion=expansion, jev=jev)
    _sync(p, "can you help with an explainer?", OPENER_AGENT_LAST, monkeypatch)
    p.queue_prefetch("can you help with an explainer?", session_id="s1")           # the person's words
    assert _wait(lambda: "s1" in p._ahead)
    (prompt, kind, kw) = expansion.calls[0]
    assert kw["agent_last"] == OPENER_AGENT_LAST and prompt == pc.AHEAD_PROMPT      # keyed on the agent's answer
    assert "can you help with an explainer?" not in prompt and kw["gate"] is False and kw["record"] is False
    assert not jev.decisions()                                                      # nobody replied: no decision
    assert p._ahead["s1"].key == pc.ahead_key(OPENER_AGENT_LAST)
    n = len(expansion.calls)
    out = p.prefetch("ok do it")                                                    # the go-ahead
    assert "The library closed on 4 June" in out
    assert len(expansion.calls) == n                                                # used, not recomputed
    assert p._last_brief.ahead_used and "s1" not in p._ahead                        # one use


def test_ahead_is_not_used_when_the_agent_said_something_else(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev,
                 expansion=FakeExpansion(expanded(["closure"], [item("closure", "closure text")])))
    _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)
    p.queue_prefetch("x", session_id="s1")
    assert _wait(lambda: "s1" in p._ahead)
    _sync(p, "ok do it", "Done. The draft is in the chat.", monkeypatch)           # a new answer: the old key is stale
    n = len(p.expansion.calls)
    p.prefetch("ok do it")
    assert not p._last_brief.ahead_used and len(p.expansion.calls) == n + 1


def test_a_proposal_that_finds_nothing_keeps_no_ahead_entry(plugin_module, monkeypatch):
    p = provider(plugin_module, config=CONTRACT_ON, expansion=FakeExpansion(expanded(["closure"], [])))
    _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)
    p.queue_prefetch("x", session_id="s1")
    assert _wait(lambda: p.expansion.calls)
    time.sleep(0.2)
    assert "s1" not in p._ahead


def test_no_ahead_unless_the_agent_ended_on_a_proposal(plugin_module, monkeypatch):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("go_ahead", 0.9)))
    _sync(p, "x", "The draft is finished and saved in the chat.", monkeypatch)
    p.queue_prefetch("x", session_id="s1")
    time.sleep(0.2)
    assert not p.expansion.calls and "s1" not in p._ahead


def test_no_ahead_with_the_switch_off(plugin_module, monkeypatch):
    for config in ({"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "off"}}},     # mode off
                   {"recall": {"prefetch": {"mode": "contract"}}}):                           # master switch off
        p = provider(plugin_module, config=config, jev=FakeJev(decision=jev_decision("go_ahead", 0.9)))
        _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)
        p.queue_prefetch("x", session_id="s1")
        time.sleep(0.2)
        assert not p.expansion.calls and not p._ahead


def test_queue_prefetch_never_raises(plugin_module, monkeypatch):
    p = provider(plugin_module, config=CONTRACT_ON)
    monkeypatch.setattr(p, "_last_exchange", lambda sid: (_ for _ in ()).throw(RuntimeError("x")))
    p.queue_prefetch("anything", session_id="s1")


# ---------------------------------------------------------------------------- the last answer after a restart

def _state_db(path, rows):
    import sqlite3
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL)")
    c.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)", rows)
    c.commit()
    c.close()


def _restarted(plugin_module, tmp_path, rows=None, **kw):
    if rows is not None:
        _state_db(tmp_path / "state.db", rows)
    p = provider(plugin_module, config=CONTRACT_ON, **kw)
    p._hermes_home = tmp_path
    return p


def test_recovery_returns_latest_assistant_message_whole(plugin_module, tmp_path):
    long = "x" * 5000
    p = _restarted(plugin_module, tmp_path, [
        ("s1", "assistant", "old", 1), ("s1", "assistant", long, 3), ("s1", "user", "newer user", 4),
        ("s1", "tool", "tool out", 5), ("s1", "assistant", "", 6), ("s1", "assistant", None, 7),
        ("s2", "assistant", "other session", 9)])
    assert p._read_last_assistant("s1") == long


def test_recovery_survives_missing_file_table_and_lock(plugin_module, tmp_path):
    import sqlite3
    p = _restarted(plugin_module, tmp_path)
    assert p._read_last_assistant("s1") == ""                       # no file
    sqlite3.connect(tmp_path / "state.db").close()
    assert p._read_last_assistant("s1") == ""                       # no table
    _state_db(tmp_path / "other.db", [])
    (tmp_path / "state.db").write_bytes(b"not a database" * 100)
    assert p._read_last_assistant("s1") == ""                       # garbage
    (tmp_path / "state.db").unlink()
    _state_db(tmp_path / "state.db", [("s1", "assistant", "hi", 1)])
    locker = sqlite3.connect(tmp_path / "state.db", isolation_level=None)
    locker.execute("BEGIN EXCLUSIVE")
    try:
        p._read_last_assistant("s1")                                # lock: no raise, no hang
    finally:
        locker.close()
    p._hermes_home = None
    assert p._read_last_assistant("s1") == ""


def test_recovery_is_read_only(plugin_module, tmp_path):
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "hello", 1)])
    db = tmp_path / "state.db"
    before, mtime = db.read_bytes(), db.stat().st_mtime_ns
    assert p._read_last_assistant("s1") == "hello"
    assert db.read_bytes() == before and db.stat().st_mtime_ns == mtime
    assert sorted(x.name for x in tmp_path.iterdir()) == ["state.db"]


def test_the_recovered_last_answer_reaches_the_decision_and_the_library_once_per_session(plugin_module, tmp_path,
                                                                                         monkeypatch):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "We met Ana at the cafe.", 1)], jev=jev)
    calls = []
    real = p._read_last_assistant
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: calls.append(sid) or real(sid))
    p.prefetch("do you remember who i saw it with?")
    p.prefetch("and then?")
    assert calls == ["s1"]
    assert jev.decisions()[0]["state"]["agent_last"] == "We met Ana at the cafe."
    assert [kw["agent_last"] for kw in p.expansion.kwargs] == ["We met Ana at the cafe."] * 2


def test_recovery_does_not_run_when_an_answer_is_held(plugin_module, tmp_path, monkeypatch):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "from the db", 1)], jev=jev)
    _sync(p, "hello", "kept answer", monkeypatch)
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: pytest.fail("must not read"))
    p.prefetch("and that?")
    assert jev.decisions()[0]["state"]["agent_last"] == "kept answer" and p.expansion.kwargs[0]["agent_last"] == "kept answer"


def test_recovery_never_fails_the_turn(plugin_module, tmp_path, monkeypatch):
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "x", 1)])
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: (_ for _ in ()).throw(RuntimeError("boom")))
    p._recover_last_answer("s1")
    assert p._last_exchange("s1") == ("", 0)


# ---------------------------------------------------------------------------- no echoes: the thread's last turns, in SQL

def _turns_provider(plugin_module, monkeypatch, n=3, config=None):
    p = provider(plugin_module, config=config if config is not None else CONTRACT_ON,
                 expansion=FakeExpansion(expanded(["roof"], [item("roof", "The roof is to be replaced in June.")])),
                 jev=FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1)))
    for i in range(1, n + 1):
        _sync(p, f"user line {i}", f"assistant reply {i}", monkeypatch)
    return p


def test_the_last_two_turns_are_excluded_for_every_search_and_older_turns_stay_eligible(plugin_module, monkeypatch):
    p = _turns_provider(plugin_module, monkeypatch, n=3)
    first, second, third = [k["source"] for k in p._retained]
    assert all(src.startswith("turn:s1:") for src in (first, second, third)) and len({first, second, third}) == 3
    assert p._echo_sources("s1") == [second, third]
    assert "The roof is to be replaced in June." in p.prefetch("and the roof?")
    (kw,) = p.expansion.kwargs
    assert kw["excluded_sources"] == [second, third] and first not in kw["excluded_sources"]


def test_a_session_with_one_turn_excludes_that_one(plugin_module, monkeypatch):
    p = _turns_provider(plugin_module, monkeypatch, n=1)
    p.prefetch("and the roof?")
    assert p.expansion.kwargs[0]["excluded_sources"] == [p._retained[0]["source"]]


def test_the_first_message_of_a_session_excludes_nothing(plugin_module, monkeypatch):
    p = _turns_provider(plugin_module, monkeypatch, n=0)
    p.prefetch("and the roof?")
    assert p.expansion.kwargs[0]["excluded_sources"] is None and p.expansion.kwargs[0]["date_from"] is None


def test_each_session_excludes_only_its_own_turns(plugin_module, monkeypatch):
    p = _turns_provider(plugin_module, monkeypatch, n=2)
    _sync(p, "other user", "other reply", monkeypatch, sid="s2")
    p.prefetch("and the roof?")
    p.prefetch("and the roof?", session_id="s2")
    s1 = [k["source"] for k in p._retained if k["source"].startswith("turn:s1:")]
    assert p.expansion.kwargs[0]["excluded_sources"] == s1
    assert p.expansion.kwargs[1]["excluded_sources"] == [p._retained[-1]["source"]]
    assert p.expansion.kwargs[1]["excluded_sources"][0].startswith("turn:s2:")


def test_drop_echoes_off_sends_no_exclusion(plugin_module, monkeypatch):
    cfg = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract", "drop_echoes": False}}}
    p = _turns_provider(plugin_module, monkeypatch, n=3, config=cfg)
    p.prefetch("and the roof?")
    assert p.expansion.kwargs[0]["excluded_sources"] is None


def test_drop_echoes_env_off_and_default_on(plugin_module, monkeypatch):
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is True
    monkeypatch.setenv("PROSPECTA_PREFETCH_DROP_ECHOES", "false")
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is False
    monkeypatch.setenv("PROSPECTA_PREFETCH_DROP_ECHOES", "true")
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is True


def test_after_a_restart_the_last_two_turns_come_from_the_bank_once(plugin_module, tmp_path, monkeypatch):
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "We spoke of the roof.", 1)],
                   jev=FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1)))
    asked = []
    monkeypatch.setattr(p, "_read_turn_sources", lambda sid: asked.append(sid) or ["turn:s1:11", "turn:s1:12"])
    p.prefetch("and the roof?")
    p.prefetch("and the roof again?")
    assert asked == ["s1"]
    assert [k["excluded_sources"] for k in p.expansion.kwargs] == [["turn:s1:11", "turn:s1:12"]] * 2


def test_turn_recovery_is_silent_on_errors_and_never_holds_the_turn(plugin_module, tmp_path, monkeypatch):
    p = _restarted(plugin_module, tmp_path)
    assert p._read_turn_sources("s1") == []                 # a Memory with no pool
    p._memory = None
    assert p._read_turn_sources("s1") == [] and p._read_turn_sources("") == []
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "x", 1)])

    def slow(sid):
        time.sleep(3)
        return ["turn:s1:1"]
    monkeypatch.setattr(p, "_read_turn_sources", slow)
    t0 = time.monotonic()
    p._recover_last_answer("s1")
    assert time.monotonic() - t0 < 1.5 and p._last_exchange("s1") == ("x", 0) and p._echo_sources("s1") == []


def test_every_search_carries_the_same_exclusion_and_window_because_they_are_arguments_of_one_call(plugin_module, monkeypatch):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1,
                                                                                    scope="past_week")))
    _sync(p, "user line 1", "assistant reply 1", monkeypatch)
    _sync(p, "user line 2", "assistant reply 2", monkeypatch)
    out = p.prefetch("and the roof?")
    (kw,) = p.expansion.kwargs                                 # ONE call: the library gives every search these filters
    assert kw["excluded_sources"] == [r["source"] for r in p._retained]
    assert (kw["date_from"], kw["date_to"]) == ("2026-10-10", "2026-10-17")
    assert "Only notes dated 2026-10-10 to 2026-10-17 were searched." in out


# ---------------------------------------------------------------------------- a skill / cron turn gets no automatic recall
#
# Invented text of the real shapes. Hermes builds them in agent/skill_commands.py (single skill, stacked,
# preloaded), agent/skill_bundles.py (bundle), cron/scheduler.py (the cron hint, the skill block, the
# skipped-skill notice) and gateway/run.py (auto-loaded skill). Core's prefetch_all keeps only the user's
# instruction of a skill turn, so for a cron wake the plugin sees the cron hint + the job's prompt.

CRON_HINT = ("[IMPORTANT: You are running as a scheduled cron job. DELIVERY: Your final response will be "
             "automatically delivered to the user — do NOT use send_message or try to deliver the output "
             "yourself. Just produce your report/output as your final response and the system handles the "
             "rest. SILENT: If there is genuinely nothing new to report, respond with exactly \"[SILENT]\" "
             "(nothing else) to suppress delivery. Never combine [SILENT] with content — either report your "
             "findings normally, or say [SILENT] and nothing more.]\n\n")
JOB_PROMPT = ("Look through the harbour-notes folder and the kitchen log for anything the person left "
              "unfinished; session notes live under /srv/example/substrate. Report only what is new.")
SKILL_BODY = ("# Daily harbour check\n\nWhen woken, open the session log, list open items and write a short "
              "report. Never deliver it yourself.\n")
SKILL_LINE = ('[IMPORTANT: The user has invoked the "harbour-check" skill, indicating they want you to follow its '
              'instructions. The full skill content is loaded below.]')
INSTRUCTION_MARK = "The user has provided the following instruction alongside the skill invocation: "

SKILL_TURNS = {
    # what Hermes builds (cron/scheduler.py:3097 + append_user_instruction; skill_commands.py:631)
    "cron-skill-wake": f"{SKILL_LINE}\n\n{SKILL_BODY}\n{INSTRUCTION_MARK}{CRON_HINT}{JOB_PROMPT}",
    "slash-skill": f"{SKILL_LINE}\n\n{SKILL_BODY}\n{INSTRUCTION_MARK}tidy the harbour notes",
    "slash-skill-bare": f"{SKILL_LINE}\n\n{SKILL_BODY}",
    # skill_bundles.py:347 and skill_commands.py:760
    "bundle": ('[IMPORTANT: The user has invoked the "harbour" skill bundle, loading 2 skills together. Treat '
               'every skill below as active guidance for this turn.]\n\nBundle: harbour\n\nUser instruction: '
               'tidy up\n\n[Loaded as part of the harbour bundle: harbour-check]\n' + SKILL_BODY),
    "stacked": ('[IMPORTANT: The user has invoked the "/harbour-check /kitchen-log" stacked skill bundle, loading '
                '2 skills together. Treat every skill below as active guidance for this turn.]\n\n'
                'Skills loaded: harbour-check, kitchen-log\n\nUser instruction: tidy up\n\n'
                '[Loaded as part of the stacked skills: harbour-check]\n' + SKILL_BODY),
    # cron/scheduler.py:3105: core does NOT strip this one (the message does not begin with the invoke marker)
    "cron-skipped-notice": ("[IMPORTANT: The following skill(s) were listed for this job but could not be found "
                            "and were skipped: ghost-skill. Start your response with a brief notice so the user "
                            "is aware, e.g.: '⚠️ Skill(s) not found and skipped: ghost-skill']\n\n" + SKILL_LINE +
                            f"\n\n{SKILL_BODY}\n{INSTRUCTION_MARK}{CRON_HINT}{JOB_PROMPT}"),
    # cron/scheduler.py:3014: a cron job WITHOUT a skill, and the remainder core hands over after stripping a skill
    "cron-hint-only": CRON_HINT + JOB_PROMPT,
    "cron-hint-with-notepad": CRON_HINT + "## Notepad\n- last seen: Tuesday\n\n" + JOB_PROMPT,
    # skill_commands.py:826 (preloaded), gateway/run.py:17984 (auto-loaded for a topic/channel)
    "preloaded": ('[IMPORTANT: The user launched this CLI session with the "harbour-check" skill preloaded. Treat '
                  'its instructions as active guidance for the duration of this session unless the user '
                  'overrides them.]\n\n' + SKILL_BODY),
    "auto-loaded": ('[IMPORTANT: The "harbour-check" skill is auto-loaded. Follow its instructions for this '
                    'session.]\n\n' + SKILL_BODY + "\n\nWhat did we decide about the Henley branch roof?"),
}

# Ordinary messages: they may mention a skill, a marker or a cron job, but they do not BEGIN as one.
ORDINARY_LINES = [
    ROOF_Q,
    "Can you write a skill that checks the Henley branch roof every morning?",
    'What does it mean when a prompt says [IMPORTANT: The user has invoked the "harbour-check" skill? Henley roof.',
    f'Please explain this text from the log: {SKILL_LINE} and why the Henley roof came up.',
    "I set up a scheduled cron job last week; what did we decide about the Henley branch roof?",
    "The user has invoked the harbour-check skill, and what did we decide about the Henley branch roof?",
    "[IMPORTANT: this is only my own bracketed note] what did we decide about the Henley branch roof?",
]


@pytest.mark.parametrize("shape", sorted(SKILL_TURNS))
def test_a_skill_or_cron_turn_is_recognised_by_its_shape(shape):
    assert pc.skill_turn_shape(SKILL_TURNS[shape]) is not None


@pytest.mark.parametrize("line", ORDINARY_LINES + ["", "yes", None, 7, ["x"]])
def test_an_ordinary_line_is_not_a_skill_turn(line):
    assert pc.skill_turn_shape(line) is None


@pytest.mark.parametrize("shape", sorted(SKILL_TURNS))
def test_a_skill_turn_makes_no_decision_no_search_no_writer_call_and_injects_nothing(shape):
    jev, expansion = FakeJev(decision=jev_decision("new_ask", 0.95, stands=0.99)), FakeExpansion()
    b = run(SKILL_TURNS[shape], make_deps(expansion, jev))
    assert b.text == "" and b.injected == 0 and b.searches == [] and b.cost_usd == 0
    assert b.reason == "skill turn" and b.decision == "zero"
    assert not jev.requests and not expansion.calls


def test_the_cron_wake_as_core_hands_it_over_is_cron_hint_plus_job_prompt():
    """Core's prefetch_all hands the plugin ``extract_user_instruction_from_skill_message(content)``: for a cron
    wake that is everything after the instruction marker, which begins with the cron hint (read on the live
    profile's state.db: the user row begins with the invoke marker, the stripped remainder with the hint)."""
    stripped = SKILL_TURNS["cron-skill-wake"].rsplit(INSTRUCTION_MARK, 1)[1]
    assert stripped.startswith(CRON_HINT) and pc.skill_turn_shape(stripped) == "cron"
    jev, expansion = FakeJev(decision=jev_decision("new_ask", 0.95)), FakeExpansion()
    assert run(stripped, make_deps(expansion, jev)).text == "" and not jev.requests and not expansion.calls


@pytest.mark.parametrize("line", ORDINARY_LINES)
def test_an_ordinary_message_about_skills_expands_as_usual(line):
    b, expansion, jev = ask(line)
    assert b.text and len(jev.decisions()) == 1 and len(expansion.calls) == 1 and b.reason == ""


@pytest.mark.parametrize("mode", ["contract", "off"])
@pytest.mark.parametrize("shape", sorted(SKILL_TURNS))
def test_the_provider_returns_nothing_for_a_skill_turn_in_either_mode(plugin_module, mode, shape, caplog):
    jev = FakeJev(decision=jev_decision("new_ask", 0.95, stands=0.99))
    config = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": mode}}}
    p = provider(plugin_module, config=config, jev=jev)
    with caplog.at_level(logging.DEBUG):
        assert p.prefetch(SKILL_TURNS[shape]) == ""
    assert not jev.requests and not p.expansion.calls
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    for word in ("harbour", "Henley", "substrate", "SILENT", "cron job"):       # no text of the turn is logged
        assert word not in caplog.text


def test_a_slash_skill_with_an_instruction_is_caught_through_on_turn_start(plugin_module):
    """Core strips the skill and hands prefetch only the instruction ('tidy the harbour notes' looks ordinary);
    on_turn_start gets the whole message first, so the provider knows the turn is a skill invocation."""
    jev = FakeJev(decision=jev_decision("new_ask", 0.95, stands=0.99))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev)
    p.on_turn_start(3, SKILL_TURNS["slash-skill"])
    assert p.prefetch("tidy the harbour notes") == "" and not jev.requests and not p.expansion.calls
    p.on_turn_start(4, ROOF_Q)                                                    # the next turn is ordinary
    assert "stored text of roof" in p.prefetch(ROOF_Q) and len(p.expansion.calls) == 1


def test_a_turn_without_on_turn_start_is_judged_by_its_own_text(plugin_module):
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev)
    assert p.prefetch(CRON_HINT + JOB_PROMPT) == "" and not jev.requests
    assert p.prefetch(ROOF_Q) != "" and len(p.expansion.calls) == 1


def test_on_turn_start_never_raises_and_ordinary_turns_are_unchanged(plugin_module):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95)))
    for bad in (None, 7, b"x", ["a"], {"k": 1}, object()):
        p.on_turn_start(1, bad)
    p.on_turn_start(2, ROOF_Q, remaining_tokens=10, model="m", platform="telegram")
    assert p.prefetch(ROOF_Q) != ""


def test_a_skill_turn_prepares_nothing_ahead_and_leaves_nothing_logged(plugin_module, monkeypatch, caplog):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1, need_library=1))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev)
    _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)                         # the agent ended on a proposal
    p.on_turn_start(5, SKILL_TURNS["cron-skill-wake"])
    with caplog.at_level(logging.DEBUG):
        p.queue_prefetch(CRON_HINT + JOB_PROMPT, session_id="s1")
        time.sleep(0.3)
    assert not p.expansion.calls and "s1" not in p._ahead and not jev.requests
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    p.on_turn_start(6, ROOF_Q)                                            # an ordinary turn prepares ahead again
    p.queue_prefetch(ROOF_Q, session_id="s1")
    assert _wait(lambda: "s1" in p._ahead)


def test_a_skill_turn_never_raises_even_when_the_library_is_down(plugin_module):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("new_ask", 0.9)))
    p._memory = None
    assert p.prefetch(SKILL_TURNS["cron-skill-wake"]) == ""
    p.on_turn_start(1, SKILL_TURNS["slash-skill"])
    assert p.prefetch("tidy the harbour notes") == ""
