"""The pre-turn recall contract (recall.prefetch.mode: contract). Offline: a fake Jev transport, a fake
question model, a fake store. Invented text only."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date
from types import SimpleNamespace

import pytest

import prefetch_contract as pc

OPENER_AGENT_LAST = ("I can draft the one-page explainer on the village library now. "
                     "Shall I write it in our explainer house style?")


# ---------------------------------------------------------------------------- fakes

TODAY = date(2026, 10, 17)


def jev_decision(kind, needs, *, stands=0.9, scope="none", **need):
    """A System One answer in the real shape; ``need`` names the need_* at or above 0.5; ``scope`` is
    the time_scope choice."""
    def noul(v):
        return {"type": "noul", "noul": v}
    answers = {
        "line_kind": {"type": "choice", "choice": kind, "probabilities": {kind: 0.9}, "confidence": 0.9},
        "stands_alone": noul(stands),
        "needs_memory": noul(needs),
        "time_scope": {"type": "choice", "choice": scope, "probabilities": {scope: 0.9}, "confidence": 0.9},
    }
    for k in pc.NEED_SOURCES:
        answers[k] = noul(0.9 if need.get(k) else 0.1)
    return {"model": "typesafe/jev-1.14-20261014", "answers": answers,
            "usage": {"input_tokens": 900, "output_tokens": 12, "cost": 0.00004}}


class FakeJev:
    """The decision and the scoring, told apart by the questions asked."""

    def __init__(self, decision=None, score=3.0, fail_decision=None, fail_rank=None, delay=0.0):
        self.decision, self.score = decision, score
        self.fail_decision, self.fail_rank = fail_decision, fail_rank
        self.delay = delay
        self.requests = []
        self.lock = threading.Lock()

    def decisions(self):
        return [r for r in self.requests if "line_kind" in r["questions"]]

    def rankings(self):
        """The writer's fallback asks Jev to rate sentences; nothing else is ever ranked by Jev."""
        return [r for r in self.requests if "p0" in r["questions"]]

    def __call__(self, request, timeout):
        with self.lock:
            self.requests.append(request)
        qs = request["questions"]
        if "line_kind" in qs:
            if self.delay:
                time.sleep(self.delay)
            if self.fail_decision:
                raise self.fail_decision
            return self.decision
        if self.fail_rank:
            raise self.fail_rank
        n = len(qs)
        scores = self.score(request) if callable(self.score) else [self.score] * n
        return {"answers": {f"p{i}": {"type": "score", "score": scores[i]} for i in range(n)},
                "usage": {"cost": 0.0001}}


class FakeWriter:
    """The small text model: returns a fixed reply (or raises, or is slow)."""

    def __init__(self, reply='{"questions": []}', fail=None, delay=0.0):
        self.reply, self.fail, self.delay = reply, fail, delay
        self.prompts = []

    def __call__(self, messages, *, json_mode=False):
        self.prompts.append(messages[0]["content"])
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return self.reply


def questions_reply(*qs):
    return json.dumps({"questions": list(qs)})


def item(name, text=None, *, date="2026-03-02", source=None, doc=None, kind="note", match="semantic",
         snippet=None, score=0.8):
    """A hit as the library's ``Memory.search`` returns it, with the cited-evidence fields."""
    return SimpleNamespace(
        content=f"stored text of {name}", original_chunk=text or f"stored text of {name}",
        source=source or f"note:{name}", score=0.5, scores={"rrf": 0.5}, metadata={"created_on": date},
        bank_id="b", document_id=doc or f"doc-{name}", created_on=date, kind=kind, match_kind=match,
        exact_snippet=snippet, relevance_score=score)


class FakeStore:
    """The library's search: one callable as the contract's ``Deps.search`` (question, scope) -> Found,
    and ``.search`` as ``Memory.search`` (what the provider calls), which also fills the trace the way
    the library does: a ``rerank`` record without a fallback reason when Stage 2 graded the hits."""

    def __init__(self, by_question=None, default=None, fail=None, delay=0.0, ranked=True):
        self.by_question, self.default, self.fail, self.delay = by_question or {}, default or [], fail, delay
        self.ranked = ranked
        self.calls, self.scopes, self.kwargs = [], [], []

    def _items(self, q):
        self.calls.append(q)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        return list(self.by_question.get(q, self.default))

    def __call__(self, q, scope=None):
        self.scopes.append(scope)
        return pc.Found(items=self._items(q), ranked=self.ranked)

    def search(self, q, **kw):
        self.kwargs.append(kw)
        items = self._items(q)
        trace = kw.get("_trace")
        if trace is not None:
            trace.append({"rerank": {"fallback_reason": None} if self.ranked
                          else {"fallback_reason": "fused order stands: RuntimeError: 503"}})
        return items


def make_deps(store, jev=None, writer=None):
    return pc.Deps(search=store, transport=jev, writer_llm=writer, today=lambda: TODAY)


SETTINGS = pc.Settings(mode="contract", decision_timeout_s=0.5, writer_timeout_s=0.5, search_timeout_s=1.0,
                       deadline_s=3.0)


def run(line, deps, *, agent_last=OPENER_AGENT_LAST, settings=SETTINGS, **kw):
    return pc.run_contract(line, agent_last=agent_last, turns=4, deps=deps, settings=settings,
                           contract=pc.load_contract(), **kw)


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
    assert c.questions["stands_alone"]["criteria"]["false"] == (
        "It leans on `agent_last` or the thread (yes, it, that, the second one).")
    assert c.questions["needs_memory"]["instructions"].startswith(
        "To do what `new_lines` asks (or what it agrees to), does the agent need something that is NOT in")
    assert c.questions["need_library"]["instructions"].endswith(": what a document in the Library says?")
    assert c.questions["need_work"]["criteria"] == {"true": "Yes.", "false": "No."}
    assert c.writer_prompt.startswith("Write the questions a memory search is asked before an agent's turn.")
    assert c.writer_prompt.endswith('Return `{"questions": ["…"]}`.')
    assert "never \"it\", \"this\", \"yes\" or \"the second one\"" in c.writer_prompt


def test_decision_request_has_the_designs_shape():
    jev = FakeJev(decision=jev_decision("new_ask", 0.1))
    run("What did we decide about the Henley branch?", make_deps(FakeStore(), jev, FakeWriter()))
    (req,) = jev.decisions()
    assert req["model"] == "typesafe/jev-1.14"
    assert list(req["state"]) == ["agent", "turn", "thread", "agent_last", "new_lines", "in_view", "today"]
    assert req["state"]["today"] == "2026-10-17"
    assert req["state"]["agent_last"] == OPENER_AGENT_LAST
    assert req["state"]["new_lines"] == ["The person (to you): What did we decide about the Henley branch?"]
    assert list(req["questions"]) == list(pc.load_contract().questions)


def test_swapping_the_contract_file_is_enough(plugin_module, tmp_path):
    text = pc.DEFAULT_CONTRACT_PATH.read_text()
    swapped = text.replace("Write the questions a memory search is asked before an agent's turn.",
                           "WRITE THE QUESTIONS (swapped file).").replace("Background from memory,", "Swapped intro,")
    path = tmp_path / "other.md"
    path.write_text(swapped)
    contract = pc.load_contract(path)
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.1, need_decisions=1))
    writer = FakeWriter(questions_reply("What happened to the Henley branch?"))
    store = FakeStore(default=[item("henley")])
    b = pc.run_contract("and that branch?", agent_last="", turns=1, deps=make_deps(store, jev, writer),
                        settings=SETTINGS, contract=contract)
    assert "WRITE THE QUESTIONS (swapped file)." in writer.prompts[0]
    assert b.text.startswith("Swapped intro,")


def test_bad_contract_files_are_refused(tmp_path):
    with pytest.raises(pc.ContractError):
        pc.load_contract(tmp_path / "missing.md")
    bad = tmp_path / "bad.md"
    bad.write_text("## jev-questions\n\n```json\n{\"line_kind\": {}}\n```\n")
    with pytest.raises(pc.ContractError):
        pc.load_contract(bad)


# ---------------------------------------------------------------------------- gate

@pytest.mark.parametrize("text", ["", "   \n ", "<memory-context>\n</memory-context>", "/new", "  /status now",
                                  "yes", "OK!", "thanks :)", "do it", None])
def test_gate_skips_without_any_call(text):
    jev, writer, store = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeWriter(), FakeStore()
    b = run(text, make_deps(store, jev, writer))
    assert b.text == "" and b.decision == "zero"
    assert not jev.requests and not writer.prompts and not store.calls


# ---------------------------------------------------------------------------- the design's six examples

def test_panama_deal_is_a_question_for_memory():
    line = "yeah, but no like that Panama deal we closed last year"
    jev = FakeJev(decision=jev_decision("follow_up", 0.92, stands=0.2, need_decisions=1))
    writer = FakeWriter(questions_reply("What was the Panama deal we closed last year?"))
    store = FakeStore(default=[item("panama", "The Panama deal closed in March 2025 for the harbour lease.")])
    b = run(line, make_deps(store, jev, writer))
    assert b.decision == "ask" and b.questions == ["What was the Panama deal we closed last year?"]
    assert store.calls == ["What was the Panama deal we closed last year?"]  # the line itself is never asked
    assert "The Panama deal closed in March 2025" in b.text
    assert b.sources == ["decisions"]
    assert OPENER_AGENT_LAST in writer.prompts[0] and line in writer.prompts[0]


def test_fine_with_me_asks_nothing():
    jev = FakeJev(decision=jev_decision("answer", 0.04))
    writer, store = FakeWriter(), FakeStore()
    b = run("fine with me", make_deps(store, jev, writer), agent_last="I'll use the shorter title. OK?")
    assert b.text == "" and b.decision == "zero" and b.reason == "no need"
    assert not writer.prompts and not store.calls and not jev.rankings()


def test_lets_keep_going_with_the_draft_in_the_chat_asks_nothing():
    jev = FakeJev(decision=jev_decision("go_ahead", 0.1, stands=0.3))
    writer, store = FakeWriter(), FakeStore()
    b = run("let's keep going, but this time add more color to the motivation", make_deps(store, jev, writer))
    assert b.text == "" and not writer.prompts and not store.calls


def test_yes_to_a_proposal_is_filtered_and_ok_do_it_decides_zero():
    jev = FakeJev(decision=jev_decision("go_ahead", 0.08, stands=0.1))
    writer, store = FakeWriter(), FakeStore()
    deps = make_deps(store, jev, writer)
    assert run("yes", deps).text == "" and not jev.requests          # core's filter, held here too
    b = run("ok do it", deps)                                         # not a trivial word: Jev decides
    assert len(jev.decisions()) == 1 and b.text == "" and b.line_kind == "go_ahead"
    assert not writer.prompts and not store.calls


def test_go_ahead_that_needs_memory_is_asked_from_the_agents_last_line():
    jev = FakeJev(decision=jev_decision("go_ahead", 0.85, stands=0.1, need_library=1))
    writer = FakeWriter(questions_reply("What happened to the village library, and when did it close?",
                                        "What is our house style for explainers?"))
    store = FakeStore(default=[item("closure", "The library closed on 4 June after the roof failed.")])
    b = run("ok do it", make_deps(store, jev, writer))
    assert b.decision == "ask" and len(b.questions) == 2
    assert sorted(store.calls) == sorted(b.questions)         # both asked
    assert OPENER_AGENT_LAST in writer.prompts[0]              # the questions come from the agent's own line
    assert b.sources == ["library"]


def test_pasted_file_is_read_whole_by_the_writer_and_never_asked():
    paste = "Minutes of the branch review. " * 2500            # 75,000 characters
    line = paste + "\nCompare this with what we decided in March."
    jev = FakeJev(decision=jev_decision("material", 0.9))
    writer = FakeWriter(questions_reply("What did we decide in March about the branch review?"))
    store = FakeStore(default=[item("march", "March: keep the Henley branch open until the roof is fixed.")])
    b = run(line, make_deps(store, jev, writer))
    assert not jev.decisions()                                  # too long for Jev: marked as material by size
    assert line in writer.prompts[0]                            # whole, never sliced
    assert store.calls == ["What did we decide in March about the branch review?"]
    assert b.line_kind == "material" and "March: keep the Henley branch open" in b.text


def test_pasted_file_with_no_working_writer_asks_nothing_and_says_why():
    line = "Minutes of the branch review. " * 2500 + "Compare with March."
    store = FakeStore(default=[item("x")])
    b = run(line, make_deps(store, FakeJev(), FakeWriter(fail=RuntimeError("no key"))), agent_last="")
    assert not store.calls                                      # a pasted text is never asked as it is
    assert b.text == "(memory was not consulted: no question could be written from this message)"


def test_a_smaller_paste_goes_to_jev_as_material_then_to_the_writer():
    line = "Notes from the call: the roof, the budget, the opening hours.\n" * 40 + "What did we decide in March?"
    jev = FakeJev(decision=jev_decision("material", 0.9, stands=0.4, need_decisions=1))
    writer = FakeWriter(questions_reply("What did we decide in March about the Henley branch?"))
    store = FakeStore(default=[item("march")])
    b = run(line, make_deps(store, jev, writer))
    assert len(jev.decisions()) == 1 and writer.prompts and b.questions == [
        "What did we decide in March about the Henley branch?"]
    assert line not in store.calls


def test_a_plain_standalone_question_is_asked_as_it_is_with_no_writer():
    line = "What happened to the village library?"
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95, need_library=1))
    writer, store = FakeWriter(), FakeStore(default=[item("closure")])
    b = run(line, make_deps(store, jev, writer), agent_last="")
    assert store.calls == [line] and not writer.prompts and b.questions == [line]


def test_commission_style_line_gets_one_question_without_the_answer_form():
    line = ("Quick question for the recall acceptance: what did we decide about the explainer's house "
            "style? One sentence.")
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.8, need_decisions=1))
    writer = FakeWriter(questions_reply("What did we decide about the explainer house style?"))
    store = FakeStore(default=[item("style", "Explainer house style: one idea per page, plain words.")])
    b = run(line, make_deps(store, jev, writer))
    assert b.questions == ["What did we decide about the explainer house style?"]
    assert store.calls == b.questions and "One sentence" not in "".join(store.calls)


# ---------------------------------------------------------------------------- reading the answers

def test_social_is_always_zero_and_needs_memory_below_half_is_zero():
    for kind, needs in (("social", 0.95), ("new_ask", 0.49), ("follow_up", 0.0)):
        jev = FakeJev(decision=jev_decision(kind, needs, need_decisions=1))
        writer, store = FakeWriter(), FakeStore()
        b = run("Thanks, that was helpful to know about it", make_deps(store, jev, writer))
        assert b.text == "" and b.decision == "zero" and not store.calls and not writer.prompts


@pytest.mark.parametrize("need,keep,drop", [
    ({"need_conversations": 1}, "turn:s1:1", "note:a"),
    ({"need_decisions": 1}, "note:a", "turn:s1:1"),
    ({}, "note:a", None),                                      # none at 0.5: every source
    ({"need_decisions": 1, "need_conversations": 1}, "note:a", None),
])
def test_need_answers_become_the_sources_asked(need, keep, drop):
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.9, **need))
    store = FakeStore(default=[item("a", source="note:a"), item("t", source="turn:s1:1")])
    b = run("What did we decide about the Henley branch last spring?", make_deps(store, jev, FakeWriter()))
    assert f"source={keep} |" in b.text
    if drop:
        assert f"source={drop} |" not in b.text
    else:
        assert "source=turn:s1:1 |" in b.text and "source=note:a |" in b.text


# ---------------------------------------------------------------------------- fallbacks

def test_jev_refusal_decides_by_rule_and_the_block_says_so():
    err = RuntimeError("HTTP 400")
    store = FakeStore(default=[item("a", "Henley branch: opening hours are 9 to 5.")])
    deps = make_deps(store, FakeJev(fail_decision=err), FakeWriter())
    b = run("What are the Henley branch opening hours these days?", deps)
    assert b.decision == "fallback" and store.calls == ["What are the Henley branch opening hours these days?"]
    assert "Questions formed without the decision model." in b.text
    short = run("sounds good", deps)
    assert short.text == "" and short.decision == "fallback"      # three words or fewer, or a social line: zero


def test_slow_jev_falls_back_inside_its_budget():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9), delay=2.0)
    store = FakeStore(default=[item("a")])
    t0 = time.monotonic()
    b = run("What did we decide about the Henley branch roof?", make_deps(store, jev, FakeWriter()))
    assert time.monotonic() - t0 < 1.5 and b.decision == "fallback" and b.questions


def test_jev_answering_only_some_questions_is_no_decision():
    partial = jev_decision("new_ask", 0.9)
    del partial["answers"]["need_work"]
    b = run("What did we decide about the Henley branch roof?",
            make_deps(FakeStore(default=[item("a")]), FakeJev(decision=partial), FakeWriter()))
    assert b.decision == "fallback"


def test_jev_without_a_transport_or_key_still_gives_an_answer():
    b = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(default=[item("a")])))
    assert b.decision == "fallback" and b.questions and b.text


@pytest.mark.parametrize("writer", [
    FakeWriter("not json at all"),
    FakeWriter('{"queries": "nope"}'),
    FakeWriter(fail=RuntimeError("provider down")),
    FakeWriter(questions_reply("What is our house style?"), delay=2.0),
])
def test_writer_failure_asks_the_sentence_jev_rates_most_relevant(writer):
    def pick(request):  # the "house style" sentence is the best one
        return [3.0 if "house style" in r["instructions"]["text"] else 0.0 for r in request["questions"].values()]
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1), score=pick)
    store = FakeStore(default=[item("a")])
    t0 = time.monotonic()
    b = run("ok do it", make_deps(store, jev, writer))
    assert time.monotonic() - t0 < 1.5
    assert b.questions == ["Shall I write it in our explainer house style?"]
    assert store.calls == b.questions


def test_writer_failure_with_a_standalone_line_asks_the_line_whole():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.9))
    store = FakeStore(default=[item("a")])
    line = "Before we start: what did we decide about the Henley roof? And who paid for it."
    b = run(line, make_deps(store, jev, FakeWriter("garbage")))
    assert b.questions == [line] and store.calls == [line]


def test_writer_failure_and_a_bare_reply_asks_nothing_and_says_so():
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1), fail_rank=RuntimeError("x"))
    store = FakeStore()
    b = run("ok go", make_deps(store, jev, FakeWriter(fail=RuntimeError("down"))), agent_last="")
    assert not store.calls
    assert b.text == "(memory was not consulted: no question could be written from this message)"


def test_writer_with_zero_questions_is_zero_and_logs_nothing(caplog):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    store = FakeStore()
    with caplog.at_level(logging.DEBUG):
        b = run("and the second one?", make_deps(store, jev, FakeWriter(questions_reply())))
    assert b.text == "" and b.decision == "zero" and not store.calls
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_writer_gets_at_most_three_questions():
    qs = [f"What about thing number {i} on the roof?" for i in range(5)]
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.1))
    store = FakeStore(default=[item("a")])
    b = run("tell me about all of it", make_deps(store, jev, FakeWriter(questions_reply(*qs))))
    assert b.questions == qs[:3]


def test_search_error_says_memory_was_not_consulted():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    b = run("What did we decide about the Henley branch roof?",
            make_deps(FakeStore(fail=RuntimeError("db down")), jev, FakeWriter()))
    assert b.text == "(memory was not consulted: the search failed)" and b.injected == 0


def test_search_that_does_not_answer_in_time_is_dropped():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    s = pc.Settings(mode="contract", search_timeout_s=0.2, deadline_s=3.0)
    t0 = time.monotonic()
    b = run("What did we decide about the Henley branch roof?",
            make_deps(FakeStore(delay=2.0), jev, FakeWriter()), settings=s)
    assert time.monotonic() - t0 < 1.0
    assert b.text == "(memory was not consulted: the search did not answer in time)"


def test_one_failed_question_does_not_drop_the_others():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Question one about the roof?", "Question two about the budget?"))

    def search(q, scope):
        if "one" in q:
            raise RuntimeError("boom")
        return pc.Found([item("budget", "The budget was 40 thousand.")])
    b = run("and that?", pc.Deps(search=search, transport=jev, writer_llm=writer))
    assert "The budget was 40 thousand." in b.text


def test_an_unranked_search_keeps_the_first_page_per_question_and_says_so():
    """Stage 2 failed in the library (it fell back to Stage 1's order): five of each question, a note."""
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Question A about the roof?", "Question B about the budget?"))
    store = FakeStore(by_question={
        "Question A about the roof?": [item(f"a{i}") for i in range(8)],
        "Question B about the budget?": [item(f"b{i}") for i in range(8)]}, ranked=False)
    b = run("and that?", make_deps(store, jev, writer))
    assert "The reranker could not rank these; they are in the search's own order." in b.text
    assert b.injected == 10 and "stored text of a4" in b.text and "stored text of a5" not in b.text
    assert b.ranked is False


def test_a_ranked_search_is_shown_as_the_library_cut_it_without_a_note():
    items = [item(f"n{i}") for i in range(14)]
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    b = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(default=items), jev, FakeWriter()))
    assert b.injected == 14 and b.ranked is True and "own order" not in b.text   # no first-page cap on a graded list


def test_jev_is_never_asked_to_rank_the_results():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    run("What did we decide about the Henley branch roof?",
        make_deps(FakeStore(default=[item("a"), item("b")]), jev, FakeWriter()))
    assert len(jev.requests) == 1 and not jev.rankings()      # the decision only: Stage 2 is the library's


# ---------------------------------------------------------------------------- zero costs nothing

def test_zero_path_makes_one_decision_call_and_nothing_else():
    jev = FakeJev(decision=jev_decision("answer", 0.02))
    writer, store = FakeWriter(), FakeStore()
    b = run("no, the second option please", make_deps(store, jev, writer))
    assert len(jev.requests) == 1 and not writer.prompts and not store.calls
    assert b.cost_usd == pytest.approx(0.00004)


# ---------------------------------------------------------------------------- the block

def test_block_is_the_librarys_cited_evidence():
    store = FakeStore(default=[item("a", date="2026-03-02", match="exact+semantic", snippet="the <<roof>> failed"),
                               item("b", date="2026-04-01", kind="conversation", match="lexical"),
                               item("c", date="2026-05-09", match="semantic")])
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    b = run("What did we decide about the Henley branch roof?", make_deps(store, jev, FakeWriter()))
    lines = b.text.split("\n")
    assert lines[0].startswith("Background from memory, gathered for this turn.")
    assert "Asked: What did we decide about the Henley branch roof?" in b.text
    assert ('[1] source=note:a | date=2026-03-02 | kind=note | match=exact+semantic\n'
            '    Exact match snippet: "the <<roof>> failed"\n    stored text of a') in b.text
    assert "[2] source=note:b | date=2026-04-01 | kind=conversation | match=lexical\n    stored text of b" in b.text
    assert "[3] source=note:c | date=2026-05-09 | kind=note | match=semantic\n    stored text of c" in b.text
    assert b.text.count("Exact match snippet:") == 1          # only where a lexical retriever fired
    assert "rated:" not in b.text and "more matched" not in b.text and b.injected == 3
    # the entries are exactly what the library's own formatter writes
    from prospecta.cited import format_cited_rag_block
    assert b.text.endswith(format_cited_rag_block(store.default, heading=None))


def test_no_count_cap_and_whole_entries_never_cut():
    long_text = "A long note about the roof. " * 2000                    # 56,000 characters
    items = [item(f"n{i}", f"note {i}: " + (long_text if i == 3 else "short text")) for i in range(14)]
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    b = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(default=items), jev, FakeWriter()))
    assert b.injected == 14 and "more matched" not in b.text
    assert f"note 3: {long_text}" in b.text                                 # the long entry, whole


def test_parallel_questions_are_each_asked_once_and_merged_one_note_once():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Question A about the roof?", "Question B about the budget?",
                                        "Question C about the opening hours?"))
    gate_ = threading.Barrier(3, timeout=2)
    seen = []

    def search(q, scope):
        gate_.wait()                      # all three must be in flight at once, or this times out
        seen.append(q)
        return pc.Found([item("shared", "one note both found"), item(q[:10], f"text for {q}")])
    b = run("and that?", pc.Deps(search=search, transport=jev, writer_llm=writer))
    assert len(seen) == 3 and b.injected == 4 and b.text.count("one note both found") == 1
    assert len(jev.requests) == 1          # the decision only


def test_a_note_two_questions_found_keeps_its_place_and_the_matched_words():
    plain = item("party", "We watched Electric Dreams at the party.", match="semantic")
    snip = item("party", "We watched Electric Dreams at the party.", match="exact+semantic",
                snippet="watched <<Electric>> <<Dreams>> at")
    other = item("other", "something else")
    merged = pc.merge_lists([[plain, other], [snip]])
    assert merged == [snip, other]                                  # one note once, first place, with its snippet
    assert pc.merge_lists([[snip], [plain]]) == [snip]              # the one with words in context stays either way
    assert pc.merge_lists([[plain], [item("party", match="lexical")]]) == [plain]   # no snippet either: the first stays


def test_nothing_answers_is_one_line():
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    empty = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(), jev, FakeWriter()))
    assert empty.text == "Memory was asked: What did we decide about the Henley branch roof?. Nothing in it answers this."
    assert empty.injected == 0


def test_max_tokens_valve_skips_whole_entries_and_is_off_by_default():
    items = [item("a", "x" * 400), item("b", "y" * 4000), item("c", "z" * 400)]
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    off = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(default=items), jev, FakeWriter()))
    assert off.injected == 3
    on = run("What did we decide about the Henley branch roof?", make_deps(FakeStore(default=items), jev, FakeWriter()),
             settings=pc.Settings(mode="contract", max_tokens=450))
    assert on.injected == 2 and "y" * 4000 not in on.text and "x" * 400 in on.text and "z" * 400 in on.text
    assert "1 more matched less well." in on.text
    assert not any(len(set(p.strip())) == 1 and 0 < len(p.strip()) < 400 for p in on.text.split("\n"))   # nothing cut mid-text


# ---------------------------------------------------------------------------- provider: switch, never fails, deadline

class FakeMemory:
    def __init__(self, store, synth=None):
        self.store, self.synth_calls = store, []
        self._synth = synth

    def search(self, q, **kw):
        return self.store.search(q, **kw)

    def recall_synth(self, q, **kw):
        self.synth_calls.append(q)
        return SimpleNamespace(synthesis="An old synthesised answer.", sources=[SimpleNamespace(source="s1")])

    def _tracer(self, *a, **k):
        pass


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_PREFETCH_DECISION_MODEL",
              "PROSPECTA_PREFETCH_QUESTION_MODEL", "PROSPECTA_PREFETCH_CONTRACT", "PROSPECTA_PREFETCH_MAX_TOKENS",
              "PROSPECTA_PREFETCH_DEADLINE", "PROSPECTA_DECISION_MODEL", "PROSPECTA_QUERY_WRITER_MODEL",
              "PROSPECTA_RERANKER_MODEL", "PROSPECTA_RECALL_STAGE3", "PROSPECTA_TIMEZONE"):
        monkeypatch.delenv(k, raising=False)


def provider(plugin_module, *, config=None, store=None, jev=None, writer=None):
    p = plugin_module.ProspectaProvider()
    p._config = dict(config or {})
    p._session_id = "s1"
    p._memory = FakeMemory(store or FakeStore())
    p._jev_transport = jev
    p._writer_llm = writer
    p._prefetch_cfg = p._resolve_prefetch_settings()
    return p


CONTRACT_ON = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract"}}}


def test_defaults_are_off_and_flat_nested_and_env_settings_read(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    s = p._resolve_prefetch_settings()
    assert (s.mode, s.decision_model, s.query_writer_model, s.max_tokens, s.deadline_s) == (
        "off", "typesafe/jev-1.14", "openrouter/google/gemini-2.5-flash-lite", 0, 5.0)
    assert p._reranker_model() == "qwen/qwen3-reranker-8b"
    p._config = {"recall.prefetch.mode": "contract", "recall.prefetch.max_tokens": "800",
                 "recall.prefetch.contract_path": "/x/c.md"}
    s = p._resolve_prefetch_settings()
    assert (s.mode, s.max_tokens, s.contract_path) == ("contract", 800, "/x/c.md")
    p._config = {"recall": {"prefetch": {"mode": "contract"}}}
    assert p._resolve_prefetch_settings().mode == "contract"
    monkeypatch.setenv("PROSPECTA_PREFETCH_MODE", "off")
    assert p._resolve_prefetch_settings().mode == "off"


def test_the_three_models_are_configurable_profile_then_env_then_default(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    assert (p._decision_model(), p._reranker_model()) == ("typesafe/jev-1.14", "qwen/qwen3-reranker-8b")
    monkeypatch.setenv("PROSPECTA_DECISION_MODEL", "typesafe/jev-env")
    monkeypatch.setenv("PROSPECTA_QUERY_WRITER_MODEL", "openrouter/x/writer-env")
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "x/reranker-env")
    s = p._resolve_prefetch_settings()
    assert (s.decision_model, s.query_writer_model, p._reranker_model()) == (
        "typesafe/jev-env", "openrouter/x/writer-env", "x/reranker-env")
    p._config = {"decision_model": "typesafe/jev-9", "query_writer_model": "openrouter/x/writer-cfg",
                 "reranker_model": "x/reranker-cfg"}                        # the profile wins over the env
    s = p._resolve_prefetch_settings()
    assert (s.decision_model, s.query_writer_model, p._reranker_model()) == (
        "typesafe/jev-9", "openrouter/x/writer-cfg", "x/reranker-cfg")
    p._config = {"decision_model": "  ", "reranker_model": 5}               # blank or not a string: not set
    assert (p._decision_model(), p._reranker_model()) == ("typesafe/jev-env", "x/reranker-env")


def test_the_old_model_settings_are_ignored_loudly(plugin_module, monkeypatch, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall.prefetch.decision_model": "typesafe/jev-old",
                 "recall": {"prefetch": {"question_model": "openrouter/x/old"}}}
    monkeypatch.setenv("PROSPECTA_PREFETCH_QUESTION_MODEL", "openrouter/x/old-env")
    with caplog.at_level(logging.WARNING):
        p._warn_renamed_settings()
    assert p._resolve_prefetch_settings().decision_model == "typesafe/jev-1.14"        # not applied
    for old, new in (("recall.prefetch.decision_model", "decision_model"),
                     ("recall.prefetch.question_model", "query_writer_model"),
                     ("PROSPECTA_PREFETCH_QUESTION_MODEL", "PROSPECTA_QUERY_WRITER_MODEL")):
        assert f"{old} is ignored now; set {new}" in caplog.text
    caplog.clear()
    monkeypatch.delenv("PROSPECTA_PREFETCH_QUESTION_MODEL")
    plugin_module.ProspectaProvider()._warn_renamed_settings()                         # nothing set: silent
    assert not caplog.records


def test_bad_mode_is_off_with_a_warning(plugin_module, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {"mode": "full", "max_tokens": "lots"}}}
    with caplog.at_level(logging.WARNING):
        s = p._resolve_prefetch_settings()
    assert s.mode == "off" and s.max_tokens == 0 and "full" in caplog.text and "lots" in caplog.text


def test_mode_is_read_at_initialize_so_the_next_session_picks_it_up(plugin_module):
    p = plugin_module.ProspectaProvider()
    p._config = {"prefetch_enabled": "true"}
    assert p._resolve_prefetch_settings().mode == "off"
    p._config = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract"}}}
    assert p._resolve_prefetch_settings().mode == "contract"   # what initialize() stores per session


def test_prefetch_disabled_changes_nothing_even_with_mode_contract(plugin_module):
    jev, store = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeStore(default=[item("a")])
    p = provider(plugin_module, config={"recall": {"prefetch": {"mode": "contract"}}}, store=store, jev=jev,
                 writer=FakeWriter())
    assert p.prefetch("What did we decide about the Henley branch roof?") == ""
    assert not jev.requests and not store.calls and p._memory.synth_calls == []
    p._config = {"prefetch_enabled": "false", "recall": {"prefetch": {"mode": "contract"}}}
    assert p.prefetch("What did we decide about the Henley branch roof?") == ""
    assert not jev.requests


def test_mode_off_keeps_the_old_synthesised_prefetch(plugin_module):
    jev, store = FakeJev(decision=jev_decision("new_ask", 0.9)), FakeStore(default=[item("a")])
    p = provider(plugin_module, config={"prefetch_enabled": "true"}, store=store, jev=jev, writer=FakeWriter())
    out = p.prefetch("What did we decide?")
    assert "An old synthesised answer." in out and out.startswith("<prospecta-recall>")
    assert p._memory.synth_calls == ["What did we decide?"] and not jev.requests and not store.calls
    p.queue_prefetch("anything")                      # no ahead work either
    time.sleep(0.1)
    assert not jev.requests


def test_mode_contract_never_synthesises(plugin_module):
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    store = FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev, writer=FakeWriter())
    out = p.prefetch("What did we decide about the Henley branch roof?")
    assert "The roof is to be replaced in June." in out
    assert p._memory.synth_calls == [] and p._last_brief.decision == "ask"
    assert "<prospecta-recall>" not in out


def test_a_successful_turn_writes_no_recall_log_line(plugin_module, caplog):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    store = FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev,
                 writer=FakeWriter(questions_reply("What is the plan for the roof?")))
    with caplog.at_level(logging.DEBUG):
        assert "The roof is to be replaced" in p.prefetch("and the roof?")
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
    assert not [r for r in caplog.records if "[recall]" in r.getMessage() or "brief" in r.getMessage()]


def test_prefetch_never_raises_whatever_fails(plugin_module, monkeypatch, caplog):
    p = provider(plugin_module, config=CONTRACT_ON, jev=FakeJev(decision=jev_decision("new_ask", 0.9)),
                 writer=FakeWriter())

    class Weird(Exception):
        pass

    for err in (RuntimeError("secret words: the Panama deal"), Weird("x"), KeyError("k"), ValueError(), OSError(5, "io")):
        def boom(*a, **k):
            raise err
        monkeypatch.setattr(plugin_module._pc, "run_contract", boom)   # the plugin's own copy of the module
        caplog.clear()
        with caplog.at_level(logging.INFO):
            assert p.prefetch("What did we decide about the Henley branch roof?") == ""
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "Panama" not in caplog.text and "Henley" not in caplog.text    # no text in the log
    monkeypatch.undo()
    p._memory = None
    assert p.prefetch("anything at all here") == ""
    p2 = provider(plugin_module, config={**CONTRACT_ON, "recall": {"prefetch": {"mode": "contract",
                                                                              "contract_path": "/no/such.md"}}})
    p2._prefetch_cfg = p2._resolve_prefetch_settings()
    assert p2.prefetch("What did we decide about the Henley branch roof?") == ""


def test_every_stage_failing_still_returns_a_string(plugin_module):
    p = provider(plugin_module, config=CONTRACT_ON, store=FakeStore(fail=RuntimeError("db")),
                 jev=FakeJev(fail_decision=RuntimeError("down")), writer=FakeWriter(fail=RuntimeError("down")))
    out = p.prefetch("What did we decide about the Henley branch roof?")
    assert isinstance(out, str) and out == "(memory was not consulted: the search failed)"


def test_the_deadline_is_honoured_and_a_late_answer_is_dropped(plugin_module, caplog):
    store = FakeStore(default=[item("roof", "late note")], delay=1.5)
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev, writer=FakeWriter())
    p._prefetch_cfg = pc.Settings(mode="contract", deadline_s=0.4, search_timeout_s=5.0)
    t0 = time.monotonic()
    with caplog.at_level(logging.INFO):
        out = p.prefetch("What did we decide about the Henley branch roof?")
    assert out == "(memory was not consulted: the search did not answer in time)"
    assert time.monotonic() - t0 < 1.0
    assert "Henley" not in caplog.text and "late note" not in out
    time.sleep(1.4)                                  # the late worker finishes: nothing is injected afterwards
    assert p._last_brief.text == out and "late note" not in p._last_brief.text


def test_the_default_deadline_is_below_cores_wait(plugin_module):
    s = plugin_module.ProspectaProvider()._resolve_prefetch_settings()
    assert s.deadline_s == 5.0 < 8.0                 # agent/memory_manager.py _EXTERNAL_PREFETCH_TIMEOUT_S
    p = plugin_module.ProspectaProvider()
    p._config = {"recall.prefetch.deadline_seconds": "30"}
    assert p._resolve_prefetch_settings().deadline_s < 8.0


# ---------------------------------------------------------------------------- the agent's last answer and ahead

def _sync(p, user, assistant, monkeypatch, sid="s1"):
    p._retained = getattr(p, "_retained", [])
    monkeypatch.setattr(p, "_retain", lambda content, **k: p._retained.append(k) or "doc")
    p.sync_turn(user, assistant, session_id=sid)
    p._sync_thread.join(2)
    time.sleep(0.003)             # a turn's source ends in the millisecond: keep two turns apart


def test_sync_turn_keeps_the_agents_last_answer_per_session(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.1))
    p = provider(plugin_module, config=CONTRACT_ON, jev=jev, writer=FakeWriter())
    _sync(p, "hello there", OPENER_AGENT_LAST, monkeypatch)
    _sync(p, "other", "A different answer for another session.", monkeypatch, sid="s2")
    p.prefetch("ok do it please")
    assert jev.decisions()[0]["state"]["agent_last"] == OPENER_AGENT_LAST
    assert "4 earlier" not in jev.decisions()[0]["state"]["in_view"] and "1 earlier" in jev.decisions()[0]["state"]["in_view"]
    p.prefetch("and then", session_id="s2")
    assert jev.decisions()[1]["state"]["agent_last"] == "A different answer for another session."


def _wait(cond, secs=3.0):
    end = time.monotonic() + secs
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_ahead_is_keyed_on_the_agents_answer_and_spares_the_next_turn(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1, need_library=1))
    writer = FakeWriter(questions_reply("What happened to the village library?"))
    store = FakeStore(default=[item("closure", "The library closed on 4 June after the roof failed.")])
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev, writer=writer)
    _sync(p, "can you help with an explainer?", OPENER_AGENT_LAST, monkeypatch)
    p.queue_prefetch("can you help with an explainer?", session_id="s1")           # the person's words
    assert _wait(lambda: "s1" in p._ahead)
    assert OPENER_AGENT_LAST in writer.prompts[0]                                  # keyed on the agent's answer
    assert "can you help with an explainer?" not in writer.prompts[0]
    assert not jev.decisions()                                                     # nobody replied: no decision
    assert p._ahead["s1"].key == pc.ahead_key(OPENER_AGENT_LAST)
    n_writer, n_search = len(writer.prompts), len(store.calls)
    out = p.prefetch("ok do it")                                                   # the go-ahead
    assert "The library closed on 4 June" in out
    assert len(writer.prompts) == n_writer and len(store.calls) == n_search        # used, not recomputed
    assert p._last_brief.ahead_used and "s1" not in p._ahead                       # one use


def test_ahead_is_not_used_when_the_agent_said_something_else(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("What happened to the village library?"))
    store = FakeStore(default=[item("closure", "closure text")])
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev, writer=writer)
    _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)
    p.queue_prefetch("x", session_id="s1")
    assert _wait(lambda: "s1" in p._ahead)
    _sync(p, "ok do it", "Done. The draft is in the chat.", monkeypatch)           # a new answer: the old key is stale
    n = len(writer.prompts)
    p.prefetch("ok do it")
    assert not p._last_brief.ahead_used and len(writer.prompts) == n + 1


def test_no_ahead_unless_the_agent_ended_on_a_proposal(plugin_module, monkeypatch):
    jev, writer = FakeJev(decision=jev_decision("go_ahead", 0.9)), FakeWriter(questions_reply("A question?"))
    p = provider(plugin_module, config=CONTRACT_ON, store=FakeStore(default=[item("a")]), jev=jev, writer=writer)
    _sync(p, "x", "The draft is finished and saved in the chat.", monkeypatch)
    p.queue_prefetch("x", session_id="s1")
    time.sleep(0.2)
    assert not writer.prompts and "s1" not in p._ahead


def test_no_ahead_with_the_switch_off(plugin_module, monkeypatch):
    for config in ({"prefetch_enabled": "true"}, {"recall": {"prefetch": {"mode": "contract"}}}):
        writer = FakeWriter(questions_reply("A question?"))
        p = provider(plugin_module, config=config, store=FakeStore(default=[item("a")]),
                     jev=FakeJev(decision=jev_decision("go_ahead", 0.9)), writer=writer)
        _sync(p, "x", OPENER_AGENT_LAST, monkeypatch)
        p.queue_prefetch("x", session_id="s1")
        time.sleep(0.2)
        assert not writer.prompts and not p._ahead


def test_queue_prefetch_never_raises(plugin_module, monkeypatch):
    p = provider(plugin_module, config=CONTRACT_ON)
    monkeypatch.setattr(p, "_last_exchange", lambda sid: (_ for _ in ()).throw(RuntimeError("x")))
    p.queue_prefetch("anything", session_id="s1")


# ---------------------------------------------------------------------------- the stage limits

STAGE_KEYS = {"decision_timeout_seconds": "decision_timeout_s", "writer_timeout_seconds": "writer_timeout_s",
              "search_timeout_seconds": "search_timeout_s"}
STAGE_ENV = {"decision_timeout_seconds": "PROSPECTA_PREFETCH_DECISION_TIMEOUT",
             "writer_timeout_seconds": "PROSPECTA_PREFETCH_WRITER_TIMEOUT",
             "search_timeout_seconds": "PROSPECTA_PREFETCH_SEARCH_TIMEOUT"}


def test_stage_limit_defaults(plugin_module):
    s = plugin_module.ProspectaProvider()._resolve_prefetch_settings()
    assert (s.decision_timeout_s, s.writer_timeout_s, s.search_timeout_s) == (1.0, 1.5, 4.5)
    assert not hasattr(s, "rank_timeout_s")          # the second rank call is gone: Stage 2 is the library's


@pytest.mark.parametrize("key", list(STAGE_KEYS))
def test_each_stage_limit_is_overridden_nested_flat_and_env(plugin_module, monkeypatch, key):
    attr = STAGE_KEYS[key]
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {key: "3.25"}}}
    s = p._resolve_prefetch_settings()
    assert getattr(s, attr) == 3.25
    assert [getattr(s, a) for a in STAGE_KEYS.values() if a != attr] == [
        getattr(pc.Settings(), a) for a in STAGE_KEYS.values() if a != attr]
    p._config = {f"recall.prefetch.{key}": 2}
    assert getattr(p._resolve_prefetch_settings(), attr) == 2.0
    monkeypatch.setenv(STAGE_ENV[key], "0.75")
    assert getattr(p._resolve_prefetch_settings(), attr) == 0.75


@pytest.mark.parametrize("key", list(STAGE_KEYS))
@pytest.mark.parametrize("bad", ["soon", "-1", "0", "nan", "inf"])
def test_bad_stage_limit_falls_back_with_one_warning(plugin_module, caplog, key, bad):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"prefetch": {key: bad}}}
    with caplog.at_level(logging.WARNING):
        s = p._resolve_prefetch_settings()
    assert getattr(s, STAGE_KEYS[key]) == getattr(pc.Settings(), STAGE_KEYS[key])
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and key in caplog.text


def _timed_search(plugin_module, search_s, **cfg):
    store = FakeStore(default=[item("roof", "the roof note")], delay=search_s)
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev, writer=FakeWriter())
    p._prefetch_cfg = pc.Settings(mode="contract", **cfg)
    return p.prefetch("What did we decide about the Henley branch roof?")


def test_a_search_of_4s_is_under_the_default_limit_and_returns_a_block(plugin_module):
    # the defaults of the stages add up to more than the deadline; the search is still given its 4.5 s
    out = _timed_search(plugin_module, 0.4, search_timeout_s=0.5, deadline_s=1.0)  # scaled: 0.4 < 0.5 limit
    assert "the roof note" in out and "not consulted" not in out
    s = pc.Settings()
    assert s.search_timeout_s == 4.5 and s.deadline_s >= 4.5 + 0.3


def test_a_slow_search_still_gets_a_block_through_the_real_defaults(plugin_module):
    out = _timed_search(plugin_module, 4.0, deadline_s=7.0)
    assert "the roof note" in out


def test_a_search_of_6s_gets_the_not_consulted_line(plugin_module):
    out = _timed_search(plugin_module, 6.0, deadline_s=7.0)
    assert out == "(memory was not consulted: the search did not answer in time)"


def test_the_deadline_wins_over_the_stage_limits(plugin_module):
    t0 = time.monotonic()
    out = _timed_search(plugin_module, 6.0, deadline_s=1.0, search_timeout_s=4.5)
    assert out == "(memory was not consulted: the search did not answer in time)"
    assert time.monotonic() - t0 < 1.5


def test_the_stage_budget_is_capped_by_what_is_left_of_the_deadline():
    now = [0.0]
    deps = pc.Deps(search=lambda q, scope: pc.Found([]), clock=lambda: now[0])
    run = pc._Run(deps, pc.Settings(deadline_s=7.0), pc.load_contract(), started=0.0)
    assert run.budget(4.5) == 4.5
    now[0] = 3.0
    assert run.budget(4.5) == 4.0
    now[0] = 8.0
    assert run.budget(4.5) == 0.0


# ---------------------------------------------------------------------------- what the writer is given

def _writer_input(prompt):
    return prompt[prompt.index("## Input"):]


def test_writer_kinds_are_any_when_no_need_reached_half():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Who was the person with at the cafe?"))
    b = run("do you remember who i saw it with?", make_deps(FakeStore(default=[item("a")]), jev, writer))
    assert b.sources == list(pc.ALL_SOURCES)          # the search still uses all of them
    text = _writer_input(writer.prompts[0])
    assert "The kinds of knowledge needed: any" in text
    assert not any(src in text.split("kinds of knowledge needed:")[1] for src in pc.ALL_SOURCES)
    assert OPENER_AGENT_LAST in text                  # the last answer, and only it, is the context


def test_writer_kinds_are_only_the_chosen_ones():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1, need_decisions=0.8))
    writer = FakeWriter(questions_reply("What was decided?"))
    run("and that?", make_deps(FakeStore(default=[item("a")]), jev, writer))
    kinds = _writer_input(writer.prompts[0]).split("The kinds of knowledge needed: ")[1].split("\n\n")[0].strip()
    assert kinds == pc.NEED_SOURCES["need_decisions"]


def test_writer_context_is_exactly_the_last_answer():
    seen = {}
    import prospecta._formulate as f
    real = f.formulate_queries

    def spy(line, **kw):
        seen["context"] = kw["context"]
        return real(line, **kw)
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    import unittest.mock as um
    with um.patch.object(f, "formulate_queries", spy):
        run("and that?", make_deps(FakeStore(default=[item("a")]), jev, FakeWriter(questions_reply("Q one?"))),
            agent_last="LAST ANSWER ONLY")
    assert seen["context"] == "LAST ANSWER ONLY"


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


def test_recovery_reaches_decision_and_writer_once_per_session(plugin_module, tmp_path, monkeypatch):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Who was the person with?"))
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "We met Ana at the cafe.", 1)],
                   store=FakeStore(default=[item("a")]), jev=jev, writer=writer)
    calls = []
    real = p._read_last_assistant
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: calls.append(sid) or real(sid))
    p.prefetch("do you remember who i saw it with?")
    p.prefetch("and then?")
    assert calls == ["s1"]
    assert jev.decisions()[0]["state"]["agent_last"] == "We met Ana at the cafe."
    assert "We met Ana at the cafe." in writer.prompts[0]
    assert "The kinds of knowledge needed: any" in writer.prompts[0]


def test_recovery_does_not_run_when_an_answer_is_held(plugin_module, tmp_path, monkeypatch):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "from the db", 1)], jev=jev,
                   writer=FakeWriter(questions_reply("Q?")), store=FakeStore(default=[item("a")]))
    _sync(p, "hello", "kept answer", monkeypatch)
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: pytest.fail("must not read"))
    p.prefetch("and that?")
    assert jev.decisions()[0]["state"]["agent_last"] == "kept answer"


def test_recovery_never_fails_the_turn(plugin_module, tmp_path, monkeypatch):
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "x", 1)])
    monkeypatch.setattr(p, "_read_last_assistant", lambda sid: (_ for _ in ()).throw(RuntimeError("boom")))
    p._recover_last_answer("s1")
    assert p._last_exchange("s1") == ("", 0)


# ---------------------------------------------------------------------------- no echoes: the thread's last turns, in SQL

def _turns_provider(plugin_module, monkeypatch, n=3, config=None, store=None):
    store = store or FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    p = provider(plugin_module, config=config if config is not None else CONTRACT_ON, store=store,
                 jev=FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1)),
                 writer=FakeWriter(questions_reply("What about the roof?")))
    for i in range(1, n + 1):
        _sync(p, f"user line {i}", f"assistant reply {i}", monkeypatch)
    return p, store


def test_the_last_two_turns_are_excluded_in_sql_and_older_turns_stay_eligible(plugin_module, monkeypatch):
    p, store = _turns_provider(plugin_module, monkeypatch, n=3)
    first, second, third = [k["source"] for k in p._retained]
    assert all(src.startswith("turn:s1:") for src in (first, second, third)) and len({first, second, third}) == 3
    assert p._echo_sources("s1") == [second, third]
    assert "The roof is to be replaced in June." in p.prefetch("and the roof?")
    (kw,) = store.kwargs
    assert kw["excluded_sources"] == [second, third] and first not in kw["excluded_sources"]
    assert kw["mode"] == "hybrid"


def test_a_session_with_one_turn_excludes_that_one(plugin_module, monkeypatch):
    p, store = _turns_provider(plugin_module, monkeypatch, n=1)
    p.prefetch("and the roof?")
    assert store.kwargs[0]["excluded_sources"] == [p._retained[0]["source"]]


def test_the_first_message_of_a_session_excludes_nothing(plugin_module, monkeypatch):
    p, store = _turns_provider(plugin_module, monkeypatch, n=0)
    p.prefetch("and the roof?")
    assert "excluded_sources" not in store.kwargs[0] and "date_from" not in store.kwargs[0]


def test_each_session_excludes_only_its_own_turns(plugin_module, monkeypatch):
    p, store = _turns_provider(plugin_module, monkeypatch, n=2)
    _sync(p, "other user", "other reply", monkeypatch, sid="s2")
    p.prefetch("and the roof?")
    p.prefetch("and the roof?", session_id="s2")
    s1 = [k["source"] for k in p._retained if k["source"].startswith("turn:s1:")]
    assert store.kwargs[0]["excluded_sources"] == s1
    assert store.kwargs[1]["excluded_sources"] == [p._retained[-1]["source"]]
    assert store.kwargs[1]["excluded_sources"][0].startswith("turn:s2:")


def test_drop_echoes_off_sends_no_exclusion(plugin_module, monkeypatch):
    cfg = {"prefetch_enabled": "true", "recall": {"prefetch": {"mode": "contract", "drop_echoes": False}}}
    p, store = _turns_provider(plugin_module, monkeypatch, n=3, config=cfg)
    p.prefetch("and the roof?")
    assert "excluded_sources" not in store.kwargs[0]


def test_drop_echoes_env_off_and_default_on(plugin_module, monkeypatch):
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is True
    monkeypatch.setenv("PROSPECTA_PREFETCH_DROP_ECHOES", "false")
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is False
    monkeypatch.setenv("PROSPECTA_PREFETCH_DROP_ECHOES", "true")
    assert provider(plugin_module)._prefetch_cfg.drop_echoes is True


def test_no_text_comparison_is_left_a_record_with_the_same_words_is_not_dropped(plugin_module, monkeypatch):
    """The old after-the-fact filter compared text; the library now excludes by source, so a note that
    merely holds the agent's words stays (and the 60-character rule for the last answer is gone)."""
    long_answer = "This is a long last answer of the agent that is over sixty characters."
    twin = item("twin", f"<user>: q\n\n<assistant>: {long_answer}", source="note:twin")
    p, store = _turns_provider(plugin_module, monkeypatch, n=0, store=FakeStore(default=[twin]))
    _sync(p, "q", long_answer, monkeypatch)
    out = p.prefetch("and the roof?")
    assert long_answer in out and not hasattr(pc, "ECHO_MIN_ANSWER_CHARS")


def test_after_a_restart_the_last_two_turns_come_from_the_bank_once(plugin_module, tmp_path, monkeypatch):
    store = FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    p = _restarted(plugin_module, tmp_path, [("s1", "assistant", "We spoke of the roof.", 1)], store=store,
                   jev=FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1)),
                   writer=FakeWriter(questions_reply("What about the roof?")))
    asked = []
    monkeypatch.setattr(p, "_read_turn_sources", lambda sid: asked.append(sid) or ["turn:s1:11", "turn:s1:12"])
    p.prefetch("and the roof?")
    p.prefetch("and the roof again?")
    assert asked == ["s1"]
    assert [k["excluded_sources"] for k in store.kwargs] == [["turn:s1:11", "turn:s1:12"]] * 2


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


# ---------------------------------------------------------------------------- Stage 0: the time scope

def _scoped(scope, *, writer=None, line="What did we decide about the Henley branch roof?", stands=0.95, store=None,
            decision=None):
    jev = FakeJev(decision=decision or jev_decision("new_ask", 0.9, stands=stands, scope=scope))
    store = store or FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    b = run(line, make_deps(store, jev, writer or FakeWriter(questions_reply("What was decided about the roof?"))))
    return b, store, jev


@pytest.mark.parametrize("scope,window", [
    ("recent", ("2026-10-16", "2026-10-17")),
    ("past_week", ("2026-10-10", "2026-10-17")),
    ("past_month", ("2026-09-16", "2026-10-17")),
    ("past_year", ("2025-10-16", "2026-10-17")),
])
def test_a_relative_time_scope_is_a_window_counted_back_from_today(scope, window):
    b, store, _ = _scoped(scope)
    (sc,) = store.scopes
    assert (sc.date_from, sc.date_to) == window and (b.date_from, b.date_to) == window and b.time_scope == scope
    assert f"Only notes dated {window[0]} to {window[1]} were searched." in b.text


def test_no_time_scope_is_no_window():
    b, store, _ = _scoped("none")
    (sc,) = store.scopes
    assert sc.date_from is None and sc.date_to is None and sc.window is None
    assert "were searched" not in b.text


def test_the_decision_request_carries_today_and_the_time_scope_question():
    _, _, jev = _scoped("none")
    (req,) = jev.decisions()
    assert req["state"]["today"] == "2026-10-17"
    assert req["questions"]["time_scope"]["type"] == "choice"
    assert list(req["questions"]["time_scope"]["criteria"]) == list(pc.TIME_SCOPES)


def test_a_named_period_is_the_question_writers_window_and_the_writer_always_runs():
    writer = FakeWriter(json.dumps({"questions": ["What did we decide in July about the roof?"],
                                    "date_from": "2026-07-01", "date_to": "2026-07-31"}))
    b, store, _ = _scoped("named", writer=writer, line="What did we decide in July about the Henley branch roof?")
    assert writer.prompts                               # a plain standalone question is not asked as written here
    assert (store.scopes[0].date_from, store.scopes[0].date_to) == ("2026-07-01", "2026-07-31")
    assert "Today: 2026-10-17" in writer.prompts[0]
    assert "The time scope the decision model found in the new lines: named" in writer.prompts[0]
    assert "date_from" in writer.prompts[0]             # the rule that asks for the dates


@pytest.mark.parametrize("reply", [
    {"questions": ["Q about the roof?"]},                                                  # no dates given
    {"questions": ["Q about the roof?"], "date_from": "2026-08-01", "date_to": "2026-07-01"},   # backwards
    {"questions": ["Q about the roof?"], "date_from": "July", "date_to": "August"},        # not dates
    {"questions": ["Q about the roof?"], "date_from": "1850-01-01", "date_to": "1850-02-01"},   # absurd
    {"questions": ["Q about the roof?"], "date_from": "2026-07-01", "date_to": "2031-01-01"},   # far future
    {"questions": ["Q about the roof?"], "date_from": "2026-07-01"},                       # half a window
])
def test_a_named_period_without_a_valid_window_searches_all_dates(reply):
    b, store, _ = _scoped("named", writer=FakeWriter(json.dumps(reply)))
    assert store.scopes[0].window is None and b.date_from is None and b.injected == 1


def test_a_named_period_with_a_failed_writer_searches_all_dates_and_still_asks():
    b, store, _ = _scoped("named", writer=FakeWriter(fail=RuntimeError("down")),
                          line="What did we decide in July about the Henley branch roof?")
    assert store.scopes[0].window is None and b.injected == 1


def test_the_writer_cannot_give_a_window_the_decision_model_did_not_find():
    writer = FakeWriter(json.dumps({"questions": ["What happened to the roof?"],
                                    "date_from": "2026-07-01", "date_to": "2026-07-31"}))
    b, store, _ = _scoped("none", writer=writer, stands=0.1)
    assert writer.prompts and store.scopes[0].window is None


def test_an_unknown_or_missing_time_scope_answer_is_no_window_never_no_decision():
    for mutate in (lambda a: a.pop("time_scope"), lambda a: a["time_scope"].update(choice="next_century"),
                   lambda a: a.update(time_scope="garbage")):
        decision = jev_decision("new_ask", 0.9, stands=0.95, scope="past_week")
        mutate(decision["answers"])
        b, store, _ = _scoped("none", decision=decision)
        assert b.decision == "ask" and store.scopes[0].window is None    # the decision stands, with no window


def test_a_contract_without_the_time_scope_question_still_works(tmp_path):
    import re
    text = pc.DEFAULT_CONTRACT_PATH.read_text()
    body = re.search(r"## jev-questions.*?```json\n(.*?)\n```", text, re.S).group(1)
    questions = json.loads(body)
    del questions["time_scope"]
    path = tmp_path / "old.md"
    path.write_text(text.replace(body, json.dumps(questions)))
    contract = pc.load_contract(path)
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week"))
    store = FakeStore(default=[item("roof")])
    b = pc.run_contract("What did we decide about the Henley branch roof?", agent_last="", turns=1,
                        deps=make_deps(store, jev, FakeWriter()), settings=SETTINGS, contract=contract)
    assert "time_scope" not in jev.decisions()[0]["questions"] and store.scopes[0].window is None and b.injected == 1


def test_a_closed_gate_searches_nothing_whatever_the_time_scope():
    jev = FakeJev(decision=jev_decision("new_ask", 0.2, stands=0.95, scope="past_week"))
    store = FakeStore(default=[item("a")])
    b = run("What did we decide about the Henley branch roof?", make_deps(store, jev, FakeWriter()))
    assert b.text == "" and not store.calls and b.time_scope == "past_week" and b.date_from is None


def test_nothing_answers_inside_a_window_says_which_window():
    b, _, _ = _scoped("past_week", store=FakeStore())
    assert b.text == ("Memory was asked: What did we decide about the Henley branch roof?. Nothing in it answers "
                      "this. Only notes dated 2026-10-10 to 2026-10-17 were searched.")


@pytest.mark.parametrize("scope", ["recent", "past_week", "past_month", "past_year", "named", "none"])
def test_the_window_math_is_pure(scope):
    w = pc.relative_window(scope, date(2026, 3, 1))
    assert (w is None) == (scope in ("none", "named"))
    if w:
        assert w[1] == "2026-03-01" and w[0] < w[1]


# ---------------------------------------------------------------------------- Stage 0: the query writer keeps "quoted phrases"

def test_quoted_phrases_are_found_in_straight_and_curly_quotes_once_each():
    assert pc.quoted_phrases('what about "electric dreams" and \u201cthe Panama deal\u201d, "electric dreams" again') == [
        "electric dreams", "the Panama deal"]
    assert pc.quoted_phrases('a 5" pipe') == []                       # an inch mark is not a quote
    assert pc.quoted_phrases('say "" or "x"? or " "') == []           # nothing worth matching
    assert pc.quoted_phrases('the "error 0x80070005" thing') == ["error 0x80070005"]
    assert pc.quoted_phrases("") == [] and pc.quoted_phrases(None) == []


def test_a_phrase_the_model_kept_is_left_alone():
    qs = ['What did we say about "electric dreams" at the party?']
    assert pc.keep_quoted('remind me about "electric dreams"', qs) == qs


def test_a_phrase_whose_words_stayed_but_lost_its_quotes_is_quoted_again_in_place():
    out = pc.keep_quoted('remind me about "electric dreams"', ["What did we say about electric dreams at the party?"])
    assert out == ['What did we say about "electric dreams" at the party?']


def test_a_phrase_the_model_dropped_is_put_back_on_the_first_question():
    out = pc.keep_quoted('remind me about "electric dreams"', ["What did we say at the party?", "Who came?"])
    assert out == ['What did we say at the party? "electric dreams"', "Who came?"]


def test_a_curly_quoted_phrase_is_written_with_straight_quotes_because_only_those_are_a_phrase_search():
    assert pc.keep_quoted("about \u201celectric dreams\u201d", ["What is \u201celectric dreams\u201d?"]) == [
        'What is "electric dreams"?']
    assert pc.keep_quoted("about \u201celectric dreams\u201d", ["What was said?"]) == ['What was said? "electric dreams"']


def test_several_phrases_all_survive_and_a_changed_case_is_not_the_phrase():
    out = pc.keep_quoted('"Electric Dreams" and "v1.14"', ["What about electric dreams and v1.14?"])
    assert '"Electric Dreams"' in out[0] and '"v1.14"' in out[0]      # the model changed the case: put back verbatim


def test_each_phrase_is_also_asked_on_its_own_so_the_exact_retriever_can_find_it():
    qs = ["What was said at the party?"]
    assert pc.with_phrase_queries('and "electric dreams"?', qs) == [qs[0], '"electric dreams"']
    assert pc.with_phrase_queries('and "electric dreams"?', [qs[0], '"electric dreams"']) == [qs[0], '"electric dreams"']
    many = pc.with_phrase_queries('"one two" "three four" "five six"', qs)
    assert many == [qs[0], '"one two"', '"three four"']                 # at most MAX_PHRASE_QUERIES
    assert pc.with_phrase_queries("no quotes", qs) == qs and pc.with_phrase_queries('"a b"', []) == []


def test_no_phrase_or_no_questions_changes_nothing():
    assert pc.keep_quoted("no quotes here", ["Q one?", "Q two?"]) == ["Q one?", "Q two?"]
    assert pc.keep_quoted('"x y"', []) == []


def test_a_quoted_phrase_survives_the_writers_output_into_the_search():
    line = 'and the "electric dreams" thing, who was there?'
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply('What did we say about "electric dreams" and who was there?'))
    store = FakeStore(default=[item("a")])
    b = run(line, make_deps(store, jev, writer))
    assert b.questions == ['What did we say about "electric dreams" and who was there?', '"electric dreams"']
    assert sorted(store.calls) == sorted(b.questions)


def test_a_quoted_phrase_the_writer_dropped_is_reinserted_before_the_search():
    line = 'and the "electric dreams" thing, who was there?'
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("What was said at the party and who was there?"))
    store = FakeStore(default=[item("a")])
    b = run(line, make_deps(store, jev, writer))
    assert b.questions == ['What was said at the party and who was there? "electric dreams"', '"electric dreams"']
    assert 'Asked: What was said at the party and who was there? "electric dreams"' in b.text


def test_the_writer_is_told_to_keep_quoted_phrases_whatever_the_contract_file_says():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Q about the roof?"))
    run("and that?", make_deps(FakeStore(default=[item("a")]), jev, writer))
    assert 'Keep every "quoted phrase" of the person\'s new lines exactly as written' in writer.prompts[0]
    assert "{%" not in writer.prompts[0]                              # the template text is rendered away


def test_a_standalone_question_with_a_quoted_phrase_is_asked_as_written():
    line = 'What did we decide about "electric dreams" last spring?'
    jev = FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95))
    writer, store = FakeWriter(), FakeStore(default=[item("a")])
    run(line, make_deps(store, jev, writer))
    assert sorted(store.calls) == sorted([line, '"electric dreams"']) and not writer.prompts


def test_when_the_writer_fails_the_phrase_stays_in_the_sentence_that_is_asked():
    def pick(request):
        return [3.0 if "house style" in r["instructions"]["text"] else 0.0 for r in request["questions"].values()]
    jev = FakeJev(decision=jev_decision("go_ahead", 0.9, stands=0.1), score=pick)
    store = FakeStore(default=[item("a")])
    run('ok, do it with "electric dreams"', make_deps(store, jev, FakeWriter(fail=RuntimeError("down"))))
    assert sorted(store.calls) == sorted(['Shall I write it in our explainer house style? "electric dreams"',
                                          '"electric dreams"'])


def test_a_follow_up_is_resolved_against_the_agents_last_line_only():
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1))
    writer = FakeWriter(questions_reply("Who did I see the cafe visit with?"))
    run("do you remember who i saw it with?", make_deps(FakeStore(default=[item("a")]), jev, writer),
        agent_last="We met Ana at the cafe on Tuesday.")
    text = writer.prompts[0]
    assert "The agent's last line: We met Ana at the cafe on Tuesday." in text
    assert "do you remember who i saw it with?" in text


# ---------------------------------------------------------------------------- what reaches the library's search

def test_the_exclusion_and_the_date_window_reach_the_librarys_search_call(plugin_module, monkeypatch):
    jev = FakeJev(decision=jev_decision("follow_up", 0.9, stands=0.1, scope="past_week"))
    store = FakeStore(default=[item("roof", "The roof is to be replaced in June.")])
    p = provider(plugin_module, config=CONTRACT_ON, store=store, jev=jev,
                 writer=FakeWriter(questions_reply("What about the roof?")))
    p._today = lambda: TODAY
    _sync(p, "user line 1", "assistant reply 1", monkeypatch)
    _sync(p, "user line 2", "assistant reply 2", monkeypatch)
    out = p.prefetch("and the roof?")
    (kw,) = store.kwargs
    assert kw["excluded_sources"] == [r["source"] for r in p._retained]
    assert (kw["date_from"], kw["date_to"]) == ("2026-10-10", "2026-10-17")
    assert kw["mode"] == "hybrid" and kw["limit"] == pc.SEARCH_LIMIT and isinstance(kw["_trace"], list)
    assert "depth" not in kw                                 # an ordinary turn: the library's standard depth
    assert "Only notes dated 2026-10-10 to 2026-10-17 were searched." in out


def test_a_library_that_raises_gives_the_not_consulted_line_and_logs_no_text(plugin_module, caplog):
    store = FakeStore(fail=ValueError("date_from, date_to and excluded_sources need the hybrid_chunk channel"))
    p = provider(plugin_module, config=CONTRACT_ON, store=store,
                 jev=FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95, scope="past_week")), writer=FakeWriter())
    with caplog.at_level(logging.INFO):
        out = p.prefetch("What did we decide about the Henley branch roof?")
    assert out == "(memory was not consulted: the search failed)"
    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []
