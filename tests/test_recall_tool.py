"""The explicit tools on the library's hybrid search: ``prospecta_search`` returns cited hits (source, date,
kind, how it matched, relevance, the matched words in context, the whole chunk) and ``prospecta_recall`` the
library's ``Memory.ask`` (Stage 1 + Stage 2, a grounded answer with [n] citations and the cited sources).

Offline (a fake Memory, invented text) except the last tests, which use the Postgres test container with
the stub embedder, the real library (Stage 1 ``hybrid_chunk`` + Stage 2 ``cross_encoder`` over a fake
rerank transport) and the plugin's real provider.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import psycopg
import pytest

from tests.conftest import CountingLLM, FakeRerank, check_call

CITED = {"n", "source", "document_id", "created_on", "kind", "match_kind", "relevance_score", "exact_snippet", "text"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_RECALL_STAGE3", "PROSPECTA_RERANKER_MODEL", "PROSPECTA_DECISION_MODEL"):
        monkeypatch.delenv(k, raising=False)


def hit(name, text=None, *, day="2026-03-02", kind="note", match="semantic", snippet=None, score=0.83):
    """A hit as the library's ``Memory.search`` returns it, with the cited-evidence fields."""
    return SimpleNamespace(
        content=f"content of {name}", original_chunk=text or f"stored text of {name}", source=f"note:{name}",
        score=0.0164, scores={"rrf": 0.0164}, metadata={}, bank_id="b", document_id=f"doc-{name}",
        created_on=day, kind=kind, match_kind=match, relevance_score=score, exact_snippet=snippet)


class FakeMemory:
    """Stands in for a Memory. Every call is checked against the real library's signature (``check_call``),
    so an argument the library removed (``mode``) or a method it no longer has (``recall_synth``) fails here."""

    def __init__(self, items=(), ranked=True, gate=None):
        self.items, self.ranked, self.gate = list(items), ranked, gate
        self.calls, self.asked = [], []

    def search(self, q, **kw):
        check_call("search", q, **kw)
        self.calls.append((q, kw))
        if kw.get("_trace") is not None:
            kw["_trace"].append({"rerank": {"fallback_reason": None} if self.ranked else None})
        return list(self.items[: kw.get("limit", 10)])

    def ask(self, q, **kw):
        check_call("ask", q, **kw)
        self.asked.append((q, kw))
        gated = bool(kw.get("sufficiency"))
        return SimpleNamespace(
            answer="The roof is to be replaced in June [1].", rendered="The roof is to be replaced in June [1].\n\nSources:\n[1] ...",
            citations=[{"n": 1, "known": True, "source": "note:roof", "created_on": "2026-03-02", "snippet": "s",
                        "exact_snippet": "the <<roof>> is replaced"}], items=self.items,
            sufficient=True if gated else None, retried=False,
            gate={"score": 0.9, "threshold": 0.7, "model": "typesafe/jev-1.14"} if gated else None)

    def _tracer(self, *a, **k):
        pass


def _p(plugin_module, items=(), config=None, **kw):
    p = plugin_module.ProspectaProvider()
    p._config = dict(config or {})
    p._memory = FakeMemory(items, **kw)
    p._llm = object()
    return p


def _call(p, tool="prospecta_search", **args):
    return json.loads(p.handle_tool_call(tool, {"query": "q", **args}))


# ---- prospecta_search: every hit is cited ----

def test_every_hit_carries_its_citation_fields_and_the_matched_words(plugin_module):
    p = _p(plugin_module, [
        hit("a", day="2026-03-02", match="exact+semantic", snippet="the <<roof>> failed", score=0.91),
        hit("b", day="2026-04-01", kind="conversation", match="semantic"),
        hit("c", day="2026-05-09", match="lexical", snippet="a <<leak>> in the roof")])
    out = _call(p)
    assert out["count"] == 3 and out["reranked"] is True
    assert all(set(r) == CITED for r in out["results"])
    a, b, c = out["results"]
    assert (a["n"], a["source"], a["created_on"], a["kind"], a["match_kind"], a["relevance_score"]) == (
        1, "note:a", "2026-03-02", "note", "exact+semantic", 0.91)
    assert a["exact_snippet"] == "the <<roof>> failed" and a["text"] == "stored text of a"
    assert b["exact_snippet"] is None and b["kind"] == "conversation" and c["match_kind"] == "lexical"
    assert [r["n"] for r in out["results"]] == [1, 2, 3]


def test_a_hit_with_no_citation_fields_still_answers_with_nulls(plugin_module):
    legacy = SimpleNamespace(content="old content", original_chunk="old chunk", source="note:old", score=0.4,
                             scores={}, metadata={}, bank_id="b", document_id="doc-old")
    (r,) = _call(_p(plugin_module, [legacy]))["results"]
    assert set(r) == CITED and r["created_on"] is None and r["kind"] is None and r["match_kind"] is None
    assert r["relevance_score"] is None and r["exact_snippet"] is None and r["text"] == "old chunk"


def test_search_says_whether_stage_2_graded_the_hits(plugin_module):
    assert _call(_p(plugin_module, [hit("a")], ranked=True))["reranked"] is True
    assert _call(_p(plugin_module, [hit("a")], ranked=False))["reranked"] is False


def test_search_passes_limit_depth_and_a_validated_date_window(plugin_module):
    p = _p(plugin_module, [hit("a")])
    _call(p, limit=7, depth="deep", date_from="2026-07-01", date_to=" 2026-07-31 ")
    (q, kw), = p._memory.calls
    assert q == "q" and kw["limit"] == 7 and kw["depth"] == "deep"
    assert kw["date_from"] == "2026-07-01" and kw["date_to"] == "2026-07-31"
    assert "excluded_sources" not in kw                         # the agent's own search leaves nothing out


def test_search_never_passes_mode_even_when_an_old_prompt_still_sends_one(plugin_module):
    """The library has one retrieval path (Stage 1 + Stage 2): ``mode`` is gone from its search. A model
    that still sends the argument gets the same search, not an error."""
    p = _p(plugin_module, [hit("a")])
    out = _call(p, mode="lexical", limit=3)
    assert out["count"] == 1 and "mode" not in p._memory.calls[0][1]


def test_search_sends_no_date_arguments_unless_given(plugin_module):
    p = _p(plugin_module, [hit("a")])
    _call(p)
    _call(p, date_from="2026-07-01")
    first, second = (kw for _, kw in p._memory.calls)
    assert "date_from" not in first and "date_to" not in first
    assert second["date_from"] == "2026-07-01" and "date_to" not in second


@pytest.mark.parametrize("bad", ["July", "2026-13-01", "yesterday", "2026/07/01", 5])
def test_a_bad_date_is_an_error_the_agent_can_read(plugin_module, bad):
    p = _p(plugin_module, [hit("a")])
    out = _call(p, date_to=bad)
    assert out == {"error": "date_to must be a date, YYYY-MM-DD"} and not p._memory.calls


def test_a_search_failure_is_an_error_not_a_crash(plugin_module):
    p = _p(plugin_module, [hit("a")])
    p._memory.search = lambda *a, **k: (_ for _ in ()).throw(ValueError("date_from, date_to and excluded_sources need the hybrid_chunk channel"))
    assert "hybrid_chunk" in _call(p, date_from="2026-07-01")["error"]


def test_the_search_schema_and_guide_name_the_citation_fields_and_the_window(plugin_module):
    p = plugin_module.ProspectaProvider()
    schemas = {s["name"]: s for s in p.get_tool_schemas()}
    props = schemas["prospecta_search"]["parameters"]["properties"]
    assert set(props) == {"query", "limit", "depth", "date_from", "date_to"}      # no mode: one retrieval path
    assert "exact_snippet" in schemas["prospecta_search"]["description"]
    assert "date_from" not in schemas["prospecta_recall"]["parameters"]["properties"]
    assert set(schemas["prospecta_retain"]["parameters"]["properties"]) == {"content", "source", "tags"}
    assert "chunks, with no model call" in schemas["prospecta_retain"]["description"].replace("\n", " ")
    guide = plugin_module._TOOL_GUIDE
    assert "exact_snippet" in guide and "date_from / date_to" in guide and "keyword" not in guide.lower()


# ---- prospecta_recall: the library's ask() (Mode 2B); Stage 3 only when the profile enables it ----

def test_recall_answers_through_ask_with_stage_1_and_stage_2_and_no_stage_3_by_default(plugin_module):
    p = _p(plugin_module, [hit("roof", "The roof is to be replaced in June.", match="exact+semantic",
                               snippet="the <<roof>> is replaced")])
    out = _call(p, "prospecta_recall", limit=6, depth="deep")
    (q, kw), = p._memory.asked
    assert q == "q" and kw == {"limit": 6, "sufficiency": False, "depth": "deep"}   # Stage 3 explicitly off
    assert p._memory.calls == []                                   # one ask(): no separate search
    assert out["answer"] == "The roof is to be replaced in June [1]." and out["synthesis"].endswith("[1] ...")
    assert out["citations"][0]["source"] == "note:roof" and out["citations"][0]["snippet"] == "s"
    assert out["citations"][0]["exact_snippet"] == "the <<roof>> is replaced"
    (src,) = out["sources"]
    assert src["n"] == 1 and src["source"] == "note:roof" and src["created_on"] == "2026-03-02"
    assert src["kind"] == "note" and src["match_kind"] == "exact+semantic" and src["relevance_score"] == 0.83
    assert src["exact_snippet"] == "the <<roof>> is replaced" and "text" not in src   # the answer already read it
    assert not {"sufficient", "retried", "gate"} & set(out)       # no Stage 3 ran, so none is reported


def test_recall_sends_no_depth_when_the_agent_gives_none_and_a_default_limit_of_ten(plugin_module):
    p = _p(plugin_module, [hit("a")])
    _call(p, "prospecta_recall")
    assert p._memory.asked == [("q", {"limit": 10, "sufficiency": False})]


def test_recall_never_calls_the_removed_recall_synth(plugin_module):
    """FakeMemory has no ``recall_synth`` (the library removed it): a call would be an error result."""
    p = _p(plugin_module, [hit("a")])
    assert "error" not in _call(p, "prospecta_recall")


def test_stage_3_is_off_by_default_and_read_from_config_or_env(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    assert p._recall_stage3() is False
    p._config = {"recall.stage3": "true"}
    assert p._recall_stage3() is True
    p._config = {"recall": {"stage3": "on"}}
    assert p._recall_stage3() is True
    p._config = {"recall": {"stage3": "nonsense"}}
    assert p._recall_stage3() is False
    monkeypatch.setenv("PROSPECTA_RECALL_STAGE3", "1")
    assert p._recall_stage3() is True
    assert "recall.stage3" in {f["key"] for f in p.get_config_schema()}


def test_recall_adds_the_stage_3_gate_only_when_the_profile_enables_it(plugin_module):
    p = _p(plugin_module, [hit("roof", "The roof is to be replaced in June.")], config={"recall": {"stage3": "true"}})
    out = _call(p, "prospecta_recall", limit=6, depth="deep")
    (q, kw), = p._memory.asked
    assert q == "q" and kw == {"limit": 6, "sufficiency": True, "depth": "deep"}
    assert out["answer"] == "The roof is to be replaced in June [1]." and out["synthesis"].endswith("[1] ...")
    assert out["citations"][0]["source"] == "note:roof" and out["sufficient"] is True and out["retried"] is False
    assert out["gate"]["threshold"] == 0.7
    (src,) = out["sources"]
    assert src["source"] == "note:roof" and src["relevance_score"] == 0.83 and "text" not in src


def test_recall_still_needs_an_llm(plugin_module):
    p = _p(plugin_module, [hit("a")])
    p._llm = None
    assert "PROSPECTA_LLM_MODEL" in _call(p, "prospecta_recall")["error"]


def test_the_stage_2_and_3_arguments_the_provider_gives_memory(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {"reranker_model": "x/reranker-cfg", "decision_model": "typesafe/jev-9"}
    kw = p._recall_stage_kwargs()
    assert kw["reranker_model"] == "x/reranker-cfg" and kw["decision_model"] == "typesafe/jev-9"
    assert callable(kw["rerank_transport"]) and "jev" not in kw        # Stage 3 is off: no Jev built
    p._config["recall"] = {"stage3": "true"}
    kw = p._recall_stage_kwargs()
    assert kw["jev"].model == "typesafe/jev-9" and callable(kw["jev"].transport)


def test_with_no_profile_models_the_provider_hands_the_library_none_so_it_resolves_them(plugin_module):
    from prospecta.stages import JEV_MODEL
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"stage3": "true"}}
    kw = p._recall_stage_kwargs()
    assert kw["reranker_model"] is None and kw["decision_model"] is None   # bank config, env, default: the library's
    assert kw["jev"].model == JEV_MODEL


# ---- the shadow log: false-abstain rate beside every abstention figure ----

OLD, NEW = "old_bank", "new_bank"


def _rm(source, bank):
    return SimpleNamespace(content=f"c {source}", original_chunk=f"chunk {source}", source=source, score=0.5,
                           scores={}, metadata={}, bank_id=bank, document_id=f"{bank}-{source}",
                           created_on=None, kind=None, match_kind=None, relevance_score=0.5, exact_snippet=None)


class NewMemory:
    def __init__(self, results):
        self.results, self.closed = results, False

    def search(self, q, **kw):
        check_call("search", q, **kw)
        return list(self.results)

    def close(self):
        self.closed = True


@pytest.fixture
def origin_ids(monkeypatch):
    import prospecta._shadow as lib
    monkeypatch.setattr(lib, "_origin_ids",
                        lambda memory, bank, results: [r.document_id.split("-", 1)[1] for r in results])


def _shadow_provider(plugin_module, old, new):
    p = plugin_module.ProspectaProvider()
    p._memory = FakeMemory([_rm(s, OLD) for s in old])
    p._memory.events = []
    p._memory._tracer = lambda kind, payload: p._memory.events.append(payload)
    p._shadow = plugin_module._ShadowReader(
        p._memory, NewMemory([_rm(s, NEW) for s in new]), bank_id=OLD, shadow_bank_id=NEW)
    return p


def _drain(p):
    shadow, p._shadow = p._shadow, None
    shadow.stop()


def test_shadow_pair_carries_counts_and_the_false_abstain_rate(plugin_module, origin_ids, caplog):
    """A question is answerable when either side found something; a side that found nothing then abstained
    falsely. (The library's lexical-only search is gone, so no third leg is asked.)"""
    p = _shadow_provider(plugin_module, old=[], new=["a", "c"])
    with caplog.at_level(logging.INFO):
        p.handle_tool_call("prospecta_search", {"query": "secret words"})
        p.handle_tool_call("prospecta_search", {"query": "secret words"})
        _drain(p)
    old_row, new_row = p._memory.events[:2]
    sh = old_row["trace"]["shadow"]
    assert (sh["old_n"], sh["new_n"]) == (0, 2) and "kw_n" not in sh
    assert sh["abstain"]["old_abstain"] == 1 and sh["abstain"]["old_false_abstain"] == 1
    assert sh["abstain"]["answerable"] == 1
    last = p._memory.events[-1]["trace"]["shadow"]["abstain"]
    assert last["pairs"] == 2 and last["old_false_abstain"] == 2
    line = [r.getMessage() for r in caplog.records if "shadow pair" in r.getMessage()][-1]
    assert "abstain old 2/2 (100.0%) false-abstain old 2/2 (100.0%)" in line
    assert "abstain new 0/2 (0.0%) false-abstain new 0/2 (0.0%)" in line
    assert "old_n=0 new_n=2" in line and "kw_n" not in line
    assert "secret" not in line and "chunk" not in line     # counts only, no content
    assert len(p._memory.calls) == 2                        # the old bank is searched by the tool only, not again for evidence


def test_shadow_compares_the_same_cited_list_the_agent_got(plugin_module, origin_ids):
    p = _shadow_provider(plugin_module, old=["a", "b"], new=["a", "c"])
    p.handle_tool_call("prospecta_search", {"query": "q", "limit": 5})
    _drain(p)
    assert p._memory.events[0]["trace"]["shadow"]["old_origin_document_ids"] == ["a", "b"]


def test_shadow_recall_compares_the_notes_ask_cited(plugin_module, origin_ids):
    p = _shadow_provider(plugin_module, old=["a"], new=[])
    p._llm = object()
    p.handle_tool_call("prospecta_recall", {"query": "q"})
    _drain(p)
    old_row, new_row = p._memory.events
    sh = old_row["trace"]["shadow"]
    assert sh["tool"] == "prospecta_recall" and sh["old_n"] == 1 and sh["new_n"] == 0
    assert sh["abstain"]["old_false_abstain"] == 0 and sh["abstain"]["new_false_abstain"] == 1
    assert old_row["synthesis"] == "The roof is to be replaced in June [1]." and new_row["synthesis"] is None
    assert old_row["mode"] == new_row["mode"] == "hybrid"


# ---- real Postgres, the real library: Stage 1 + Stage 2 ----

def _rows(pg_url, sql, *args):
    with psycopg.connect(pg_url) as c:
        return c.execute(sql, args).fetchall()


def test_real_search_returns_the_matched_words_and_citation_metadata_on_every_hit(hybrid_bank):
    p = hybrid_bank()
    p._retain("We watched Electric Dreams at the party and everyone cheered.", source="note:party",
              metadata={"created_on": "2026-10-12", "source_kind": "conversation"})
    p._retain("Dreams are electric when the lights go out on the hill.", source="note:scattered",
              metadata={"created_on": "2026-10-11"})
    out = _call(p, query='"electric dreams"')
    assert out["reranked"] is True and out["count"] == 2
    assert all(set(r) == CITED for r in out["results"])
    party = next(r for r in out["results"] if r["source"] == "note:party")
    assert party["exact_snippet"] and "<<Electric>> <<Dreams>>" in party["exact_snippet"]
    assert party["match_kind"] in ("exact", "exact+semantic") and party["created_on"] == "2026-10-12"
    assert party["kind"] == "conversation" and party["relevance_score"] == 1.0
    assert party["text"] == "We watched Electric Dreams at the party and everyone cheered."
    other = next(r for r in out["results"] if r["source"] == "note:scattered")
    assert other["exact_snippet"] is None                          # the words are not side by side
    assert p.rerank.requests and "Query: \"electric dreams\"" in p.rerank.requests[0]["query"]


def test_real_search_window_and_a_failing_reranker(hybrid_bank):
    p = hybrid_bank(rerank=FakeRerank(fail=RuntimeError("HTTP 503")))
    p._retain("The Henley branch roof will be replaced in June.", source="note:new", metadata={"created_on": "2026-10-12"})
    p._retain("The Henley branch roof leaks in July.", source="note:old", metadata={"created_on": "2026-07-14"})
    out = _call(p, query="Henley branch roof", date_from="2026-10-01", date_to="2026-10-31")
    assert [r["source"] for r in out["results"]] == ["note:new"]
    assert out["reranked"] is False                                # Stage 2 failed: Stage 1's own order, said so
    assert out["results"][0]["relevance_score"] is not None


class Writer:
    """The answer model of ``ask()`` (the Memory's ``llm``): it records the prompt it was given."""

    def __init__(self, answer="The roof will be replaced in June [1]."):
        self.answer, self.prompts = answer, []

    def __call__(self, messages=None, *, json_mode=False, **kw):
        self.prompts.append(messages[0]["content"])
        return self.answer


def test_real_recall_is_the_librarys_ask_stage_1_and_stage_2_then_one_cited_answer(hybrid_bank):
    writer = Writer()
    p = hybrid_bank(llm=writer)
    p._retain("The Henley branch roof will be replaced in June.", source="note:new", metadata={"created_on": "2026-10-12"})
    p._retain("The Henley branch roof leaks; the board will look at it in July.", source="note:old",
              metadata={"created_on": "2026-07-14"})
    out = _call(p, "prospecta_recall", query="When will the Henley branch roof be replaced?")
    assert out["answer"] == "The roof will be replaced in June [1]."
    assert out["synthesis"].startswith(out["answer"]) and "Sources:" in out["synthesis"]
    assert p.rerank.requests                                         # Stage 2 ran (the cross-encoder, over the fake transport)
    assert len(writer.prompts) == 1                                  # one model call: the answer (no query writer, no formulate)
    assert "The Henley branch roof will be replaced in June." in writer.prompts[0]   # grounded in the cited chunks
    assert "[1] source=" in writer.prompts[0] and "date=2026-10-12" in writer.prompts[0]
    first = out["sources"][0]
    assert set(first) >= CITED - {"text"} and "text" not in first
    assert first["created_on"] and first["match_kind"] and first["relevance_score"] is not None
    cited = {c["n"]: c for c in out["citations"]}
    assert cited[1]["known"] and cited[1]["source"] == first["source"] and cited[1]["snippet"]
    assert "sufficient" not in out and p._memory._jev is None       # Stage 3 is off: no gate, no decision model call


def test_real_tools_on_an_old_shaped_bank_an_empty_channel_config_and_no_recall_config(hybrid_bank):
    """augur and forge store ``channel_config = []`` and no ``recall_config``: the library reads that as its default
    Stage 1 (no Stage 2), and both explicit tools work on it."""
    p = hybrid_bank(hybrid=False, llm=Writer())
    p._retain("The Henley branch roof will be replaced in June.", source="note:new", metadata={"created_on": "2026-10-12"})
    found = _call(p, query="Henley branch roof", date_from="2026-10-01")
    assert [r["source"] for r in found["results"]] == ["note:new"] and found["reranked"] is False
    asked = _call(p, "prospecta_recall", query="When is the Henley branch roof replaced?")
    assert asked["answer"] == "The roof will be replaced in June [1]." and [s["source"] for s in asked["sources"]] == ["note:new"]
    assert not p.rerank.requests                                    # no Stage 2 on this bank


def test_real_recall_exposes_the_exact_snippet_of_a_phrase_match(hybrid_bank):
    p = hybrid_bank(llm=Writer("They watched it at the party [1]."))
    p._retain("We watched Electric Dreams at the party and everyone cheered.", source="note:party",
              metadata={"created_on": "2026-10-12"})
    out = _call(p, "prospecta_recall", query='"electric dreams"')
    (src,) = out["sources"]
    assert src["source"] == "note:party" and "<<Electric>> <<Dreams>>" in src["exact_snippet"]
    (cite,) = out["citations"]
    assert cite["source"] == "note:party" and "<<Electric>> <<Dreams>>" in cite["exact_snippet"]


def test_real_recall_with_nothing_in_memory_says_so_and_calls_no_model(hybrid_bank):
    writer = Writer()
    p = hybrid_bank(llm=writer)
    out = _call(p, "prospecta_recall", query="What colour is the Henley branch roof?")
    assert out["answer"] == "not in memory" and out["sources"] == [] and out["citations"] == []
    assert writer.prompts == []


def test_real_recall_honours_depth_and_limit(hybrid_bank, pg_url):
    p = hybrid_bank(llm=Writer())
    for i in range(5):
        p._retain(f"The Henley branch roof report number {i}: the roof needs work.", source=f"note:{i}",
                  metadata={"created_on": "2026-10-12"})
    out = _call(p, "prospecta_recall", query="Henley branch roof report", limit=2, depth="deep")
    assert len(out["sources"]) == 2
    assert "error" not in _call(p, "prospecta_recall", query="Henley branch roof", depth="standard")
    assert [r[0] for r in _rows(pg_url, "SELECT depth FROM recall_events ORDER BY created_at")] == ["deep", "standard"]


def test_real_recall_through_ask_with_the_stage_3_gate_on(hybrid_bank):
    from prospecta.stages import JevScore

    seen = []

    def gate(request, timeout):
        seen.append(request)
        return {"answers": {"p0": {"type": "score", "score": 3}}, "usage": {"cost": 0.0}}

    p = hybrid_bank(config={"recall": {"stage3": "true"}}, llm=Writer())
    p._memory._jev = JevScore(gate)                                # the library's own model, no profile value
    p._retain("The Henley branch roof will be replaced in June.", source="note:new", metadata={"created_on": "2026-10-12"})
    out = _call(p, "prospecta_recall", query="Henley branch roof")
    assert out["answer"] == "The roof will be replaced in June [1]."
    assert out["sufficient"] is True and out["retried"] is False and out["gate"]["model"] == "typesafe/jev-1.14"
    assert len(seen) == 1 and seen[0]["model"] == "typesafe/jev-1.14"          # the gate ran, on the decision model
    (cite,) = out["citations"]
    assert cite["n"] == 1 and cite["known"] and cite["source"] == "note:new" and cite["created_on"] == "2026-10-12"
    assert cite["snippet"] == "The Henley branch roof will be replaced in June."
    assert out["synthesis"].startswith("The roof will be replaced in June [1].") and "Sources:" in out["synthesis"]
    assert [s["source"] for s in out["sources"]] == ["note:new"]


def test_a_retain_call_that_still_sends_index_text_writes_the_note_as_chunks_only(hybrid_bank, pg_url):
    """The tool has no ``index_text`` argument any more (the library has none): a model that still sends one
    gets the note saved as chunks, findable by Stage 1, with no question item."""
    p = hybrid_bank()
    out = json.loads(p.handle_tool_call("prospecta_retain", {
        "content": "Augur Prospecta live smoke: host-wide Roger PostgreSQL is active.",
        "source": "smoke:index-text", "index_text": "What is Augur using as his Prospecta substrate?"}))
    assert out["source"] == "smoke:index-text" and "error" not in out
    kinds = sorted({r[0] for r in _rows(pg_url, "SELECT kind FROM memory_items")})
    assert kinds == ["chunk"]
    hits = _call(p, query="Prospecta substrate PostgreSQL")["results"]
    assert [h["source"] for h in hits] == ["smoke:index-text"] and hits[0]["text"].startswith("Augur Prospecta live smoke")
