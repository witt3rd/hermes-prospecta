"""prospecta_search: keyword leg + keyword-first union + one token-budget block.

Offline (fake Memory, fake keyword leg, invented text) except the last tests,
which use the Postgres test container with the stub embedder.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

import recall_fusion as rf
from tests.conftest import stub_embed, stub_llm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PROSPECTA_RECALL_FUSION", "PROSPECTA_RECALL_BLOCK_TOKENS", "PROSPECTA_RECALL_KEYWORD",
              "PROSPECTA_RECALL_KEYWORD_UNIT", "PROSPECTA_RECALL_KEYWORD_INDEX_WARN_MB",
              "PROSPECTA_RECALL_KEYWORD_PAUSE_MS"):
        monkeypatch.delenv(k, raising=False)


def _store_item(name, text=None, date=None, doc=None):
    meta = {"created_on": date} if date else {}
    return SimpleNamespace(
        content=f"question about {name}", original_chunk=text or f"stored text of {name}", source=f"src-{name}",
        score=0.5, scores={"semantic": 0.5, "lexical": 0.0, "lexical_body": 0.0, "rrf": 0.5},
        metadata=meta, bank_id="b", document_id=doc or f"doc-{name}")


def _kw_hit(name, text=None, date="2026-02-02"):
    t = text or f"keyword text of {name}"
    return rf.KeywordHit(content=t, original_chunk=t, source=f"src-{name}", score=3.0, scores={"bm25": 3.0},
                         metadata={}, bank_id="b", document_id=f"doc-{name}", created_on=date)


class FakeMemory:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def search(self, q, **kw):
        self.calls.append((q, kw))
        return list(self.items[: kw.get("limit", 10)])

    def _tracer(self, *a, **k):
        pass


class FakeLeg:
    def __init__(self, hits=(), status="ok", unit="chunk", docs=None):
        self.hits, self.status, self.calls = list(hits), status, []
        self.unit, self.docs = unit, dict(docs or {})

    def documents(self, ids):
        return {i: self.docs[i] for i in ids if i in self.docs}

    def search(self, q, limit):
        self.calls.append((q, limit))
        return self.status, list(self.hits)[:limit]

    def stop(self):
        pass


def _p(plugin_module, store=(), kw=None, config=None):
    p = plugin_module.ProspectaProvider()
    p._config = dict(config or {})
    p._memory = FakeMemory(store)
    p._keyword = kw
    return p


def _search(p, **args):
    return json.loads(p.handle_tool_call("prospecta_search", {"query": "q", **args}))


# ---- settings ----

def test_defaults_are_union_1500_keyword_on(plugin_module):
    p = plugin_module.ProspectaProvider()
    assert p._recall_fusion() == "keyword_first_union"
    assert p._recall_block_tokens() == 1500
    assert p._recall_keyword_enabled() is True


def test_settings_from_nested_and_dotted_config_and_env(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"fusion": "store", "block_tokens": 800, "keyword": False}}
    assert (p._recall_fusion(), p._recall_block_tokens(), p._recall_keyword_enabled()) == ("store", 800, False)
    p._config = {"recall.fusion": "store", "recall.block_tokens": "0", "recall.keyword": "false"}
    assert (p._recall_fusion(), p._recall_block_tokens(), p._recall_keyword_enabled()) == ("store", 0, False)
    monkeypatch.setenv("PROSPECTA_RECALL_FUSION", "keyword_first_union")
    monkeypatch.setenv("PROSPECTA_RECALL_BLOCK_TOKENS", "300")
    monkeypatch.setenv("PROSPECTA_RECALL_KEYWORD", "on")
    assert (p._recall_fusion(), p._recall_block_tokens(), p._recall_keyword_enabled()) == (
        "keyword_first_union", 300, True)


def test_bad_settings_fall_back_to_defaults_loudly(plugin_module, caplog):
    p = plugin_module.ProspectaProvider()
    p._config = {"recall": {"fusion": "rrf++", "block_tokens": "lots"}}
    with caplog.at_level(logging.WARNING):
        assert p._recall_fusion() == "keyword_first_union"
        assert p._recall_block_tokens() == 1500
    assert "rrf++" in caplog.text and "lots" in caplog.text
    p._config = {"recall": {"block_tokens": -5}}
    assert p._recall_block_tokens() == 1500


def test_keyword_unit_and_index_options(plugin_module, monkeypatch, caplog):
    p = plugin_module.ProspectaProvider()
    assert p._recall_keyword_unit() == "documents"
    assert p._recall_float("PROSPECTA_RECALL_KEYWORD_INDEX_WARN_MB", "keyword_index_warn_mb", 0.0) == 0.0
    p._config = {"recall": {"keyword_unit": "chunks", "keyword_index_warn_mb": 300, "keyword_batch_pause_ms": 0}}
    assert p._recall_keyword_unit() == "chunks"
    assert p._recall_float("PROSPECTA_RECALL_KEYWORD_INDEX_WARN_MB", "keyword_index_warn_mb", 0.0) == 300.0
    assert p._recall_float("PROSPECTA_RECALL_KEYWORD_PAUSE_MS", "keyword_batch_pause_ms", 50.0) == 0.0
    monkeypatch.setenv("PROSPECTA_RECALL_KEYWORD_UNIT", "documents")
    assert p._recall_keyword_unit() == "documents"
    p._config = {"recall": {"keyword_unit": "paragraphs", "keyword_index_warn_mb": "lots"}}
    monkeypatch.delenv("PROSPECTA_RECALL_KEYWORD_UNIT")
    with caplog.at_level(logging.WARNING):
        assert p._recall_keyword_unit() == "documents"
        assert p._recall_float("PROSPECTA_RECALL_KEYWORD_INDEX_WARN_MB", "keyword_index_warn_mb", 0.0) == 0.0
    assert "paragraphs" in caplog.text and "lots" in caplog.text


def test_config_schema_lists_the_three_options(plugin_module):
    schema = {f["key"]: f for f in plugin_module.ProspectaProvider().get_config_schema()}
    assert schema["recall.fusion"]["default"] == "keyword_first_union"
    assert schema["recall.fusion"]["choices"] == ["keyword_first_union", "store"]
    assert schema["recall.block_tokens"]["default"] == "1500"
    assert schema["recall.keyword"]["default"] == "true"
    assert schema["recall.keyword_unit"]["default"] == "documents"
    assert schema["recall.keyword_unit"]["choices"] == ["documents", "chunks"]
    assert schema["recall.keyword_index_warn_mb"]["default"] == "0"
    assert schema["recall.keyword_batch_pause_ms"]["default"] == "50"


# ---- the block ----

def test_search_returns_one_block_keyword_first_tagged_with_source_and_date(plugin_module):
    store = [_store_item(n, date="2026-01-0%d" % i) for i, n in enumerate("abcde", 1)]
    leg = FakeLeg([_kw_hit("k1"), _kw_hit("k2")])
    out = _search(_p(plugin_module, store, leg))
    block = out["block"]
    assert isinstance(block, str)
    order = [block.index(s) for s in ("src-k1", "src-k2", "src-a", "src-b", "src-c")]
    assert order == sorted(order)
    assert "[source: src-k1 | date: 2026-02-02]" in block
    assert "[source: src-a | date: 2026-01-01]" in block
    assert "keyword text of k1" in block and "stored text of a" in block   # raw text, not the question
    assert "question about" not in block
    assert out["keyword"] == "ok" and out["entries"] == len(out["sources"])
    assert out["budget"] == 1500 and 0 < out["tokens"] <= 1500
    assert "results" not in out


def _doc_hit(doc, text, date="2026-04-04", source=None):
    return rf.KeywordHit(content=text, original_chunk=text, source=source or f"src-{doc}", score=0.0, scores={},
                         metadata={}, bank_id="b", document_id=f"doc-{doc}", created_on=date, origin="store")


def test_document_mode_one_entry_per_note_with_the_notes_own_text(plugin_module):
    # the store returns two question items of note a and one of note b; the leg also found note a
    store = [_store_item("a", "part one of a"), _store_item("a", "part two of a"), _store_item("b", "part of b")]
    docs = {"doc-a": _doc_hit("a", "the whole of note a"), "doc-b": _doc_hit("b", "the whole of note b")}
    leg = FakeLeg([_kw_hit("a", "the whole of note a")], unit="document", docs=docs)
    out = _search(_p(plugin_module, store, leg))
    assert [s["source"] for s in out["sources"]] == ["src-a", "src-b"]
    assert out["block"].count("the whole of note a") == 1 and "the whole of note b" in out["block"]
    assert "part one" not in out["block"] and "part of b" not in out["block"]
    assert "[source: src-b | date: 2026-04-04]" in out["block"]            # the note's own date


def test_document_mode_note_found_only_by_the_store_reads_the_same_text(plugin_module):
    store = [_store_item("a", "a chunk of a")]
    leg = FakeLeg([], unit="document", docs={"doc-a": _doc_hit("a", "the whole of note a")})
    out = _search(_p(plugin_module, store, leg))
    assert "the whole of note a" in out["block"] and "a chunk of a" not in out["block"]


def test_document_mode_keeps_the_stores_text_for_a_note_newer_than_the_index(plugin_module):
    store = [_store_item("new", "text of a note written since the build")]
    out = _search(_p(plugin_module, store, FakeLeg([], unit="document")))
    assert "text of a note written since the build" in out["block"]


def test_document_mode_dedupes_the_union_by_document_id_across_legs(plugin_module):
    store = [_store_item(n) for n in "abc"]
    docs = {f"doc-{n}": _doc_hit(n, f"note {n}") for n in "abcx"}
    leg = FakeLeg([_kw_hit("x"), _kw_hit("b"), _kw_hit("a")], unit="document", docs=docs)
    out = _search(_p(plugin_module, store, leg))
    assert [s["source"] for s in out["sources"]] == ["src-x", "src-b", "src-a", "src-c"]
    assert out["sources"][1]["origin"] == "keyword"


def test_document_mode_old_note_list_uses_note_text_and_fusion_store_is_untouched(plugin_module):
    store = [_store_item("a", "a chunk of a")]
    docs = {"doc-a": _doc_hit("a", "the whole of note a")}
    out = _search(_p(plugin_module, store, FakeLeg([], unit="document", docs=docs), {"recall": {"block_tokens": 0}}))
    assert out["results"][0]["content"] == "the whole of note a"
    out = _search(_p(plugin_module, store, FakeLeg([], unit="document", docs=docs), {"recall": {"fusion": "store"}}))
    assert "a chunk of a" in out["block"]                                   # fusion=store: exactly as before


def test_document_mode_warming_leg_answers_from_the_store_alone(plugin_module):
    store = [_store_item("a", "a chunk of a")]
    out = _search(_p(plugin_module, store, FakeLeg([], status="warming", unit="document")))
    assert out["keyword"] == "warming" and "a chunk of a" in out["block"]


def test_union_keeps_the_store_top_hits_too(plugin_module):
    store = [_store_item(n) for n in "abcdef"]
    leg = FakeLeg([_kw_hit(f"k{i}") for i in range(1, 6)])
    out = _search(_p(plugin_module, store, leg))
    names = [s["source"] for s in out["sources"]]
    assert names[:6] == ["src-k1", "src-k2", "src-k3", "src-a", "src-b", "src-c"]
    assert names[6:8] == ["src-d", "src-e"]


def test_fusion_store_is_the_store_order_only(plugin_module):
    store = [_store_item(n) for n in "abc"]
    leg = FakeLeg([_kw_hit("k1")])
    out = _search(_p(plugin_module, store, leg, {"recall": {"fusion": "store"}}))
    assert [s["source"] for s in out["sources"]] == ["src-a", "src-b", "src-c"]


def test_budget_skips_what_does_not_fit_and_never_cuts(plugin_module):
    huge = " ".join(["word"] * 3000)
    store = [_store_item("a", "short one"), _store_item("big", huge), _store_item("c", "short two")]
    out = _search(_p(plugin_module, store, FakeLeg(), {"recall": {"block_tokens": 100}}))
    assert [s["source"] for s in out["sources"]] == ["src-a", "src-c"]
    assert [o["source"] for o in out["omitted"]] == ["src-big"]
    assert huge not in out["block"] and "word word" not in out["block"]
    assert out["tokens"] <= 100


def test_nothing_fits_says_so_instead_of_cutting(plugin_module):
    out = _search(_p(plugin_module, [_store_item("a", " ".join(["w"] * 500))], FakeLeg(),
                     {"recall": {"block_tokens": 20}}))
    assert out["block"] == "" and out["entries"] == 0 and len(out["omitted"]) == 1
    assert "budget" in out["note"]


def test_warming_or_unavailable_leg_falls_back_to_the_store(plugin_module):
    store = [_store_item(n) for n in "ab"]
    for status in ("warming", "unavailable"):
        out = _search(_p(plugin_module, store, FakeLeg([_kw_hit("k1")], status=status)))
        assert out["keyword"] == status
        assert [s["source"] for s in out["sources"]] == ["src-a", "src-b"]


def test_no_leg_at_all_is_the_store_alone(plugin_module):
    out = _search(_p(plugin_module, [_store_item("a")], None))
    assert out["keyword"] == "off" and [s["source"] for s in out["sources"]] == ["src-a"]


def test_keyword_leg_only_joins_hybrid_mode(plugin_module):
    leg = FakeLeg([_kw_hit("k1")])
    p = _p(plugin_module, [_store_item("a")], leg)
    out = _search(p, mode="semantic")
    assert leg.calls == [] and out["keyword"] == "off"
    assert [s["source"] for s in out["sources"]] == ["src-a"]


def test_block_widens_the_candidate_pool_only_when_block_is_on(plugin_module):
    p = _p(plugin_module, [_store_item("a")], None)
    _search(p, limit=10)
    assert p._memory.calls[-1][1]["limit"] == 50
    p = _p(plugin_module, [_store_item("a")], None, {"recall": {"block_tokens": 0}})
    _search(p, limit=10)
    assert p._memory.calls[-1][1]["limit"] == 10


def test_block_tokens_zero_and_store_is_the_old_note_list_exactly(plugin_module):
    store = [_store_item("a", date="2026-01-01"), _store_item("b")]
    p = _p(plugin_module, store, FakeLeg([_kw_hit("k1")]), {"recall": {"fusion": "store", "block_tokens": 0}})
    out = _search(p)
    assert set(out) == {"results"}
    assert [r["source"] for r in out["results"]] == ["src-a", "src-b"]
    assert set(out["results"][0]) == {"content", "source", "scores"}
    assert out["results"][0]["content"] == "stored text of a"


def test_block_tokens_zero_with_union_lists_fused_notes(plugin_module):
    p = _p(plugin_module, [_store_item("a")], FakeLeg([_kw_hit("k1")]), {"recall": {"block_tokens": 0}})
    out = _search(p, limit=2)
    assert [r["source"] for r in out["results"]] == ["src-k1", "src-a"]


def test_dates_come_from_item_metadata_else_the_database_else_undated(plugin_module):
    p = _p(plugin_module, [_store_item("a", date="2026-05-06"), _store_item("b")], None)
    out = _search(p)
    assert "[source: src-a | date: 2026-05-06]" in out["block"]
    assert "[source: src-b | date: undated]" in out["block"]   # fake memory has no database


def test_library_floor_option_is_used_only_when_it_exists(plugin_module, monkeypatch):
    assert plugin_module._library_floor_kw() == {}              # the pinned library has none on search()
    import prospecta._index as idx

    def with_floor(memory, text, *, mode="hybrid", limit=10, floor=False, depth=None):
        raise AssertionError

    monkeypatch.setattr(idx, "search", with_floor)
    assert plugin_module._library_floor_kw() == {"floor": True}
    p = _p(plugin_module, [_store_item("a")], None)
    monkeypatch.setattr(plugin_module, "_library_floor_kw", lambda: {"floor": True})
    _search(p)
    assert p._memory.calls[-1][1]["floor"] is True


def test_store_failure_is_an_error_not_a_crash(plugin_module):
    p = _p(plugin_module, [], FakeLeg([_kw_hit("k1")]))
    p._memory.search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    assert "boom" in _search(p)["error"]


# ---- the shadow log: false-abstain rate beside every abstention figure ----

OLD, NEW = "old_bank", "new_bank"


def _rm(source, bank):
    return SimpleNamespace(content=f"c {source}", original_chunk=f"chunk {source}", source=source, score=0.5,
                           scores={}, metadata={}, bank_id=bank, document_id=f"{bank}-{source}")


class NewMemory:
    def __init__(self, results):
        self.results, self.closed = results, False

    def search(self, q, **kw):
        return list(self.results)

    def close(self):
        self.closed = True


@pytest.fixture
def origin_ids(monkeypatch):
    import prospecta._shadow as lib
    monkeypatch.setattr(lib, "_origin_ids",
                        lambda memory, bank, results: [r.document_id.split("-", 1)[1] for r in results])


def _shadow_provider(plugin_module, old, new, kw):
    p = plugin_module.ProspectaProvider()
    p._memory = FakeMemory([_rm(s, OLD) for s in old])
    p._memory.events = []
    p._memory._tracer = lambda kind, payload: p._memory.events.append(payload)
    p._keyword = kw
    p._shadow = plugin_module._ShadowReader(
        p._memory, NewMemory([_rm(s, NEW) for s in new]), bank_id=OLD, shadow_bank_id=NEW, keyword=kw)
    return p


def _drain(p):
    shadow, p._shadow = p._shadow, None
    shadow.stop()


def test_shadow_pair_carries_counts_and_the_false_abstain_rate(plugin_module, origin_ids, caplog):
    p = _shadow_provider(plugin_module, old=[], new=["a", "c"], kw=FakeLeg([_kw_hit("k1")]))
    with caplog.at_level(logging.INFO):
        p.handle_tool_call("prospecta_search", {"query": "secret words"})
        p.handle_tool_call("prospecta_search", {"query": "secret words"})
        _drain(p)
    old_row, new_row = p._memory.events[:2]
    sh = old_row["trace"]["shadow"]
    assert (sh["old_n"], sh["new_n"], sh["kw_n"]) == (0, 2, 1)
    assert sh["abstain"]["old_abstain"] == 1 and sh["abstain"]["old_false_abstain"] == 1
    assert sh["abstain"]["answerable"] == 1
    last = p._memory.events[-1]["trace"]["shadow"]["abstain"]
    assert last["pairs"] == 2 and last["old_false_abstain"] == 2
    line = [r.getMessage() for r in caplog.records if "shadow pair" in r.getMessage()][-1]
    assert "abstain old 2/2 (100.0%) false-abstain old 2/2 (100.0%)" in line
    assert "abstain new 0/2 (0.0%) false-abstain new 0/2 (0.0%)" in line
    assert "old_n=0 new_n=2 kw_n=1" in line
    assert "secret" not in line and "chunk" not in line     # counts only, no content


def test_shadow_without_a_keyword_leg_still_prints_the_rate(plugin_module, origin_ids, caplog):
    p = _shadow_provider(plugin_module, old=["a"], new=[], kw=None)
    with caplog.at_level(logging.INFO):
        p.handle_tool_call("prospecta_search", {"query": "q"})
        _drain(p)
    line = [r.getMessage() for r in caplog.records if "shadow pair" in r.getMessage()][-1]
    assert "kw_n=n/a" in line and "false-abstain new 1/1 (100.0%)" in line
    assert p._memory.events[0]["trace"]["shadow"]["kw_n"] is None


def test_shadow_compares_the_stores_list_not_the_fused_one(plugin_module, origin_ids):
    p = _shadow_provider(plugin_module, old=["a", "b"], new=["a", "c"], kw=FakeLeg([_kw_hit("k1")]))
    p.handle_tool_call("prospecta_search", {"query": "q", "limit": 5})
    _drain(p)
    assert p._memory.events[0]["trace"]["shadow"]["old_origin_document_ids"] == ["a", "b"]


def test_shadow_recall_tools_count_keyword_evidence_in_the_worker(plugin_module, origin_ids):
    p = _shadow_provider(plugin_module, old=["a"], new=[], kw=FakeLeg([_kw_hit("k1")]))
    p._llm = object()
    p._memory.recall_synth = lambda q, **kw: SimpleNamespace(
        synthesis="old answer", sources=[], queries=[SimpleNamespace(text="q1")])
    p.handle_tool_call("prospecta_recall", {"query": "q"})
    _drain(p)
    sh = p._memory.events[0]["trace"]["shadow"]
    assert sh["kw_n"] == 1 and sh["old_n"] == 0 and sh["abstain"]["old_false_abstain"] == 1


# ---- real Postgres: the leg reads the plugin's own pool ----

def _seed(p, name, text, date="2026-03-01", chunks=True):
    p._memory.retain(content=text, source=name, index_text=[f"what about {name}?"], child_chunks=chunks,
                     metadata={"created_on": date})


def _settle(leg, query="alpha"):
    """Let a due check run (background build), then wait for it."""
    leg.wait_idle(10)
    leg._checked_at = None
    leg._min_rebuild_s = 0
    leg.search(query, 1)
    leg.wait_idle(10)


def test_real_bank_keyword_leg_finds_an_exact_identifier(provider_with_stubs):
    p = provider_with_stubs
    assert p._keyword is not None
    _seed(p, "turn:1", "we shipped a release and wrote the changelog")
    _seed(p, "turn:2", "the flag deploy-config_v2.yaml controls staging")
    _seed(p, "turn:3", "unrelated chatter about lunch")
    p._keyword.wait_idle(10)
    p._keyword._checked_at = None                  # force a fingerprint look; rebuild allowed at once
    p._keyword._min_rebuild_s = 0
    status, hits = p._keyword.search("deploy-config_v2.yaml", 3)
    p._keyword.wait_idle(10)
    status, hits = p._keyword.search("deploy-config_v2.yaml", 3)
    assert status == "ok" and hits[0].source == "turn:2" and hits[0].created_on == "2026-03-01"
    out = json.loads(p.handle_tool_call("prospecta_search", {"query": "deploy-config_v2.yaml"}))
    assert out["block"].startswith("[source: turn:2 | date: 2026-03-01]")   # keyword top hit first
    assert out["sources"][0]["source"] == "turn:2"


def test_real_bank_without_chunk_items_still_has_a_working_keyword_leg(provider_with_stubs):
    """The live case: question items only, no chunk items. The leg indexes the notes."""
    p = provider_with_stubs
    _seed(p, "turn:1", "we shipped a release and wrote the changelog", chunks=False)
    _seed(p, "turn:2", "the flag deploy-config_v2.yaml controls staging", chunks=False)
    with p._memory._pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM memory_items WHERE kind = 'chunk'").fetchone()[0] == 0
    leg = p._keyword
    assert leg.unit == "document"
    _settle(leg, "deploy-config_v2.yaml")
    status, hits = leg.search("deploy-config_v2.yaml", 3)
    assert status == "ok" and hits[0].source == "turn:2" and hits[0].created_on == "2026-03-01"
    assert hits[0].original_chunk == "the flag deploy-config_v2.yaml controls staging" and hits[0].item_id == ""
    out = json.loads(p.handle_tool_call("prospecta_search", {"query": "deploy-config_v2.yaml"}))
    assert out["block"].startswith("[source: turn:2 | date: 2026-03-01]")
    chunk_leg = rf.KeywordLeg(rf.PgChunkSource(p._memory._pool, "test"), bank_id="test", background=False)
    assert chunk_leg.search("deploy-config_v2.yaml", 3) == ("ok", [])      # the old unit finds nothing here


def test_real_bank_leg_appends_new_notes_without_a_full_read(provider_with_stubs):
    p = provider_with_stubs
    _seed(p, "turn:1", "alpha one", chunks=False)
    leg = p._keyword
    _settle(leg)
    assert leg.last_build["mode"] == "full" and leg.last_build["rows"] == 1
    _seed(p, "turn:2", "alpha two with zebrafish", chunks=False)
    _settle(leg, "zebrafish")
    assert leg.last_build["mode"] == "append" and leg.last_build["added"] == 1 and leg.last_build["rows"] == 2
    assert [h.source for h in leg.search("zebrafish", 3)[1]] == ["turn:2"]
    assert len(leg.search("alpha", 10)[1]) == 2                       # no duplicate of the first note
    with p._memory._pool.connection() as conn:                        # an in-place edit: full rebuild
        conn.execute("UPDATE documents SET original_text = 'alpha edited', updated_at = now() "
                     "WHERE source = 'turn:1'")
        conn.commit()
    _settle(leg, "edited")
    assert leg.last_build["mode"] == "full" and [h.source for h in leg.search("edited", 3)[1]] == ["turn:1"]


def test_real_bank_chunk_unit_is_still_selectable(plugin_module, pg_url, hermes_home, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    (hermes_home / "prospecta.json").write_text(json.dumps(
        {"bank_id": "t", "embedding_dim": "32", "recall": {"keyword_unit": "chunks"}}))
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: stub_llm)
    p.initialize("s", hermes_home=str(hermes_home))
    try:
        assert p._keyword.unit == "chunk"
        _seed(p, "turn:1", "the flag deploy-config_v2.yaml controls staging", chunks=True)
        _settle(p._keyword, "deploy-config_v2.yaml")
        status, hits = p._keyword.search("deploy-config_v2.yaml", 3)
        assert status == "ok" and hits[0].item_id and hits[0].source == "turn:1"
    finally:
        p.shutdown()


def test_real_bank_leg_uses_the_bounded_pool_and_stops_with_the_provider(provider_with_stubs):
    p = provider_with_stubs
    _seed(p, "turn:1", "alpha beta gamma")
    p._keyword.wait_idle(10)
    pool = p._memory._pool
    p._keyword._checked_at = None
    p._keyword._min_rebuild_s = 0
    p._keyword.search("alpha", 1)
    p._keyword.wait_idle(10)
    assert len(pool._open) <= pool._max
    leg = p._keyword
    p.shutdown()
    assert leg.search("alpha", 1) == ("unavailable", [])
    assert p._keyword is None


@pytest.mark.parametrize("cls,chunks", [("PgDocumentSource", False), ("PgChunkSource", True)])
def test_real_bank_source_fingerprint_and_keyset_batches(provider_with_stubs, cls, chunks):
    p = provider_with_stubs
    pool = p._memory._pool
    src = getattr(rf, cls)(pool, "test", batch=2)
    assert src.fingerprint()["count"] == 0
    for i in range(5):
        _seed(p, f"turn:{i}", f"note number {i}", chunks=chunks)
    fp = src.fingerprint()
    assert fp["count"] == 5
    pages = list(src.batches())
    assert [len(b) for b in pages] == [2, 2, 1]                       # keyset pages, 2 at a time
    rows = [r for b in pages for r in b]
    assert [r["source"] for r in rows] == [f"turn:{i}" for i in range(5)]
    cursor = pages[1][-1]["cursor"]
    rest = [r for b in src.batches(after=cursor) for r in b]
    assert [r["source"] for r in rest] == ["turn:4"]
    fp2 = src.fingerprint(cursor)
    assert fp2["prefix_count"] == 4 and fp2["prefix_max_updated"] == max(r["updated_at"] for r in rows[:4])
    _seed(p, "turn:9", "later", chunks=chunks)
    fp3 = src.fingerprint(cursor)
    assert fp3["count"] == 6 and fp3["prefix_count"] == 4 and fp3["max_created"] > fp["max_created"]
    assert len(pool._open) <= pool._max


def test_recall_keyword_off_builds_no_leg(plugin_module, pg_url, hermes_home, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    (hermes_home / "prospecta.json").write_text(json.dumps(
        {"bank_id": "t", "embedding_dim": "32", "recall": {"keyword": False}}))
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p, "_build_llm", lambda: stub_llm)
    p.initialize("s", hermes_home=str(hermes_home))
    try:
        assert p._keyword is None
    finally:
        p.shutdown()
