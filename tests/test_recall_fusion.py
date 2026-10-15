"""Keyword leg, keyword-first union, token-budget block, abstention tally.

Everything here is offline and uses invented fixtures only.
"""
from __future__ import annotations

import threading
import time

import pytest

import recall_fusion as rf


# ---- tokenizer ----

def test_tokens_are_lowercase_words():
    assert rf.tokenize("The Quick, brown fox!") == ["the", "quick", "brown", "fox"]


def test_identifiers_keep_whole_and_parts():
    toks = rf.tokenize("see deploy-config_v2.yaml now")
    assert "deploy-config_v2.yaml" in toks
    for part in ("deploy", "config", "v2", "yaml"):
        assert part in toks
    # ordinary sentence punctuation does not glue words together
    assert rf.tokenize("done. next") == ["done", "next"]
    assert "done." not in rf.tokenize("done. next")


def test_underscored_and_hyphenated_identifiers():
    toks = rf.tokenize("run my_tool and step-3")
    assert {"my_tool", "my", "tool", "step-3", "step", "3"} <= set(toks)


def test_no_stemming():
    assert rf.tokenize("running runs") == ["running", "runs"]


# ---- BM25 index ----

def _row(i, text, doc=None, source=None, date="2026-01-01"):
    return {"item_id": f"i{i}", "document_id": doc or f"d{i}", "source": source or f"s{i}",
            "created_on": date, "text": text}


def _index(rows):
    idx = rf.Bm25Index(bank_id="b")
    idx.add_rows(rows)
    return idx


def test_bm25_ranks_rare_token_over_common():
    idx = _index([
        _row(1, "alpha beta gamma"), _row(2, "alpha beta delta"),
        _row(3, "alpha zephyr beta"), _row(4, "alpha beta beta beta"),
    ])
    hits = idx.search("zephyr beta", 4)
    assert hits[0].source == "s3"


def test_bm25_or_semantics_and_no_match_is_empty():
    idx = _index([_row(1, "red apple"), _row(2, "green pear")])
    assert {h.source for h in idx.search("apple pear", 5)} == {"s1", "s2"}
    assert idx.search("banana", 5) == []
    assert idx.search("", 5) == []


def test_bm25_exact_identifier_wins():
    idx = _index([_row(1, "notes about the build"), _row(2, "edit deploy-config_v2.yaml today"),
                  _row(3, "deploy a config and build notes")])
    assert idx.search("deploy-config_v2.yaml", 3)[0].source == "s2"


def test_bm25_stemming_off_distinguishes_forms():
    idx = _index([_row(1, "she was running"), _row(2, "he runs daily")])
    assert [h.source for h in idx.search("running", 5)] == ["s1"]


def test_bm25_hit_carries_text_source_date_and_ids():
    idx = _index([_row(7, "hello world", doc="doc-7", source="turn:7", date="2026-03-04")])
    h = idx.search("hello", 1)[0]
    assert (h.original_chunk, h.source, h.created_on, h.document_id, h.bank_id) == (
        "hello world", "turn:7", "2026-03-04", "doc-7", "b")
    assert h.content == "hello world" and h.scores["bm25"] == h.score > 0
    assert h.origin == "keyword"


def test_bm25_respects_limit_and_is_deterministic_on_ties():
    idx = _index([_row(i, "same words here") for i in range(6)])
    first = [h.source for h in idx.search("same", 3)]
    assert first == [h.source for h in idx.search("same", 3)]
    assert len(first) == 3


# ---- keyword-first union ----

class _It:
    def __init__(self, doc, chunk=None, origin="store"):
        self.document_id, self.original_chunk, self.content, self.origin = doc, chunk or doc, chunk or doc, origin

    def __repr__(self):
        return f"It({self.document_id})"


def _ids(items):
    return [i.document_id for i in items]


def test_union_takes_top3_of_each_then_fills_in_store_order():
    kw = [_It(f"k{i}", origin="keyword") for i in range(1, 7)]
    store = [_It(f"s{i}") for i in range(1, 9)]
    fused = rf.keyword_first_union(kw, store, slots=10)
    assert _ids(fused) == ["k1", "k2", "k3", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]


def test_union_dedupes_and_never_demotes_keyword_top_hit():
    kw = [_It("a"), _It("b"), _It("c"), _It("d")]
    store = [_It("x"), _It("a"), _It("y"), _It("z"), _It("b")]
    fused = rf.keyword_first_union(kw, store, slots=6)
    assert _ids(fused)[:3] == ["a", "b", "c"]
    assert _ids(fused) == ["a", "b", "c", "x", "y", "z"]
    assert len(set(_ids(fused))) == len(fused)


def test_union_same_note_different_chunk_is_not_a_duplicate():
    kw = [_It("d1", "chunk one")]
    store = [_It("d1", "chunk two"), _It("d1", "chunk one")]
    fused = rf.keyword_first_union(kw, store, slots=5)
    assert [i.original_chunk for i in fused] == ["chunk one", "chunk two"]


def test_union_fills_from_keyword_remainder_when_store_runs_out():
    kw = [_It(f"k{i}") for i in range(1, 8)]
    store = [_It("s1")]
    assert _ids(rf.keyword_first_union(kw, store, slots=6)) == ["k1", "k2", "k3", "s1", "k4", "k5"]


def test_union_with_empty_keyword_leg_is_the_store_list():
    store = [_It(f"s{i}") for i in range(5)]
    assert _ids(rf.keyword_first_union([], store, slots=3)) == ["s0", "s1", "s2"]


def test_union_slots_cap_even_the_head():
    kw = [_It(f"k{i}") for i in range(5)]
    store = [_It(f"s{i}") for i in range(5)]
    assert len(rf.keyword_first_union(kw, store, slots=4)) == 4


# ---- token budget block ----

def test_estimate_tokens_is_whitespace_based_with_a_character_floor():
    assert rf.estimate_tokens("") == 0
    assert rf.estimate_tokens("one two three") == 4          # ceil(3 * 1.3)
    assert rf.estimate_tokens("x" * 400) == 100              # character floor (len/4)


def _entry(i, words, date="2026-01-0%d"):
    return rf.BlockEntry(source=f"src{i}", date=(date % i) if "%d" in date else date,
                         text=" ".join(f"w{i}x{j}" for j in range(words)), origin="store")


def test_block_is_one_text_tagged_with_source_and_date_in_order():
    r = rf.build_block([_entry(1, 5), _entry(2, 5)], budget=1000)
    assert isinstance(r.text, str)
    assert r.text.index("src1") < r.text.index("src2")
    assert "[source: src1 | date: 2026-01-01]" in r.text and "[source: src2 | date: 2026-01-02]" in r.text
    assert [e.source for e in r.included] == ["src1", "src2"] and r.omitted == []
    assert 0 < r.tokens <= 1000


def test_block_skips_an_entry_that_does_not_fit_and_takes_a_later_one():
    big, small = _entry(1, 400), _entry(2, 5)
    r = rf.build_block([_entry(0, 20), big, small], budget=100)
    assert [e.source for e in r.included] == ["src0", "src2"]
    assert [e.source for e in r.omitted] == ["src1"]
    assert r.tokens <= 100


def test_block_never_cuts_stored_text():
    e = _entry(1, 60)
    r = rf.build_block([e, _entry(2, 3)], budget=60)
    assert e.text not in r.text and "w1x0" not in r.text   # folded away whole, not cut
    assert all(e.text in r.text for e in r.included)


def test_block_nothing_fits_is_empty_not_truncated():
    r = rf.build_block([_entry(1, 500)], budget=10)
    assert r.text == "" and r.included == [] and len(r.omitted) == 1


def test_block_tokens_never_exceed_budget():
    entries = [_entry(i, 30 + 7 * i) for i in range(1, 9)]
    for budget in (50, 120, 333, 1500):
        r = rf.build_block(entries, budget)
        assert r.tokens <= budget
        assert rf.estimate_tokens(r.text) <= budget


# ---- abstention tally ----

def test_tally_false_abstain_rate_sits_beside_abstention():
    t = rf.AbstentionTally()
    t.record(old_found=True, new_found=True, kw_found=True)
    t.record(old_found=False, new_found=True, kw_found=False)    # old abstained, other side found: false abstain
    t.record(old_found=False, new_found=False, kw_found=False)   # abstained, nothing anywhere: not answerable
    t.record(old_found=True, new_found=False, kw_found=False)    # new abstained, old found: false abstain (new)
    s = t.snapshot()
    assert s == {"pairs": 4, "answerable": 3,
                 "old_abstain": 2, "old_false_abstain": 1,
                 "new_pairs": 4, "new_abstain": 2, "new_false_abstain": 1}
    line = rf.format_tally(s)
    assert "abstain old 2/4 (50.0%) false-abstain old 1/3 (33.3%)" in line
    assert "abstain new 2/4 (50.0%) false-abstain new 1/3 (33.3%)" in line


def test_tally_keyword_evidence_counts_as_answerable():
    t = rf.AbstentionTally()
    t.record(old_found=False, new_found=False, kw_found=True)
    s = t.snapshot()
    assert s["answerable"] == 1 and s["old_false_abstain"] == 1 and s["new_false_abstain"] == 1


def test_tally_new_side_error_is_not_counted():
    t = rf.AbstentionTally()
    t.record(old_found=False, new_found=None, kw_found=True)
    s = t.snapshot()
    assert s["new_pairs"] == 0 and s["new_abstain"] == 0 and s["old_false_abstain"] == 1
    assert "abstain new n/a" in rf.format_tally(s)


def test_tally_empty_rates_are_na_not_an_error():
    line = rf.format_tally(rf.AbstentionTally().snapshot())
    assert "n/a" in line


# ---- the keyword leg: in-memory index, fingerprint rebuild, never blocks ----

class FakeSource:
    def __init__(self, rows):
        self.rows = list(rows)
        self.fp_calls = 0
        self.batch_calls = 0
        self.fail = None
        self.gate = None

    def fingerprint(self):
        self.fp_calls += 1
        if self.fail:
            raise self.fail
        return (len(self.rows), len(self.rows))

    def batches(self):
        self.batch_calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        yield self.rows[:2]
        yield self.rows[2:]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _leg(src, clock=None, **kw):
    kw.setdefault("background", False)
    return rf.KeywordLeg(src, bank_id="b", clock=clock or Clock(), **kw)


def test_leg_builds_on_first_use_then_serves_from_memory():
    src = FakeSource([_row(i, f"token{i} common") for i in range(5)])
    leg = _leg(src)
    status, hits = leg.search("token3", 3)
    assert status == "ok" and hits[0].source == "s3"
    leg.search("token1", 3)
    assert src.batch_calls == 1   # one scan, then memory only


def test_leg_rebuilds_when_fingerprint_changes_after_the_min_interval():
    clock = Clock()
    src = FakeSource([_row(i, f"token{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=60)
    leg.search("token0", 3)
    src.rows.append(_row(9, "freshword"))
    clock.t += 31                                  # due for a check, but too soon to rebuild
    assert leg.search("freshword", 3)[1] == []
    clock.t += 61
    leg.search("freshword", 3)                     # check -> rebuild (synchronous in this test)
    assert [h.source for h in leg.search("freshword", 3)[1]] == ["s9"]
    assert src.batch_calls == 2


def test_leg_unchanged_fingerprint_does_not_rebuild():
    clock = Clock()
    src = FakeSource([_row(i, f"t{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=0)
    leg.search("t1", 1)
    clock.t += 100
    leg.search("t1", 1)
    assert src.batch_calls == 1 and src.fp_calls >= 2


def test_leg_background_build_never_blocks_the_caller():
    src = FakeSource([_row(i, f"t{i}") for i in range(4)])
    src.gate = threading.Event()
    leg = rf.KeywordLeg(src, bank_id="b", background=True)
    t0 = time.monotonic()
    assert leg.search("t1", 2) == ("warming", [])
    assert time.monotonic() - t0 < 0.5
    src.gate.set()
    leg.wait_idle(5)
    status, hits = leg.search("t1", 2)
    assert status == "ok" and hits[0].source == "s1"
    leg.stop()


def test_leg_source_failure_is_contained_and_logged(caplog):
    src = FakeSource([_row(1, "a")])
    src.fail = RuntimeError("db down: details " * 3)
    leg = _leg(src)
    with caplog.at_level("ERROR"):
        status, hits = leg.search("a", 3)
    assert (status, hits) == ("unavailable", [])
    assert "db down" in caplog.text


def test_leg_keeps_serving_the_old_index_when_a_recheck_fails():
    clock = Clock()
    src = FakeSource([_row(i, f"t{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=0)
    leg.search("t1", 1)
    src.fail = RuntimeError("blip")
    clock.t += 31
    status, hits = leg.search("t1", 1)
    assert status == "ok" and hits[0].source == "s1"


def test_leg_refuses_a_bank_over_the_chunk_cap(caplog):
    class Big(FakeSource):
        def fingerprint(self):
            return (10_000_000, 0)
    leg = _leg(Big([_row(1, "a")]), max_chunks=1000)
    with caplog.at_level("ERROR"):
        assert leg.search("a", 1) == ("unavailable", [])
    assert "1000" in caplog.text
    assert leg.search("a", 1)[0] == "unavailable"


def test_leg_stop_prevents_new_work():
    src = FakeSource([_row(1, "a")])
    leg = _leg(src)
    leg.stop()
    assert leg.search("a", 1) == ("unavailable", [])
    assert src.batch_calls == 0
