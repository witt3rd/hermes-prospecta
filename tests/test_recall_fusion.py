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

def _row(i, text, doc=None, source=None, date="2026-01-01", updated=None):
    created = f"2026-01-01 00:00:{i:02d}.000000"
    return {"item_id": f"i{i}", "document_id": doc or f"d{i}", "source": source or f"s{i}",
            "created_on": date, "text": text, "cursor": (created, doc or f"d{i}"),
            "updated_at": updated or created}


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
    """The source protocol: fingerprint(cursor) and batches(after), rows ordered by cursor."""

    unit = "document"

    def __init__(self, rows):
        self.rows = list(rows)
        self.fp_calls = 0
        self.batch_calls = []          # the ``after`` of each scan
        self.rows_read = 0
        self.fail = None
        self.gate = None

    def fingerprint(self, cursor=None):
        self.fp_calls += 1
        if self.fail:
            raise self.fail
        rows = sorted(self.rows, key=lambda r: r["cursor"])
        fp = {"count": len(rows), "max_created": max((r["cursor"][0] for r in rows), default=""),
              "max_updated": max((r["updated_at"] for r in rows), default="")}
        if cursor is not None:
            pre = [r for r in rows if r["cursor"] <= tuple(cursor)]
            fp["prefix_count"] = len(pre)
            fp["prefix_max_updated"] = max((r["updated_at"] for r in pre), default="")
        return fp

    def batches(self, after=None):
        self.batch_calls.append(after)
        if self.gate is not None:
            self.gate.wait(5)
        rows = sorted(self.rows, key=lambda r: r["cursor"])
        rows = [r for r in rows if after is None or r["cursor"] > tuple(after)]
        for lo in range(0, len(rows), 2):
            self.rows_read += len(rows[lo:lo + 2])
            yield rows[lo:lo + 2]


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
    assert len(src.batch_calls) == 1   # one scan, then memory only


def test_leg_appends_new_rows_only_and_is_not_held_back_by_the_rebuild_interval():
    clock = Clock()
    src = FakeSource([_row(i, f"token{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=600)
    leg.search("token0", 3)
    assert src.rows_read == 3
    src.rows.append(_row(9, "freshword"))
    clock.t += 31                                  # due for a check: appended at once, not a full read
    leg.search("freshword", 3)
    assert [h.source for h in leg.search("freshword", 3)[1]] == ["s9"]
    assert src.rows_read == 4 and src.batch_calls[-1] == ("2026-01-01 00:00:02.000000", "d2")
    assert leg.last_build["mode"] == "append" and leg.last_build["rows"] == 4
    assert [h.source for h in leg.search("token1", 3)[1]] == ["s1"]   # old rows still served


def test_leg_rebuilds_in_full_when_indexed_rows_changed_and_the_interval_has_passed():
    clock = Clock()
    src = FakeSource([_row(i, f"token{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=60)
    leg.search("token0", 3)
    src.rows[1] = _row(1, "rewritten", updated="2026-02-01 00:00:00.000000")    # edited in place
    clock.t += 31                                  # due for a check, but too soon to rebuild
    assert leg.search("rewritten", 3)[1] == []
    clock.t += 61
    leg.search("rewritten", 3)
    assert [h.source for h in leg.search("rewritten", 3)[1]] == ["s1"]
    assert leg.search("token1", 3)[1] == []
    assert src.batch_calls[-1] is None and src.rows_read == 6 and leg.last_build["mode"] == "full"


def test_leg_rebuilds_in_full_after_a_delete_or_a_late_row_behind_the_cursor():
    clock = Clock()
    src = FakeSource([_row(i, f"token{i}") for i in range(4)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=0)
    leg.search("token0", 3)
    del src.rows[1]                                # a note was deleted
    clock.t += 31
    leg.search("token0", 3)
    assert leg.search("token1", 3)[1] == [] and leg.last_build["mode"] == "full"
    src.rows.append(_row(0, "late", doc="dz"))     # committed late, ordered before the cursor
    src.rows[-1]["cursor"] = ("2026-01-01 00:00:00.500000", "dz")
    clock.t += 31
    leg.search("token0", 3)
    assert [h.document_id for h in leg.search("late", 3)[1]] == ["dz"] and leg.last_build["mode"] == "full"


def test_leg_appending_never_duplicates_a_row():
    clock = Clock()
    src = FakeSource([_row(i, f"t{i} shared") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=0)
    leg.search("t0", 1)
    for k in range(3, 6):
        src.rows.append(_row(k, f"t{k} shared"))
        clock.t += 31
        leg.search("t0", 1)
    assert len(leg.search("shared", 50)[1]) == 6


def test_leg_unchanged_fingerprint_does_not_rebuild():
    clock = Clock()
    src = FakeSource([_row(i, f"t{i}") for i in range(3)])
    leg = _leg(src, clock, recheck_s=30, min_rebuild_s=0)
    leg.search("t1", 1)
    clock.t += 100
    leg.search("t1", 1)
    assert len(src.batch_calls) == 1 and src.fp_calls >= 2


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


def test_leg_stop_prevents_new_work():
    src = FakeSource([_row(1, "a")])
    leg = _leg(src)
    leg.stop()
    assert leg.search("a", 1) == ("unavailable", [])
    assert src.batch_calls == []


def test_leg_has_no_size_cap_it_builds_whatever_the_bank_holds():
    class Huge(FakeSource):
        def fingerprint(self, cursor=None):
            fp = super().fingerprint(cursor)
            fp["count"] = 50_000_000                 # a bank far past the old 400,000 limit
            return fp
    leg = _leg(Huge([_row(1, "alpha")]))
    assert leg.search("alpha", 1)[0] == "ok"


def test_leg_logs_counts_size_and_time_and_warns_over_the_soft_threshold(caplog):
    src = FakeSource([_row(i, "alpha beta " * 50) for i in range(6)])
    leg = _leg(src, warn_mb=0.0001)
    with caplog.at_level("INFO"):
        leg.search("alpha", 1)
    info = [r for r in caplog.records if r.levelname == "INFO" and "keyword leg" in r.message]
    assert info and "6 documents" in info[0].getMessage() and "bank b" in info[0].getMessage()
    assert " MB" in info[0].getMessage() and "in " in info[0].getMessage()
    warn = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warn) == 1 and "soft threshold" in warn[0].getMessage()
    assert leg.search("alpha", 1)[0] == "ok"        # warned, never turned off
    assert "alpha beta" not in caplog.text          # counts only


def test_leg_no_warning_when_the_threshold_is_off(caplog):
    leg = _leg(FakeSource([_row(i, "alpha " * 100) for i in range(5)]))
    with caplog.at_level("WARNING"):
        leg.search("alpha", 1)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_leg_pauses_after_every_database_batch(monkeypatch):
    sleeps = []
    monkeypatch.setattr(rf.time, "sleep", lambda s: sleeps.append(s))
    leg = _leg(FakeSource([_row(i, f"t{i}") for i in range(5)]), pause_s=0.2)
    leg.search("t1", 1)
    assert sleeps.count(0.2) == 3                  # 5 rows in batches of 2, 2, 1
    assert sleeps.count(0) >= 3                    # and a yield after every slice


def test_leg_never_runs_the_build_in_the_callers_connection_scope():
    """Rows are indexed after the source has handed the batch back: the source's
    generator is suspended between batches, so a pooled connection is not held."""
    order = []

    class Spy(FakeSource):
        def batches(self, after=None):
            for b in super().batches(after):
                order.append("fetch")
                yield b
                order.append("resumed")

    leg = _leg(Spy([_row(i, f"t{i}") for i in range(4)]))
    leg.search("t1", 1)
    assert order == ["fetch", "resumed", "fetch", "resumed"]


def test_bm25_search_while_rows_are_still_being_added():
    idx = _index([_row(i, f"w{i} common") for i in range(5)])
    hits = idx.search("common", 10)
    idx.add_rows([_row(7, "w7 common")])
    assert len(idx.search("common", 10)) == len(hits) + 1 and len(idx) == 6


def test_bm25_documents_lookup_and_stats():
    idx = _index([_row(1, "alpha one"), _row(2, "beta two")])
    docs = idx.documents(["d2", "nope", "d2"])
    assert list(docs) == ["d2"] and docs["d2"].content == "beta two" and docs["d2"].origin == "store"
    st = idx.stats()
    assert st["rows"] == 2 and st["terms"] == 4 and st["postings"] == 4 and st["approx_bytes"] > 0


def test_union_by_document_counts_a_note_found_by_both_legs_once():
    kw = [_It("dA", "whole note A"), _It("dB", "whole note B")]
    st = [_It("dA", "part of note A"), _It("dC", "note C")]
    assert [i.document_id for i in rf.keyword_first_union(kw, st, slots=10, by_document=True)] == ["dA", "dB", "dC"]
    # by chunk (the old key) the two texts of note A are two entries
    assert len(rf.keyword_first_union(kw, st, slots=10)) == 4


def test_with_document_text_swaps_in_the_whole_note_and_keeps_unindexed_items():
    doc = rf.KeywordHit(content="the whole note", original_chunk="the whole note", source="S", score=0.0,
                        scores={}, metadata={}, bank_id="b", document_id="dA", created_on="2026-01-02", origin="store")
    items = [_It("dA", "a part"), _It("dNew", "new note")]
    out = rf.with_document_text(items, {"dA": doc})
    assert out[0].original_chunk == "the whole note" and out[0].created_on == "2026-01-02"
    assert out[1] is items[1]


def test_with_document_text_never_cuts_a_note_too_long_for_the_budget():
    long_doc = rf.KeywordHit(content="word " * 5000, original_chunk="word " * 5000, source="S", score=0.0,
                             scores={}, metadata={}, bank_id="b", document_id="dA", origin="store")
    item = _It("dA", "short part")
    assert rf.with_document_text([item], {"dA": long_doc}, 1500)[0] is item
    assert rf.with_document_text([item], {"dA": long_doc}, 0)[0].original_chunk == long_doc.content
