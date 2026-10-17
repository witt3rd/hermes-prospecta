"""The shadow log's abstention counts: always printed with the false-abstain rate."""
from __future__ import annotations

import abstention as ab


def test_tally_false_abstain_rate_sits_beside_abstention():
    t = ab.AbstentionTally()
    t.record(old_found=True, new_found=True)
    t.record(old_found=False, new_found=True)    # old abstained, the other side found: false abstain
    t.record(old_found=False, new_found=False)   # abstained, nothing on either side: not answerable
    t.record(old_found=True, new_found=False)    # new abstained, old found: false abstain (new)
    s = t.snapshot()
    assert s == {"pairs": 4, "answerable": 3,
                 "old_abstain": 2, "old_false_abstain": 1,
                 "new_pairs": 4, "new_abstain": 2, "new_false_abstain": 1}
    line = ab.format_tally(s)
    assert "abstain old 2/4 (50.0%) false-abstain old 1/3 (33.3%)" in line
    assert "abstain new 2/4 (50.0%) false-abstain new 1/3 (33.3%)" in line


def test_tally_has_no_lexical_leg_any_more():
    """The library's lexical-only search is gone (one retrieval path), so a question is answerable only
    when the old or the new side found something."""
    import inspect
    assert "kw_found" not in inspect.signature(ab.AbstentionTally.record).parameters


def test_tally_new_side_error_is_not_counted():
    t = ab.AbstentionTally()
    t.record(old_found=False, new_found=None)
    t.record(old_found=True, new_found=None)
    s = t.snapshot()
    assert s["new_pairs"] == 0 and s["new_abstain"] == 0 and s["old_false_abstain"] == 0
    assert s["pairs"] == 2 and s["answerable"] == 1 and s["old_abstain"] == 1
    assert "abstain new n/a" in ab.format_tally(s)


def test_tally_empty_rates_are_na_not_an_error():
    line = ab.format_tally(ab.AbstentionTally().snapshot())
    assert "n/a" in line
