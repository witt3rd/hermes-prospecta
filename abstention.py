"""Abstention counts for the shadow log, always printed with the false-abstain rate.

Counts only (no question or note text). Used by the shadow reader (``__init__._ShadowReader``):
a side *abstains* on a question when it returned nothing; the question is *answerable* when
the old side or the new side found something; a *false abstain* is an abstention on an answerable
question. (The library's lexical-only search, once a third leg of this evidence, is gone: it has one
retrieval path.)
"""
from __future__ import annotations

import threading
from typing import Dict, Optional


class AbstentionTally:
    """Running counts of a shadow comparison.

    A side *abstains* on a question when it returned nothing. A question is
    *answerable* when the old side or the new side found something. A *false abstain* is an abstention on an
    answerable question.
    ``new_found=None`` (the new side errored) is left out of the new-side counts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c = {"pairs": 0, "answerable": 0, "old_abstain": 0, "old_false_abstain": 0,
                   "new_pairs": 0, "new_abstain": 0, "new_false_abstain": 0}

    def record(self, *, old_found: bool, new_found: Optional[bool]) -> Dict[str, int]:
        answerable = bool(old_found or new_found)
        with self._lock:
            c = self._c
            c["pairs"] += 1
            c["answerable"] += int(answerable)
            if not old_found:
                c["old_abstain"] += 1
                c["old_false_abstain"] += int(answerable)
            if new_found is not None:
                c["new_pairs"] += 1
                if not new_found:
                    c["new_abstain"] += 1
                    c["new_false_abstain"] += int(answerable)
            return dict(c)

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._c)


def _rate(n: int, d: int) -> str:
    return f"{n}/{d} ({100.0 * n / d:.1f}%)" if d else "n/a"


def format_tally(s: Dict[str, int]) -> str:
    """Counts only: abstention and, right beside it, the false-abstain rate."""
    new = (f"abstain new {_rate(s['new_abstain'], s['new_pairs'])} "
           f"false-abstain new {_rate(s['new_false_abstain'], s['answerable']) if s['new_pairs'] else 'n/a'}")
    old = (f"abstain old {_rate(s['old_abstain'], s['pairs'])} "
           f"false-abstain old {_rate(s['old_false_abstain'], s['answerable'])}")
    return f"{old}; {new}"
