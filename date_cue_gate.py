"""Skip the library's date/people extraction model call when the question has no date cue.

``prospecta._index._search_channels`` calls ``extract_filters`` (one model call,
about 2 s) on every hybrid search. A question with no date cue cannot name a
period, so the call buys nothing but people matching, and that part is a cheap
regex over the bank's own people vocabulary. This module wraps the symbol the
library's ``_index`` module uses: a question with a cue goes to the library
unchanged; one without gets the people regex and no model call.
"""
from __future__ import annotations

import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

_MONTHS = ("january february march april may june july august september "
           "october november december").split()
_MONTH_ABBR = "jan feb mar apr jun jul aug sep sept oct nov dec".split()
_WEEKDAYS = ("monday tuesday wednesday thursday friday saturday sunday "
             "mon tue tues wed thu thur thurs fri sat sun").split()
_SEASONS = "spring summer autumn fall winter".split()
_UNAMBIGUOUS_MONTHS = [m for m in _MONTHS if m not in ("march", "may")]

_PREP = r"(?:in|since|during|until|till|before|after|on|of|from|by|early|late|mid|end of|start of)"
_UNITS = r"(?:week|month|year|quarter|season|weekend|morning|evening|night|afternoon|decade)"
_WD = "|".join(_WEEKDAYS)
_MON_ALL = "|".join(_UNAMBIGUOUS_MONTHS + [m for m in _MONTH_ABBR if m != "mar"])
_DAYNUM = r"(?:[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?:19|20)\d{2}"

_PATTERNS = [
    # unambiguous month names / abbreviations
    rf"\b(?:{_MON_ALL})\b",
    # may / march / mar as months: a year or day number, or a preposition before
    rf"\b(?:may|march|mar)\.?\s+(?:{_DAYNUM}\b|{_YEAR}\b)",
    rf"\b{_DAYNUM}\s+(?:of\s+)?(?:may|march|mar)\b",
    rf"\b{_PREP}\s+(?:the\s+)?(?:may|march)\b",
    # weekday names (full; short forms only unambiguous ones)
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tues|thurs?)\b",
    r"\b(?:mon|wed|fri|sat|sun)\.?(?=\s*(?:morning|evening|night|afternoon|\d))",
    # relative words
    r"\bago\b", r"\byesterday\b", r"\btoday\b", r"\btomorrow\b", r"\btonight\b",
    r"\brecently\b", r"\blately\b", r"\bearlier\s+(?:this|today)\b",
    rf"\b(?:this|last|next|previous|past|coming)\s+(?:{_UNITS}|{_WD})\b",
    r"\b(?:last|past|previous|next|coming)\s+(?:\d+\s+|(?:few|couple\s+of|two|three|four|five|six|seven|eight|nine|ten|several)\s+)?(?:weeks?|days?|months?|years?)\b",
    r"\b(?:weeks?|days?|months?|years?)\s+(?:ago|back)\b",
    # quarters, seasons
    r"\bq[1-4]\b", r"\b(?:first|second|third|fourth)\s+quarter\b",
    rf"\b(?:{'|'.join(s for s in _SEASONS if s != 'fall')})\b",
    r"\b(?:this|last|next|early|late|mid|in\s+the|in)\s+fall\b",
    # ISO / numeric dates, ordinals
    r"\b\d{4}-\d{1,2}(?:-\d{1,2})?\b",
    r"(?<![\w.:/])(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])(?:/(?:\d{2}|\d{4}))?(?![\w:/])",
    r"\b\d{1,2}(?:st|nd|rd|th)\b",
    # years
    rf"(?<![\w.:/#-]){_YEAR}(?![\w:/-])",
    # since / between ... and
    r"\bsince\b",
    r"\bbetween\b[^.?!]*?\b(?:\d{1,4})\b[^.?!]*?\band\b",
    r"\bbetween\b[^.?!]*\band\b[^.?!]*?\d",
]
_CUE_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


def has_date_cue(text: str) -> bool:
    """True when the question carries any cue that it may name a period."""
    return bool(text) and _CUE_RE.search(text) is not None


def _people(text: str, people_vocab) -> list:
    # identical to the people part of prospecta.channels.extract.regex_filters
    return [p for p in people_vocab
            if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, re.IGNORECASE)]


DEFAULT_TIMEOUT = 6.0
MAX_INFLIGHT = 8  # abandoned + running model calls allowed at once
_timeout = DEFAULT_TIMEOUT
_slots = threading.BoundedSemaphore(MAX_INFLIGHT)
_stats = {"timeouts": 0, "errors": 0, "skipped": 0}


def set_timeout(seconds) -> None:
    """Wall-clock cap (s) on the model call; 0 or less disables."""
    global _timeout
    _timeout = float(seconds)


def _call_with_timeout(fn, timeout):
    """Run ``fn`` in a daemon thread; return (done, value). Never raises.
    The thread cannot be cancelled: on timeout it is abandoned, its result and
    any exception are dropped, and it holds one of MAX_INFLIGHT slots until it
    ends. With no free slot the call is not started (done=False)."""
    if not _slots.acquire(blocking=False):
        _stats["skipped"] += 1
        return False, "no free slot (%d stalled calls)" % MAX_INFLIGHT
    box = {}
    finished = threading.Event()

    def run():
        try:
            box["v"] = fn()
        except BaseException as e:  # noqa: BLE001 - reported to the caller or dropped
            box["e"] = e
        finally:
            _slots.release()
            finished.set()

    try:
        threading.Thread(target=run, name="prospecta-date-extract", daemon=True).start()
    except Exception as e:
        _slots.release()
        return False, repr(e)
    if not finished.wait(timeout):
        return False, "timed out after %.1fs" % timeout
    if "e" in box:
        return False, "%s: %s" % (type(box["e"]).__name__, box["e"])
    return True, box["v"]


def make_gated(original, timeout=None):
    """Wrap ``extract_filters``: no cue -> people regex only, no model call; with a
    cue the library call runs under a wall-clock timeout (``timeout`` seconds, or
    the module setting) and falls back to the library's regex baseline."""
    def extract_filters(text, *, llm, people_vocab, now, **kwargs):
        if llm is None:
            return original(text, llm=llm, people_vocab=people_vocab, now=now, **kwargs)
        if has_date_cue(text):
            limit = _timeout if timeout is None else timeout
            if not limit or limit <= 0:
                return original(text, llm=llm, people_vocab=people_vocab, now=now, **kwargs)
            t0 = time.monotonic()
            ok, val = _call_with_timeout(
                lambda: original(text, llm=llm, people_vocab=people_vocab, now=now, **kwargs),
                limit)
            if ok:
                return val
            _stats["timeouts" if "timed out" in str(val) else "errors"] += 1
            logger.warning("prospecta: date extraction fell back to regex after %.2fs (%s)",
                           time.monotonic() - t0, val)
            from prospecta.channels.extract import regex_filters
            return regex_filters(text, people_vocab, now)
        from prospecta.channels.base import Filters
        return Filters(people=_people(text, people_vocab), date_from=None,
                       date_to=None, hard=False)
    extract_filters._date_cue_gate = True
    extract_filters.__wrapped__ = original
    return extract_filters


_lock = threading.Lock()
_holders = 0
_original = None


def install() -> bool:
    """Patch ``prospecta._index.extract_filters`` once (counted); True if this call
    is now holding the gate. Feature-detects and never double-wraps."""
    global _holders, _original
    with _lock:
        try:
            import prospecta._index as idx
            current = getattr(idx, "extract_filters", None)
            if not callable(current):
                logger.info("prospecta: date cue gate not installed (no _index.extract_filters)")
                return False
            if getattr(current, "_date_cue_gate", False):
                _holders += 1
                return True
            _original = current
            idx.extract_filters = make_gated(current)
            _holders = 1
            return True
        except Exception as e:  # never break the provider over an optimisation
            logger.warning("prospecta: date cue gate not installed: %s", e)
            return False


def uninstall() -> None:
    """Release one hold; restore the library's function when the last is released."""
    global _holders, _original
    with _lock:
        if _holders <= 0:
            return
        _holders -= 1
        if _holders == 0 and _original is not None:
            try:
                import prospecta._index as idx
                if getattr(idx.extract_filters, "_date_cue_gate", False):
                    idx.extract_filters = _original
            finally:
                _original = None
