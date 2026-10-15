"""Recall fixes for the Prospecta plugin, from the memory bake-off findings.

Four small, separate pieces (no I/O of their own except ``PgChunkSource``):

* ``Bm25Index`` / ``KeywordLeg``: a plain raw-text keyword search over a bank's
  chunk items: Okapi BM25, stemming OFF, OR semantics, identifier-friendly
  tokens. In memory only; the leg rebuilds when the bank's fingerprint changes.
  Rows are read through the plugin's own bounded pool (short borrows, never a
  connection of its own), and a search never waits for a build.
* ``keyword_first_union``: the top hits of the keyword leg first, then the
  store's own order, so the best channel's top hit is never demoted.
* ``build_block``: one block of raw text in that order, sized by a token budget;
  an entry that does not fit is skipped whole, never cut.
* ``AbstentionTally``: abstention counts printed together with the
  false-abstain rate (counts only, no content).
"""
from __future__ import annotations

import heapq
import logging
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RECALL_FUSIONS = ("keyword_first_union", "store")
DEFAULT_FUSION = "keyword_first_union"
DEFAULT_BLOCK_TOKENS = 1500
UNION_FIRST = 3                 # top hits taken from each leg before the store fills the rest
BLOCK_POOL = 50                 # candidates considered for a block (the bake-off widened 10 -> 50 notes)
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_MAX_CHUNKS = 400_000    # a bank with more chunk items than this gets no in-memory keyword leg
DEFAULT_RECHECK_S = 30.0        # how often the fingerprint is looked at
DEFAULT_MIN_REBUILD_S = 60.0    # a changed bank is rebuilt at most this often
_SCAN_BATCH = 5000


# ---- tokens ----

# An identifier or file name: alphanumeric runs joined by - _ or . (so
# "deploy-config_v2.yaml" and "step-3" stay whole). A '.' not followed by an
# alphanumeric is sentence punctuation and ends the token.
_IDENT = re.compile(r"[^\W_]+(?:[-_.]+[^\W_]+)*")
_PART = re.compile(r"[^\W_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase tokens, no stemming. An identifier with - _ or . inside is
    emitted whole, followed by its parts."""
    out: List[str] = []
    for m in _IDENT.finditer((text or "").lower()):
        word = m.group(0)
        parts = _PART.findall(word)
        if len(parts) > 1:
            out.append(word)
        out.extend(parts)
    return out


def estimate_tokens(text: str) -> int:
    """Token estimate used for the block budget. The plugin has no tokenizer, so
    this is a whitespace count scaled for sub-word pieces (English prose runs
    about 1.3 tokens per word), floored by characters / 4 so that text without
    spaces (identifiers, paths, base64) is not under-counted."""
    if not text:
        return 0
    return max(math.ceil(len(text.split()) * 1.3), math.ceil(len(text) / 4))


# ---- BM25 ----

@dataclass(frozen=True)
class KeywordHit:
    """A keyword-leg result, shaped like the library's RecalledMemory (the
    attributes the plugin reads) plus the note's date."""

    content: str
    original_chunk: str
    source: str
    score: float
    scores: Dict[str, float]
    metadata: Dict[str, Any]
    bank_id: str
    document_id: str
    created_on: str = ""
    item_id: str = ""
    origin: str = "keyword"


class Bm25Index:
    """Okapi BM25 over chunk texts. Built once, then read-only."""

    def __init__(self, bank_id: str = "") -> None:
        self.bank_id = bank_id
        self._rows: List[Tuple[str, str, str, str, str]] = []  # item_id, document_id, source, date, text
        self._lengths: List[int] = []
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._rows)

    def add_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        for r in rows:
            text = r.get("text") or ""
            toks = tokenize(text)
            i = len(self._rows)
            self._rows.append((str(r.get("item_id") or ""), str(r.get("document_id") or ""),
                               r.get("source") or "", r.get("created_on") or "", text))
            self._lengths.append(len(toks))
            self._total_len += len(toks)
            for tok, tf in Counter(toks).items():
                self._postings.setdefault(tok, []).append((i, tf))

    def search(self, query: str, limit: int) -> List[KeywordHit]:
        n = len(self._rows)
        terms = set(tokenize(query))
        if n == 0 or not terms or limit < 1:
            return []
        avgdl = (self._total_len / n) or 1.0
        scores: Dict[int, float] = {}
        for t in terms:
            plist = self._postings.get(t)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tf in plist:
                norm = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * self._lengths[i] / avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (BM25_K1 + 1.0) / norm
        top = heapq.nlargest(limit, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        hits = []
        for i, s in top:
            item_id, doc_id, source, date, text = self._rows[i]
            hits.append(KeywordHit(
                content=text, original_chunk=text, source=source, score=s, scores={"bm25": s},
                metadata={}, bank_id=self.bank_id, document_id=doc_id, created_on=date, item_id=item_id))
        return hits


class PgChunkSource:
    """Reads a bank's chunk items (kind='chunk', text in original_chunk) through
    the plugin's pool. Every call is a short borrow; nothing is held between
    batches, and no connection of its own is ever opened."""

    def __init__(self, pool: Any, bank_id: str, batch: int = _SCAN_BATCH) -> None:
        self._pool = pool
        self._bank_id = bank_id
        self._batch = batch

    def fingerprint(self) -> Tuple[Any, ...]:
        """Changes when chunk items are added, replaced or removed."""
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*), max(created_at)::text, max(updated_at)::text "
                "FROM memory_items WHERE bank_id = %s AND kind = 'chunk'", (self._bank_id,)).fetchone()
        return (int(row[0]), row[1], row[2])

    def batches(self) -> Iterator[List[Dict[str, Any]]]:
        last = "00000000-0000-0000-0000-000000000000"
        while True:
            with self._pool.connection() as conn:
                rows = conn.execute(
                    "SELECT mi.id::text, mi.document_id::text, mi.original_chunk, d.source, "
                    "COALESCE(d.created_on::text, NULLIF(d.document_metadata->>'created_on', '')) "
                    "FROM memory_items mi JOIN documents d ON d.id = mi.document_id "
                    "WHERE mi.bank_id = %s AND mi.kind = 'chunk' AND mi.id > %s::uuid "
                    "ORDER BY mi.id LIMIT %s", (self._bank_id, last, self._batch)).fetchall()
            if not rows:
                return
            last = rows[-1][0]
            yield [{"item_id": r[0], "document_id": r[1], "text": r[2], "source": r[3] or "",
                    "created_on": r[4] or ""} for r in rows]


class _BankTooBig(RuntimeError):
    pass


class KeywordLeg:
    """The in-memory keyword index of one bank, kept fresh by fingerprint.

    ``search`` never touches the database and never waits for a build: with no
    index yet it starts one on a background thread and answers "warming". While
    the fingerprint is unchanged nothing is read; when it changes the index is
    rebuilt (at most every ``min_rebuild_s``) and the old one serves until the
    new one is swapped in. A failure is logged with the full exception, leaves
    any old index serving, and is retried after ``recheck_s``. ``background=False``
    runs the work inline (tests).
    """

    def __init__(self, source: Any, *, bank_id: str = "", clock: Callable[[], float] = time.monotonic,
                 recheck_s: float = DEFAULT_RECHECK_S, min_rebuild_s: float = DEFAULT_MIN_REBUILD_S,
                 max_chunks: int = DEFAULT_MAX_CHUNKS, background: bool = True) -> None:
        self._source = source
        self._bank_id = bank_id
        self._clock = clock
        self._recheck_s = recheck_s
        self._min_rebuild_s = min_rebuild_s
        self._max_chunks = max_chunks
        self._background = background
        self._lock = threading.Lock()
        self._index: Optional[Bm25Index] = None
        self._fingerprint: Optional[Tuple[Any, ...]] = None
        self._built_at = 0.0
        self._checked_at: Optional[float] = None
        self._failed_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._closed = False

    # -- public --

    def search(self, query: str, limit: int) -> Tuple[str, List[KeywordHit]]:
        """("ok" | "warming" | "unavailable", hits)."""
        with self._lock:
            if self._closed:
                return "unavailable", []
            idx = self._index
            due = self._due_locked(idx)
        if due:
            self._refresh_async()
            with self._lock:
                idx = self._index
        if idx is None:
            with self._lock:
                failed = self._failed_at is not None and not self._running
                running = self._running
            return ("warming", []) if running or not failed else ("unavailable", [])
        try:
            return "ok", idx.search(query, limit)
        except Exception:
            logger.exception("prospecta keyword leg search failed")
            return "unavailable", []

    def ready(self) -> bool:
        return self._index is not None

    def wait_idle(self, timeout: float = 30.0) -> None:
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    def stop(self) -> None:
        with self._lock:
            self._closed = True
            self._index = None
        self.wait_idle(10.0)

    # -- internals --

    def _due_locked(self, idx: Optional[Bm25Index]) -> bool:
        if self._running:
            return False
        now = self._clock()
        if idx is None:
            return self._failed_at is None or now - self._failed_at >= self._recheck_s
        return self._checked_at is None or now - self._checked_at >= self._recheck_s

    def _refresh_async(self) -> None:
        with self._lock:
            if self._running or self._closed:
                return
            self._running = True
        if not self._background:
            self._refresh()
            return
        self._thread = threading.Thread(target=self._refresh, daemon=True, name="prospecta-keyword-build")
        self._thread.start()

    def _refresh(self) -> None:
        try:
            fp = self._source.fingerprint()
            now = self._clock()
            with self._lock:
                have = self._index is not None
                unchanged = have and fp == self._fingerprint
                too_soon = have and now - self._built_at < self._min_rebuild_s
                if unchanged or too_soon:
                    self._checked_at = now
                    return
            if int(fp[0]) > self._max_chunks:
                raise _BankTooBig(
                    f"bank {self._bank_id!r} has {fp[0]} chunk items, over the {self._max_chunks} "
                    "the in-memory keyword leg will index; the keyword leg is off for this bank")
            started = time.monotonic()
            idx = Bm25Index(self._bank_id)
            for rows in self._source.batches():
                with self._lock:
                    if self._closed:
                        return
                idx.add_rows(rows)
            with self._lock:
                if self._closed:
                    return
                self._index, self._fingerprint = idx, fp
                self._built_at = self._clock()
                self._checked_at = self._built_at
                self._failed_at = None
            logger.info("prospecta keyword leg: indexed %d chunks of bank %s in %.1fs",
                        len(idx), self._bank_id, time.monotonic() - started)
        except Exception:
            logger.exception("prospecta keyword leg could not refresh bank %s; %s", self._bank_id,
                             "keeping the previous index" if self._index is not None else "recall uses the store only")
            with self._lock:
                self._failed_at = self._clock()
                self._checked_at = self._failed_at
        finally:
            with self._lock:
                self._running = False


# ---- keyword-first union ----

def _key(item: Any) -> Tuple[str, str]:
    return (str(getattr(item, "document_id", "")),
            getattr(item, "original_chunk", None) or getattr(item, "content", "") or "")


def keyword_first_union(keyword: Sequence[Any], store: Sequence[Any], *, slots: int,
                        first: int = UNION_FIRST) -> List[Any]:
    """Top ``first`` of the keyword leg, then top ``first`` of the store (both
    de-duplicated), then the store's own order for the remaining slots (the
    keyword leg's remainder only if the store runs out). An entry is the same
    when it is the same chunk of the same note."""
    out: List[Any] = []
    seen = set()

    def take(items: Iterable[Any]) -> None:
        for it in items:
            if len(out) >= slots:
                return
            k = _key(it)
            if k not in seen:
                seen.add(k)
                out.append(it)

    take(list(keyword)[:first])
    take(list(store)[:first])
    take(list(store)[first:])
    take(list(keyword)[first:])
    return out


# ---- token-budget block ----

@dataclass
class BlockEntry:
    source: str
    date: str
    text: str
    origin: str = "store"


@dataclass
class BlockResult:
    text: str
    tokens: int
    included: List[BlockEntry] = field(default_factory=list)
    omitted: List[BlockEntry] = field(default_factory=list)


def format_entry(e: BlockEntry) -> str:
    return f"[source: {e.source or 'unknown'} | date: {e.date or 'undated'}]\n{e.text}"


def build_block(entries: Sequence[BlockEntry], budget: int) -> BlockResult:
    """One block of raw text in the given order. An entry that does not fit the
    token ``budget`` is skipped whole and the next that fits is taken; stored
    text is never cut."""
    parts: List[str] = []
    included: List[BlockEntry] = []
    omitted: List[BlockEntry] = []
    text = ""
    for e in entries:
        candidate = "\n\n".join(parts + [format_entry(e)])
        if estimate_tokens(candidate) <= budget:
            parts.append(format_entry(e))
            included.append(e)
            text = candidate
        else:
            omitted.append(e)
    return BlockResult(text=text, tokens=estimate_tokens(text), included=included, omitted=omitted)


# ---- abstention, always with the false-abstain rate ----

class AbstentionTally:
    """Running counts of a shadow comparison.

    A side *abstains* on a question when it returned nothing. A question is
    *answerable* when the old side, the new side or the keyword leg found
    something. A *false abstain* is an abstention on an answerable question.
    ``new_found=None`` (the new side errored) is left out of the new-side counts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c = {"pairs": 0, "answerable": 0, "old_abstain": 0, "old_false_abstain": 0,
                   "new_pairs": 0, "new_abstain": 0, "new_false_abstain": 0}

    def record(self, *, old_found: bool, new_found: Optional[bool], kw_found: bool) -> Dict[str, int]:
        answerable = bool(old_found or new_found or kw_found)
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
