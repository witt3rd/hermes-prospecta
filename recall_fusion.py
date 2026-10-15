"""Recall fixes for the Prospecta plugin, from the memory bake-off findings.

Four small, separate pieces (no I/O of their own except the ``Pg*Source`` classes):

* ``Bm25Index`` / ``KeywordLeg``: a plain raw-text keyword search over a bank's
  notes (the ``documents`` table, which every bank has; or, selectable, its
  chunk items): Okapi BM25, stemming OFF, OR semantics, identifier-friendly
  tokens. In memory only. Rows are read through the plugin's own bounded pool
  (short borrows, keyset pages, never a connection of its own) by a
  low-priority background thread with pauses, and a search never waits for a
  build. A changed bank is brought up to date by reading only the new rows
  when what was indexed is untouched, else by a full rebuild. No size cap.
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
import os
import re
import threading
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

RECALL_FUSIONS = ("keyword_first_union", "store")
DEFAULT_FUSION = "keyword_first_union"
DEFAULT_BLOCK_TOKENS = 1500
UNION_FIRST = 3                 # top hits taken from each leg before the store fills the rest
BLOCK_POOL = 50                 # candidates considered for a block (the bake-off widened 10 -> 50 notes)
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_RECHECK_S = 30.0        # how often the fingerprint is looked at
DEFAULT_MIN_REBUILD_S = 60.0    # a changed bank is rebuilt at most this often
DEFAULT_BATCH = 2000            # rows per database read
DEFAULT_PAUSE_S = 0.05          # pause after each database batch while building
_SLICE = 250                    # rows indexed between two yields to the gateway's threads
KEYWORD_UNITS = ("documents", "chunks")
DEFAULT_UNIT = "documents"


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
    attributes the plugin reads) plus the note's date. In the default
    (document) mode ``document_id`` is the note's own id and ``content`` is the
    note's raw text; ``item_id`` is empty."""

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
    """Okapi BM25 over raw texts (a bank's notes, or its chunk items). Append
    only: ``add_rows`` can be called again to extend a live index, and a search
    running meanwhile sees whole slices only (rows are published by one
    assignment after they are fully added), so no lock is needed. Postings are
    kept in compact arrays; nothing is capped."""

    # rough per-object costs (CPython 64-bit), used only for the size estimate
    _ROW_OVERHEAD = 5 * 56 + 80 + 100          # five strings, the row tuple, dict/array slots
    _TERM_OVERHEAD = 2 * 88 + 100 + 60         # two arrays, the dict slot, the term string
    _POSTING_BYTES = 8                         # uint32 row + uint32 tf

    def __init__(self, bank_id: str = "") -> None:
        self.bank_id = bank_id
        self._rows: List[Tuple[str, str, str, str, str]] = []  # item_id, document_id, source, date, text
        self._lengths = array("I")
        self._postings: Dict[str, Tuple[array, array]] = {}
        self._by_doc: Dict[str, int] = {}
        self._committed: Tuple[int, int] = (0, 0)               # (rows, total tokens) visible to search
        self._total_len = 0
        self._text_bytes = 0
        self._n_postings = 0
        self.cursor: Optional[Tuple[str, str]] = None           # (created_at, id) of the last row added
        self.max_updated = ""                                   # greatest updated_at among the rows

    def __len__(self) -> int:
        return self._committed[0]

    def add_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        for r in rows:
            text = r.get("text") or ""
            toks = tokenize(text)
            i = len(self._rows)
            doc_id = str(r.get("document_id") or "")
            self._rows.append((str(r.get("item_id") or ""), doc_id, r.get("source") or "",
                               r.get("created_on") or "", text))
            self._by_doc.setdefault(doc_id, i)
            self._lengths.append(len(toks))
            self._total_len += len(toks)
            self._text_bytes += len(text) + len(doc_id) + len(r.get("source") or "")
            for tok, tf in Counter(toks).items():
                pl = self._postings.get(tok)
                if pl is None:
                    pl = self._postings[tok] = (array("I"), array("I"))
                pl[0].append(i)
                pl[1].append(tf)
                self._n_postings += 1
            cur = r.get("cursor")
            if cur:
                self.cursor = (str(cur[0]), str(cur[1]))
            upd = r.get("updated_at") or ""
            if upd > self.max_updated:
                self.max_updated = upd
        self._committed = (len(self._rows), self._total_len)

    def stats(self) -> Dict[str, int]:
        """Counts and a size estimate (bytes), no content."""
        rows = self._committed[0]
        est = (self._text_bytes + rows * (self._ROW_OVERHEAD + 4)
               + self._n_postings * self._POSTING_BYTES + len(self._postings) * self._TERM_OVERHEAD)
        return {"rows": rows, "terms": len(self._postings), "postings": self._n_postings,
                "text_chars": self._text_bytes, "approx_bytes": est}

    def _hit(self, i: int, score: float, origin: str = "keyword") -> KeywordHit:
        item_id, doc_id, source, date, text = self._rows[i]
        return KeywordHit(
            content=text, original_chunk=text, source=source, score=score, scores={"bm25": score},
            metadata={}, bank_id=self.bank_id, document_id=doc_id, created_on=date, item_id=item_id,
            origin=origin)

    def documents(self, document_ids: Iterable[str]) -> Dict[str, KeywordHit]:
        """The indexed row of each given document id that is in the index
        (document mode: the note itself, score 0, origin 'store')."""
        n = self._committed[0]
        out: Dict[str, KeywordHit] = {}
        for d in document_ids:
            i = self._by_doc.get(str(d))
            if i is not None and i < n and str(d) not in out:
                out[str(d)] = self._hit(i, 0.0, origin="store")
        return out

    def search(self, query: str, limit: int) -> List[KeywordHit]:
        n, total_len = self._committed
        terms = set(tokenize(query))
        if n == 0 or not terms or limit < 1:
            return []
        avgdl = (total_len / n) or 1.0
        scores: Dict[int, float] = {}
        lengths = self._lengths
        for t in terms:
            pl = self._postings.get(t)
            if not pl:
                continue
            rows_, tfs = pl
            hits = [(i, tf) for i, tf in zip(rows_, tfs) if i < n]
            df = len(hits)
            if not df:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tf in hits:
                norm = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * lengths[i] / avgdl)
                scores[i] = scores.get(i, 0.0) + idf * tf * (BM25_K1 + 1.0) / norm
        top = heapq.nlargest(limit, scores.items(), key=lambda kv: (kv[1], -kv[0]))
        return [self._hit(i, s) for i, s in top]


# ---- sources (Postgres, through the plugin's pool) ----

_TS_FMT = "'YYYY-MM-DD HH24:MI:SS.US'"


def _ts(col: str) -> str:
    """A timestamptz column as fixed-width UTC text (sorts as text)."""
    return f"to_char(({col}) AT TIME ZONE 'UTC', {_TS_FMT})"


class _PgSource:
    """Reads a bank's rows through the plugin's pool, in keyset-paginated
    batches ordered by (created_at, id). Every call is a short borrow of one
    connection; nothing is held between batches or while the rows are being
    indexed, and no connection of its own is ever opened. The same ordering
    lets a refresh read only the rows after the last one it indexed."""

    unit = "document"
    # subclasses: FROM / WHERE (alias t) for the fingerprint, a full row query, and the key/time columns
    _FP_FROM = ""
    _ROW_FROM = ""
    _ROW_COLS = ""
    _KEY = "t.id"
    _CREATED = "t.created_at"
    _UPDATED = "t.updated_at"

    def __init__(self, pool: Any, bank_id: str, batch: int = DEFAULT_BATCH) -> None:
        self._pool = pool
        self._bank_id = bank_id
        self._batch = max(1, int(batch))

    def _after(self) -> str:
        return (f"({self._CREATED}, {self._KEY}) > "
                "(%s::timestamp AT TIME ZONE 'UTC', %s::uuid)")

    def fingerprint(self, cursor: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """One cheap aggregate: the bank's row count and latest write times, and,
        given the cursor of the last row indexed, the count and latest update
        time of the rows up to and including that cursor (the "prefix")."""
        sql = (f"SELECT count(*), {_ts('max(' + self._CREATED + ')')}, {_ts('max(' + self._UPDATED + ')')}")
        params: List[Any] = []
        if cursor is not None:
            le = f"({self._CREATED}, {self._KEY}) <= (%s::timestamp AT TIME ZONE 'UTC', %s::uuid)"
            sql += (f", count(*) FILTER (WHERE {le}), "
                    f"{_ts('max(' + self._UPDATED + ') FILTER (WHERE ' + le + ')')}")
            params += [cursor[0], cursor[1], cursor[0], cursor[1]]
        sql += f" FROM {self._FP_FROM}"
        params.append(self._bank_id)
        with self._pool.connection() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        fp = {"count": int(row[0]), "max_created": row[1] or "", "max_updated": row[2] or ""}
        if cursor is not None:
            fp["prefix_count"] = int(row[3])
            fp["prefix_max_updated"] = row[4] or ""
        return fp

    def batches(self, after: Optional[Tuple[str, str]] = None) -> Iterator[List[Dict[str, Any]]]:
        """Rows after ``after`` (a cursor) or all rows, ``batch`` at a time."""
        cursor = after
        while True:
            where = self._after() if cursor is not None else "true"
            params: List[Any] = [self._bank_id]
            if cursor is not None:
                params += [cursor[0], cursor[1]]
            params.append(self._batch)
            sql = (f"SELECT {self._ROW_COLS} FROM {self._ROW_FROM} AND {where} "
                   f"ORDER BY {self._CREATED}, {self._KEY} LIMIT %s")
            with self._pool.connection() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
            if not rows:
                return
            out = [self._row(r) for r in rows]
            cursor = out[-1]["cursor"]
            yield out

    def _row(self, r: Sequence[Any]) -> Dict[str, Any]:
        raise NotImplementedError


class PgDocumentSource(_PgSource):
    """A bank's notes: ``documents.original_text`` (the raw note), with its id,
    source and date. Every bank has them, whether or not it has chunk items."""

    unit = "document"
    _FP_FROM = "documents t WHERE t.bank_id = %s"
    _ROW_FROM = "documents t WHERE t.bank_id = %s"
    _ROW_COLS = (f"t.id::text, t.original_text, t.source, "
                 f"COALESCE(t.created_on::text, NULLIF(t.document_metadata->>'created_on', '')), "
                 f"{_ts('t.created_at')}, {_ts('t.updated_at')}")

    def _row(self, r: Sequence[Any]) -> Dict[str, Any]:
        return {"item_id": "", "document_id": r[0], "text": r[1] or "", "source": r[2] or "",
                "created_on": r[3] or "", "cursor": (r[4], r[0]), "updated_at": r[5]}


class PgChunkSource(_PgSource):
    """The old unit: a bank's chunk items (kind='chunk', text in original_chunk).
    Empty on a bank that has none (the live banks written before chunking)."""

    unit = "chunk"
    _KEY = "t.id"
    _FP_FROM = "memory_items t WHERE t.kind = 'chunk' AND t.bank_id = %s"
    _ROW_FROM = "memory_items t JOIN documents d ON d.id = t.document_id WHERE t.kind = 'chunk' AND t.bank_id = %s"
    _ROW_COLS = (f"t.id::text, t.document_id::text, t.original_chunk, d.source, "
                 f"COALESCE(d.created_on::text, NULLIF(d.document_metadata->>'created_on', '')), "
                 f"{_ts('t.created_at')}, {_ts('t.updated_at')}")

    def _row(self, r: Sequence[Any]) -> Dict[str, Any]:
        return {"item_id": r[0], "document_id": r[1], "text": r[2] or "", "source": r[3] or "",
                "created_on": r[4] or "", "cursor": (r[5], r[0]), "updated_at": r[6]}


def _rss_mb() -> Optional[float]:
    """Resident set of this process in MB (Linux /proc), or None."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        return None


def _lower_thread_priority() -> None:
    """Make the calling thread nicer than the gateway's own (Linux threads have
    their own nice value). Best effort."""
    try:
        tid = threading.get_native_id()
        os.setpriority(os.PRIO_PROCESS, tid, min(19, os.getpriority(os.PRIO_PROCESS, tid) + 10))
    except Exception:
        logger.debug("prospecta keyword leg: could not lower the build thread's priority", exc_info=True)


class KeywordLeg:
    """The in-memory keyword index of one bank, kept fresh by fingerprint.

    ``search`` never touches the database and never waits for a build: with no
    index yet it starts one on a background thread and answers "warming". While
    the fingerprint is unchanged nothing is read. When it changes and the rows
    already indexed are unchanged (the common case: notes were appended), only
    the new rows are read and added; otherwise the index is rebuilt in full
    (at most every ``min_rebuild_s``) and the old one serves until the new one
    is swapped in. A failure is logged with the full exception, leaves any old
    index serving, and is retried after ``recheck_s``. ``background=False`` runs
    the work inline (tests).

    There is no size cap. The build is polite instead: a low-priority thread,
    work in small slices, a pause after every database batch (``pause_s``). It
    logs the row count, the size estimate, the build time and the process
    memory, and, if ``warn_mb`` is set and the estimate exceeds it, a warning.
    """

    def __init__(self, source: Any, *, bank_id: str = "", clock: Callable[[], float] = time.monotonic,
                 recheck_s: float = DEFAULT_RECHECK_S, min_rebuild_s: float = DEFAULT_MIN_REBUILD_S,
                 pause_s: float = DEFAULT_PAUSE_S, warn_mb: float = 0.0, background: bool = True) -> None:
        self._source = source
        self._bank_id = bank_id
        self._clock = clock
        self._recheck_s = recheck_s
        self._min_rebuild_s = min_rebuild_s
        self._pause_s = max(0.0, pause_s)
        self._warn_mb = max(0.0, warn_mb or 0.0)
        self._warned_mb = 0.0
        self._background = background
        self.unit = getattr(source, "unit", "document")
        self._lock = threading.Lock()
        self._index: Optional[Bm25Index] = None
        self._fingerprint: Optional[Dict[str, Any]] = None
        self._built_at = 0.0
        self._checked_at: Optional[float] = None
        self._failed_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._closed = False
        self.last_build: Dict[str, Any] = {}   # counts, seconds and sizes of the latest build (no content)

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

    def documents(self, document_ids: Iterable[str]) -> Dict[str, KeywordHit]:
        """The indexed note for each document id the index holds (document mode)."""
        idx = self._index
        if idx is None or self.unit != "document":
            return {}
        try:
            return idx.documents(document_ids)
        except Exception:
            logger.exception("prospecta keyword leg document lookup failed")
            return {}

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

    def _read_into(self, idx: Bm25Index, after: Optional[Tuple[str, str]]) -> int:
        """Read the source's rows (after ``after``) into ``idx`` in slices, with a
        pause after each database batch. Returns the rows added; -1 if closed."""
        added = 0
        for rows in self._source.batches(after):
            # the connection is already back in the pool here; this is CPU only
            for lo in range(0, len(rows), _SLICE):
                with self._lock:
                    if self._closed:
                        return -1
                part = rows[lo:lo + _SLICE]
                idx.add_rows(part)
                added += len(part)
                time.sleep(0)               # let the gateway's own threads in
            if self._pause_s:
                time.sleep(self._pause_s)
        return added

    def _refresh(self) -> None:
        if self._background:
            _lower_thread_priority()
        try:
            with self._lock:
                idx = self._index
                last_fp = self._fingerprint
            cursor = idx.cursor if idx is not None else None
            fp = self._source.fingerprint(cursor) if cursor is not None else self._source.fingerprint()
            now = self._clock()
            mode = "full"
            if idx is not None:
                if fp_same(fp, last_fp):
                    with self._lock:
                        self._checked_at = now
                    return
                if (cursor is not None and fp.get("prefix_count") == len(idx)
                        and fp.get("prefix_max_updated", "") == idx.max_updated):
                    mode = "append"             # what was indexed is untouched; rows were added after it
                elif now - self._built_at < self._min_rebuild_s:
                    with self._lock:
                        self._checked_at = now
                    return
            started = time.monotonic()
            before = len(idx) if (mode == "append" and idx is not None) else 0
            if mode == "append" and idx is not None:
                work = idx
                added = self._read_into(work, cursor)
            else:
                work = Bm25Index(self._bank_id)
                added = self._read_into(work, None)
            if added < 0:
                return
            with self._lock:
                if self._closed:
                    return
                self._index, self._fingerprint = work, fp
                self._built_at = self._clock()
                self._checked_at = self._built_at
                self._failed_at = None
            self._report(work, mode, added, before, time.monotonic() - started)
        except Exception:
            logger.exception("prospecta keyword leg could not refresh bank %s; %s", self._bank_id,
                             "keeping the previous index" if self._index is not None else "recall uses the store only")
            with self._lock:
                self._failed_at = self._clock()
                self._checked_at = self._failed_at
        finally:
            with self._lock:
                self._running = False

    def _report(self, idx: Bm25Index, mode: str, added: int, before: int, seconds: float) -> None:
        st = idx.stats()
        mb = st["approx_bytes"] / 1e6
        rss = _rss_mb()
        self.last_build = {"mode": mode, "rows": st["rows"], "added": added, "seconds": seconds,
                           "approx_mb": mb, "terms": st["terms"], "postings": st["postings"], "rss_mb": rss}
        noun = "documents" if self.unit == "document" else "chunks"
        logger.info(
            "prospecta keyword leg: %s %d %s (%d now) of bank %s in %.1fs; index about %.0f MB "
            "(%d terms, %d postings)%s",
            "appended" if mode == "append" else "indexed", added if mode == "append" else st["rows"],
            noun, st["rows"], self._bank_id, seconds, mb, st["terms"], st["postings"],
            "" if rss is None else f"; process RSS {rss:.0f} MB")
        if self._warn_mb and mb > self._warn_mb and mb > self._warned_mb * 1.25:
            self._warned_mb = mb
            logger.warning(
                "prospecta keyword leg: the index of bank %s is about %.0f MB, over the soft threshold of "
                "%.0f MB (recall.keyword_index_warn_mb); it is kept and the leg stays on. Raise the threshold, "
                "or set recall.keyword to false to stop building it", self._bank_id, mb, self._warn_mb)


def fp_same(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
    """The bank-wide part of two fingerprints is equal."""
    if a is None or b is None:
        return False
    return all(a.get(k) == b.get(k) for k in ("count", "max_created", "max_updated"))


# ---- keyword-first union ----

def _key(item: Any, by_document: bool = False) -> Tuple[str, str]:
    doc = str(getattr(item, "document_id", "") or "")
    text = getattr(item, "original_chunk", None) or getattr(item, "content", "") or ""
    return (doc, "") if by_document and doc else (doc, text)


def keyword_first_union(keyword: Sequence[Any], store: Sequence[Any], *, slots: int,
                        first: int = UNION_FIRST, by_document: bool = False) -> List[Any]:
    """Top ``first`` of the keyword leg, then top ``first`` of the store (both
    de-duplicated), then the store's own order for the remaining slots (the
    keyword leg's remainder only if the store runs out). An entry is the same
    when it is the same chunk of the same note; with ``by_document`` it is the
    same when it is the same note, whatever part of it each side returned."""
    out: List[Any] = []
    seen = set()

    def take(items: Iterable[Any]) -> None:
        for it in items:
            if len(out) >= slots:
                return
            k = _key(it, by_document)
            if k not in seen:
                seen.add(k)
                out.append(it)

    take(list(keyword)[:first])
    take(list(store)[:first])
    take(list(store)[first:])
    take(list(keyword)[first:])
    return out


def with_document_text(items: Sequence[Any], documents: Dict[str, "KeywordHit"], budget: int = 0) -> List[Any]:
    """Give every store result the text of its whole note, so a note found by
    both legs reads the same whichever found it. ``documents`` maps a document
    id to its indexed note (``KeywordLeg.documents``); items whose note is not
    in the index (written since it was built) are kept as they are. A note too
    long to ever fit the token ``budget`` keeps the store's own, shorter text
    when the store returned one that fits (a note is never cut)."""
    out: List[Any] = []
    for it in items:
        doc = documents.get(str(getattr(it, "document_id", "")))
        if doc is None or isinstance(it, KeywordHit):
            out.append(it)
            continue
        own = getattr(it, "original_chunk", None) or getattr(it, "content", "") or ""
        if budget > 0 and own and estimate_tokens(doc.content) > budget >= estimate_tokens(own):
            out.append(it)
        else:
            out.append(replace(doc, origin=getattr(it, "origin", "store")))
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
