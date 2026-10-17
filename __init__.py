"""Prospecta memory provider for Hermes Agent.

Hybrid search and RAG over Postgres (Prospecta's docs/postgres-hybrid.md): notes
are written as chunks with no model call; recall is the library's Stage 1 (one SQL
query: semantic, stemmed and exact/phrase retrievers with reserved slots) and
Stage 2 (a cross-encoder rerank), every hit cited (source, date, kind, match,
exact-match snippet). The read side can also expand a question with an LLM.

Substrate: Postgres + pgvector. Two deployment modes:
  - BYO Postgres: set DATABASE_URL env.
  - Embedded: plugin spins up docker-compose Postgres on first init.

LLM: reads PROSPECTA_LLM_MODEL env (or config llm_model) and calls
     prospecta.defaults.make_default_llm — same pattern as Hindsight.
     No Hermes ctx.llm dependency.
Embedder: uses prospecta.defaults make_default_embedder (LiteLLM-backed),
          OR a configured-per-plugin embedder via prospecta.embed.*.

Tool surface (visible to the agent):
  - prospecta_retain(content, source=..., index_text=None, tags=None)
  - prospecta_recall(query, limit=10) -> synthesis + cited sources
  - prospecta_search(query, mode="hybrid", limit=10, date_from=, date_to=) -> cited chunks
  - prospecta_recall_set(query, entity, aliases=None) -> complete cited list (map-reduce)
  (recall/search take an optional depth: standard | deep)

Shadow reads (OFF by default; see README): with ``shadow_bank_id`` set, every
search / recall / prefetch is also run against that bank on a background
thread and the old/new pair is stored in recall_events. The agent only ever
sees the old bank's answer. The shadow log prints the false-abstain rate beside
every abstention figure.

Recall (see README, "Recall: Stage 1 + Stage 2, cited"): ``prospecta_search`` and
``prospecta_recall`` read through the library (the bank's ``hybrid_chunk`` channel
and ``cross_encoder`` rerank), and every hit carries its citation fields.

Linked / dual writes (OFF by default; see README): ``linked_writes`` builds the
library's linker for the main Memory; ``dual_write_bank_id`` also retains every
note into a second bank on a background thread, with a JSONL retry file.

Note dates (see README): the library stamps every new note's date
(``documents.created_on``); the ``timezone`` config (default America/Los_Angeles)
is the zone of that day (handed to the library as PROSPECTA_TIMEZONE).

Prefetch (OFF by default; see README, "Prefetch"): with ``prefetch_enabled`` on,
``recall.prefetch.mode`` picks the old synthesised answer ('off', the default) or the
written recall contract ('contract', ``prefetch_contract.py``: one typed decision per
message (needs memory? which period of time?), then the library's query expansion
(``Memory.expand_and_recall``: the message becomes structured searches, run in parallel
through Stage 1 + Stage 2 with the thread's last turns excluded in SQL, one cited block),
no synthesis, never fails a turn).

Lifecycle hooks: sync_turn (non-blocking daemon thread), on_session_end,
                  shutdown. Prefetch OPT-IN via config (default OFF —
                  spine cost should not be a surprise tax).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone as _utc_tz
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

try:  # as a package (how hermes loads the plugin) ...
    from . import abstention as _ab
    from . import date_cue_gate as _dcg
    from . import prefetch_contract as _pc
except ImportError:  # ... or with the checkout on sys.path (tests, console scripts)
    import abstention as _ab
    import date_cue_gate as _dcg
    import prefetch_contract as _pc

logger = logging.getLogger(__name__)

# Silence LiteLLM's verbose debug output before it gets imported anywhere
# in this process. LiteLLM emits DEBUG messages during module initialization
# and on every completion call; those flood the Hermes CLI terminal.
# Set by logger name first (catches import-time messages), then re-applied
# after import via litellm.suppress_debug_info in _build_llm().
for _litellm_logger in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
    logging.getLogger(_litellm_logger).setLevel(logging.WARNING)


_TOOL_GUIDE = (
    "Memory tool guide:\n"
    "- prospecta_search: no answer-writing model. It returns cited hits: each has its source, date, "
    "kind, how it matched, a relevance score, the matched words in context (exact_snippet) and the "
    "whole chunk. Put an \"exact phrase\" in double quotes to match those words side by side; "
    "identifiers and file names match whole. Use it whenever you want to read the notes yourself.\n"
    "- prospecta_recall: reads the matches and writes one synthesized answer. It costs LLM "
    "calls, so use it when you need an answer, not a list.\n"
    "- depth (search and recall): 'standard' suits ordinary questions. 'deep' reads a wider pool "
    "for about 3x the cost; use it for 'everything about X' or 'every time ...' questions.\n"
    "- date_from / date_to (search, YYYY-MM-DD): only notes dated in that window. A note's date "
    "is the day it was saved unless it carries its own.\n"
    "- A hit may be a generated question that points to a note, not the note itself. Read the "
    "note it points to before relying on it.\n"
    "- Retaining is automatic after every turn. Call prospecta_retain only for a fact the "
    "person asks you to remember, with a clear `source`."
)


def _depth_kw(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pass-through of the agent's optional ``depth`` ("standard" | "deep").

    Omitted means the library's own default (the bank's recall_config.depth,
    else standard); the value is validated by Prospecta, not here.
    """
    depth = args.get("depth")
    return {"depth": depth} if depth else {}


def _iso_date(value: Any, name: str) -> str:
    """A tool's date argument as YYYY-MM-DD; ValueError (shown to the agent) when it is not one."""
    from datetime import date
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        raise ValueError(f"{name} must be a date, YYYY-MM-DD") from None


def _stage2_graded(trace: Any) -> bool:
    """Did the library's Stage 2 grade this search's hits? False when the bank has no rerank stage or
    when it failed and Stage 1's own order stands (the trace of ``Memory.search(_trace=...)`` says
    which). Never raises."""
    try:
        for tr in trace or []:
            rerank = tr.get("rerank")
            if rerank and not rerank.get("fallback_reason") and not tr.get("fallback_reason"):
                return True
    except Exception:
        pass
    return False


def _cited_hit(n: int, item: Any, *, text: bool = True) -> Dict[str, Any]:
    """One hit with its citation fields (the library's cited-evidence contract): where and when it is
    from, what kind of record it is, why it matched, how relevant it is and, when a lexical retriever
    fired, the matched words in context (``<<marked>>``). ``text`` adds the whole chunk."""
    score = getattr(item, "relevance_score", None)
    hit: Dict[str, Any] = {
        "n": n,
        "source": getattr(item, "source", None),
        "document_id": getattr(item, "document_id", None),
        "created_on": getattr(item, "created_on", None),
        "kind": getattr(item, "kind", None),
        "match_kind": getattr(item, "match_kind", None),
        "relevance_score": round(float(score), 4) if isinstance(score, (int, float)) else None,
        "exact_snippet": getattr(item, "exact_snippet", None),
    }
    if text:
        hit["text"] = getattr(item, "original_chunk", None) or getattr(item, "content", "") or ""
    return hit


# ---- shadow reads (embedding migration) ----

_SHADOW_QUEUE_MAX = 16          # pending shadow jobs; beyond this a job is dropped, loudly
# the run ends itself at its deadline with the one-line "not consulted" note; this grace keeps the
# outer wait from cutting it a moment first (the outer wait then still returns "" if the run hangs)
_DEADLINE_GRACE_S = 0.25
_RECOVER_SOURCES_TIMEOUT_S = 0.5   # reading the last two turns' sources from the bank after a restart
_SHADOW_DRAIN_TIMEOUT_S = 30.0  # how long shutdown waits for pending shadow jobs
_SHADOW_STOP = object()
_SHADOW_DEFAULT_EMBED_MODEL = "openrouter/openai/text-embedding-3-large"


_DEFAULT_TIMEZONE = "America/Los_Angeles"


def local_date_iso(tz_name: Optional[str] = None, now: Optional[datetime] = None) -> str:
    """Today's date (YYYY-MM-DD) in the named timezone; never raises.

    ``now`` (an aware datetime, default the current time) is converted to the
    zone, so the date is the local calendar day, not the UTC one. An empty or
    unknown zone name falls back to America/Los_Angeles (logged), and if no
    zone data is available at all, to UTC.
    """
    now = now or datetime.now(_utc_tz.utc)
    try:
        from zoneinfo import ZoneInfo
    except Exception:  # pragma: no cover - zoneinfo is stdlib on 3.9+
        return now.astimezone(_utc_tz.utc).date().isoformat()
    wanted = (tz_name or "").strip() or _DEFAULT_TIMEZONE
    for name in (wanted, _DEFAULT_TIMEZONE):
        try:
            return now.astimezone(ZoneInfo(name)).date().isoformat()
        except Exception:
            logger.warning("prospecta: unknown timezone %r for note dates; trying the default", name)
    return now.astimezone(_utc_tz.utc).date().isoformat()


_DEFAULT_MAX_CONNECTIONS = 4
_DEFAULT_POOL_IDLE_SECONDS = 300
_POOL_WAIT_S = 30.0


class _BoundedPool:
    """Stands in for a Memory's ``_pool``: at most ``max_connections`` shared
    connections, however many threads use it.

    The library's own pool keeps one connection per thread and never closes it
    when the thread ends, which piles up on a gateway that runs each turn on a
    new thread. Here a thread borrows one connection for the length of a
    ``connection()``/``cursor()`` block (nested blocks on the same thread reuse
    it) and gives it back. A connection that has been lost is dropped and the
    next borrower gets a fresh one. A connection that sits unused for longer
    than ``idle_seconds`` is closed (by a timer, and never handed out stale);
    the next use opens a new one. Same methods as the library pool:
    connection(), cursor(), dedicated(), close(), ``_conn``, ``database_url``.
    """

    def __init__(self, database_url: str, max_connections: int = _DEFAULT_MAX_CONNECTIONS,
                 idle_seconds: float = _DEFAULT_POOL_IDLE_SECONDS,
                 clock: Any = time.monotonic) -> None:
        self._database_url = database_url
        self._max = max(1, int(max_connections))
        self._idle_seconds = idle_seconds
        self._clock = clock
        self._idle_since: Dict[int, float] = {}  # id(conn) -> when it went idle
        self._timer: Optional[threading.Timer] = None
        self._cond = threading.Condition()
        self._idle: List[Any] = []
        self._open: List[Any] = []   # every connection this pool opened and still holds
        self._creating = 0
        self._closed = False
        self._local = threading.local()  # the connection this thread has borrowed right now

    @property
    def database_url(self) -> str:
        return self._database_url

    @property
    def max_size(self) -> int:
        """How many connections the pool holds at most (the library's query expansion bounds the
        searches it runs in the database at once by it)."""
        return self._max

    @property
    def _conn(self):
        return getattr(self._local, "conn", None)

    @staticmethod
    def _lost(conn) -> bool:
        return bool(conn.closed or getattr(conn, "broken", False))

    def _drop(self, conn) -> None:
        with self._cond:
            if conn in self._open:
                self._open.remove(conn)
            self._cond.notify()
        try:
            if not conn.closed:
                conn.close()
        except Exception:
            logger.debug("prospecta: closing a lost connection failed", exc_info=True)

    def _expire_idle(self) -> List[Any]:
        """Take the connections idle longer than ``idle_seconds`` out of the
        pool (caller holds the lock and closes them)."""
        now = self._clock()
        stale = [c for c in self._idle if now - self._idle_since.get(id(c), now) > self._idle_seconds]
        for c in stale:
            self._idle.remove(c)
            self._idle_since.pop(id(c), None)
            if c in self._open:
                self._open.remove(c)
        if stale:
            self._cond.notify_all()
        return stale

    def _close_all(self, conns: List[Any]) -> None:
        for c in conns:
            try:
                if not c.closed:
                    c.close()
            except Exception:
                logger.debug("prospecta: closing an idle connection failed", exc_info=True)

    def reap(self) -> None:
        """Close the connections that have been idle too long."""
        with self._cond:
            stale = self._expire_idle()
        self._close_all(stale)

    def _reap_later(self) -> None:
        """Run ``reap`` once the oldest idle connection is due (lock held)."""
        if self._timer is not None or self._closed or not self._idle:
            return
        now = self._clock()
        oldest = min(self._idle_since.get(id(c), now) for c in self._idle)
        delay = max(0.0, oldest + self._idle_seconds - now) + 0.01
        self._timer = threading.Timer(delay, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        with self._cond:
            self._timer = None
        try:
            self.reap()
        finally:
            with self._cond:
                self._reap_later()

    def _acquire(self):
        import psycopg
        deadline = time.monotonic() + _POOL_WAIT_S
        with self._cond:
            stale = self._expire_idle()
        self._close_all(stale)
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("prospecta connection pool is closed")
                while self._idle:
                    conn = self._idle.pop()
                    self._idle_since.pop(id(conn), None)
                    if not self._lost(conn):
                        return conn
                    self._open.remove(conn)
                    try:
                        if not conn.closed:
                            conn.close()  # broken: only frees the socket, no round trip
                    except Exception:
                        pass
                if len(self._open) + self._creating < self._max:
                    self._creating += 1
                    break
                if not self._cond.wait(timeout=max(0.0, deadline - time.monotonic())) \
                        and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"no free database connection after {_POOL_WAIT_S:.0f}s "
                        f"(max_connections={self._max})")
        try:
            conn = psycopg.connect(self._database_url, autocommit=False)
        except BaseException:
            with self._cond:
                self._creating -= 1
                self._cond.notify()
            raise
        with self._cond:
            self._creating -= 1
            if self._closed:
                conn.close()
                raise RuntimeError("prospecta connection pool is closed")
            self._open.append(conn)
        return conn

    def _release(self, conn) -> None:
        if not self._lost(conn):
            try:
                if conn.info.transaction_status != 0:  # not IDLE: never hand on a half-done transaction
                    conn.rollback()
            except Exception:
                pass
        if self._lost(conn) or self._closed:
            self._drop(conn)
            return
        with self._cond:
            self._idle.append(conn)
            self._idle_since[id(conn)] = self._clock()
            self._cond.notify()
            self._reap_later()

    @contextmanager
    def _borrowed(self):
        conn = self._conn
        if conn is not None:  # nested on this thread: same connection, same transaction
            yield conn
            return
        conn = self._acquire()
        self._local.conn = conn
        try:
            yield conn
        finally:
            self._local.conn = None
            self._release(conn)

    @contextmanager
    def connection(self):
        """Yield a connection. Caller manages commit/rollback."""
        with self._borrowed() as conn:
            try:
                yield conn
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

    @contextmanager
    def cursor(self):
        """Yield a cursor with commit on success, rollback on error."""
        with self._borrowed() as conn:
            try:
                with conn.cursor() as cur:
                    yield cur
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise

    @contextmanager
    def dedicated(self):
        """A private connection for work on another thread; closed afterwards."""
        import psycopg
        conn = psycopg.connect(self._database_url, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            conns, self._open, self._idle = self._open, [], []
            self._idle_since.clear()
            timer, self._timer = self._timer, None
            self._cond.notify_all()
        if timer is not None:
            timer.cancel()
        for c in conns:
            try:
                if not c.closed:
                    c.close()
            except Exception:
                logger.debug("prospecta: closing a connection failed", exc_info=True)


def _build_memory(max_connections: int = _DEFAULT_MAX_CONNECTIONS,
                  pool_idle_seconds: float = _DEFAULT_POOL_IDLE_SECONDS, **kw: Any):
    """``prospecta.Memory(**kw)`` whose connections are a ``_BoundedPool``
    (also used by its default tracer sink), so ``close()`` ends them all."""
    from prospecta.memory import Memory
    memory = Memory(**kw)
    old = getattr(memory, "_pool", None)
    if old is not None:
        pool = _BoundedPool(old.database_url, max_connections, pool_idle_seconds)
        memory._pool = pool
        tracer = getattr(memory, "_tracer", None)
        if getattr(tracer, "_pool", None) is old:
            tracer._pool = pool
    return memory


_DEFAULT_CONNECTION_WATCH_PERCENT = 70
_CONNECTION_WATCH_INTERVAL_S = 120.0
_HEALTH_FILENAME = "memory-provider-health.json"  # hermes-agent's agent/memory_health.py
_WATCH_KIND = "connection_limit"


class _ConnectionWatch:
    """Warns loudly, in time, when the database nears its connection limit.

    A daemon thread checks at start and then every ``interval`` seconds, with
    one cheap query on its own short-lived connection (never the pool's). At
    ``percent`` of ``max_connections`` or more, or when the query itself fails
    (a full server refuses it), it logs at ERROR with the numbers and writes
    ``$HERMES_HOME/memory-provider-health.json`` in hermes-agent's format, so
    the being's next turn shows a notice. It clears that notice, and only its
    own, once the numbers are back under the line.
    """

    def __init__(self, database_url: str, hermes_home: Path, percent: int = _DEFAULT_CONNECTION_WATCH_PERCENT,
                 interval: float = _CONNECTION_WATCH_INTERVAL_S, provider: str = "prospecta") -> None:
        self._url = database_url
        self._path = Path(hermes_home) / _HEALTH_FILENAME
        self._percent = percent
        self._interval = interval
        self._provider = provider
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="prospecta-connection-watch")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=10.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check()
            self._stop.wait(self._interval)

    def _count(self) -> "tuple[int, int]":
        import psycopg
        with psycopg.connect(self._url, autocommit=True, connect_timeout=5) as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()")
            used = int(cur.fetchone()[0])
            cur.execute("SHOW max_connections")
            limit = int(cur.fetchone()[0])
        return used, limit

    def check(self) -> None:
        """One check; never raises."""
        try:
            try:
                used, limit = self._count()
            except Exception as e:
                self._alarm(f"the connection check failed, the database may be full: {type(e).__name__}: {e}")
                return
            pct = 100.0 * used / limit if limit > 0 else 100.0
            if pct >= self._percent:
                self._alarm(f"the database is close to its connection limit: {used} of {limit} "
                            f"connections in use ({pct:.0f}%, warning line {self._percent}%); "
                            "memory will fail when it is full")
            else:
                self._clear()
        except Exception:
            logger.exception("prospecta connection watch failed")

    def _read(self) -> Dict[str, Any]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _alarm(self, reason: str) -> None:
        logger.error("memory_provider_unavailable provider=%s source=connection_watch: %s",
                     self._provider, reason)
        prev = self._read()
        same = prev.get("provider") == self._provider and prev.get("kind") == _WATCH_KIND
        if prev.get("status") == "unavailable" and not same:
            return  # another cause already owns the notice; do not overwrite it
        now = datetime.now(_utc_tz.utc).isoformat(timespec="seconds")
        state = {"status": "unavailable", "provider": self._provider, "kind": _WATCH_KIND,
                 "reason": reason, "source": "connection_watch",
                 "since": (prev.get("since") if same else None) or now, "last_checked": now}
        try:
            tmp = self._path.with_name(self._path.name + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:
            logger.exception("prospecta: could not write %s", self._path)

    def _clear(self) -> None:
        prev = self._read()
        if prev.get("provider") == self._provider and prev.get("kind") == _WATCH_KIND:
            try:
                self._path.unlink(missing_ok=True)
                logger.info("memory_provider_recovered provider=%s source=connection_watch", self._provider)
            except Exception:
                logger.exception("prospecta: could not remove %s", self._path)


class _BackgroundWorker:
    """One daemon thread behind a bounded queue that owns one ``Memory``.

    Shared by the shadow reader and the dual writer. ``submit`` never blocks and
    never raises; ``stop`` drains the queue, joins, then closes the owned Memory.
    Subclasses set their own attributes first, then call ``__init__`` (it starts
    the thread), and implement ``_handle``.
    """

    _label = "worker"

    def __init__(self, owned_memory) -> None:
        self._owned = owned_memory
        self._queue: "queue.Queue" = queue.Queue(maxsize=_SHADOW_QUEUE_MAX)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"prospecta-{self._label}")
        self._thread.start()

    def submit(self, job: Dict[str, Any]) -> None:
        """Queue a job; never blocks and never raises."""
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            try:
                self._on_full(job)
            except Exception:
                logger.exception("prospecta %s queue-full handler failed", self._label)
        except Exception:
            logger.exception("prospecta %s submit failed", self._label)

    def stop(self) -> None:
        try:
            self._queue.put(_SHADOW_STOP, timeout=_SHADOW_DRAIN_TIMEOUT_S)
        except queue.Full:
            logger.error("prospecta %s queue stayed full for %ss at shutdown",
                         self._label, _SHADOW_DRAIN_TIMEOUT_S)
        self._thread.join(timeout=_SHADOW_DRAIN_TIMEOUT_S)
        if self._thread.is_alive():
            logger.error("prospecta %s worker still busy after %ss; leaving its "
                         "connection open (daemon thread dies with the process)",
                         self._label, _SHADOW_DRAIN_TIMEOUT_S)
            try:
                self._abandon()
            except Exception:
                logger.exception("prospecta %s could not save its pending jobs", self._label)
            return
        try:
            self._owned.close()
        except Exception:
            logger.exception("prospecta %s memory close failed", self._label)

    def _describe(self, job: Dict[str, Any]) -> str:
        return repr(job)

    def _on_full(self, job: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _abandon(self) -> None:
        """Called when the worker did not finish in time (queue still holds jobs)."""

    def _handle(self, job: Dict[str, Any]) -> None:
        raise NotImplementedError

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _SHADOW_STOP:
                return
            try:
                self._handle(job)
            except Exception:
                logger.exception("prospecta %s job failed: %s", self._label, self._describe(job))


class _ShadowReader(_BackgroundWorker):
    """Replays the old bank's recalls against a second (new) bank, off-path.

    The library's ``Memory(shadow_bank_id=...)`` only shadows ``Memory.recall``
    (this provider calls ``search`` / ``recall_synth``) and opens a fresh
    connection pool per call, so this reuses its pair format (two
    ``recall_events`` rows sharing ``trace.shadow.id``, with ``jaccard`` over
    original document ids) but keeps ONE shadow ``Memory`` for the provider's
    life and runs it on ONE daemon worker thread behind a bounded queue.
    Nothing here raises into, or waits on, the caller.
    """

    _label = "shadow"

    def __init__(self, memory, shadow_memory, *, bank_id: str, shadow_bank_id: str,
                 mode: str = "search") -> None:
        self._memory = memory            # old-bank Memory: writes both events, resolves ids
        self._shadow = shadow_memory     # new-bank Memory, created once, reused
        self._bank_id = bank_id
        self._shadow_bank_id = shadow_bank_id
        self._mode = mode                # "search" | "full"
        self._tally = _ab.AbstentionTally()
        super().__init__(shadow_memory)

    def _on_full(self, job: Dict[str, Any]) -> None:
        logger.error(
            "prospecta shadow queue full (%d pending); dropped shadow pair for %s %r",
            _SHADOW_QUEUE_MAX, job.get("tool"), job.get("message"),
        )

    def _describe(self, job: Dict[str, Any]) -> str:
        return f"{job.get('tool')} {job.get('message')!r}"

    def _handle(self, job: Dict[str, Any]) -> None:
        self._record(job)

    def _lexical_count(self, job: Dict[str, Any]) -> Optional[int]:
        """How many hits a plain lexical search of the old bank has for this question (None:
        unknown): the evidence for the false-abstain rate. It is the library's own lexical mode
        (Postgres full text), asked here on the worker thread, off the old path."""
        try:
            return len(self._memory.search(job["message"], mode="lexical", limit=10))
        except Exception:
            logger.exception("prospecta shadow lexical count failed")
            return None

    def _record(self, job: Dict[str, Any]) -> None:
        from prospecta._shadow import _origin_ids
        from prospecta.memory import _serialize_recall_results

        depth_kw = {"depth": job["depth"]} if job.get("depth") else {}
        queries = list(job["queries"])
        synthesis = None
        flat: list = []
        error = None
        t0 = time.monotonic()
        try:
            if self._mode == "full" and job["tool"] != "prospecta_search":
                res = self._shadow.recall_synth(
                    job["message"], limit=job["limit"], mode=job["mode"], **depth_kw)
                flat = list(res.sources or [])
                synthesis = res.synthesis
                queries = [q.text for q in (res.queries or [])] or queries
            else:
                for q in queries:
                    flat.extend(self._shadow.search(
                        q, mode=job["mode"], limit=job["limit"], **depth_kw))
        except Exception as e:
            logger.exception("prospecta shadow %s against %s failed for %r",
                             job["tool"], self._shadow_bank_id, job["message"])
            error = f"{type(e).__name__}: {e}"
        new_ms = int((time.monotonic() - t0) * 1000)

        old_results = job["results"]
        try:
            old_origin = _origin_ids(self._memory, self._bank_id, old_results)
            new_origin = _origin_ids(self._memory, self._shadow_bank_id, flat) if error is None else []
        except Exception:
            logger.exception("prospecta shadow could not resolve document ids")
            return
        old_set, new_set = set(old_origin), set(new_origin)
        union = old_set | new_set
        kw_n = self._lexical_count(job)
        old_found = bool(old_results)
        new_found = None if error else bool(flat)  # None: the new side failed, not an abstention
        counts = self._tally.record(old_found=old_found, new_found=new_found, kw_found=bool(kw_n))
        overlap = {
            "old_origin_document_ids": old_origin,
            "new_origin_document_ids": new_origin,
            "jaccard": (len(old_set & new_set) / len(union)) if union else 1.0,
            # counts only: abstention is "returned nothing"; the false-abstain rate
            # (abstained where the other side or the keyword leg found something,
            # over answerable questions) always sits beside it
            "old_n": len(old_results), "new_n": None if error else len(flat), "kw_n": kw_n,
            "abstain": counts,
        }
        logger.info("prospecta shadow pair tool=%s old_n=%d new_n=%s kw_n=%s jaccard=%.2f | %s",
                    job["tool"], len(old_results), "err" if error else len(flat),
                    "n/a" if kw_n is None else kw_n, overlap["jaccard"], _ab.format_tally(counts))
        shadow_id = str(uuid.uuid4())
        for role, bank, peer, qs, results, syn, ms, err in (
            ("old", self._bank_id, self._shadow_bank_id, job["queries"], old_results,
             job.get("synthesis"), job["old_ms"], None),
            ("new", self._shadow_bank_id, self._bank_id, queries, flat, synthesis, new_ms, error),
        ):
            try:
                self._memory._tracer("recall", {
                    "bank_id": bank,
                    "queries": list(qs),
                    "mode": job["mode"],
                    "n_results": len(results),
                    "duration_ms": ms,
                    "trace": {"shadow": {
                        "id": shadow_id, "role": role, "peer_bank": peer, "error": err,
                        "tool": job["tool"], "shadow_mode": self._mode,
                        "old_ms": job["old_ms"], "new_ms": new_ms,
                        **overlap,
                    }},
                    "results": _serialize_recall_results(results),
                    "synthesis": syn,
                    "depth": job.get("depth"),
                })
            except Exception:
                logger.exception("prospecta shadow could not store the %s event", role)


class _DualWriter(_BackgroundWorker):
    """Writes each note the main bank accepted into a second bank, off-path.

    One ``Memory`` for the second bank (its own embedder, linker on, chunk
    children on), one daemon thread, the same bounded queue as the shadow
    reader. A note whose content hash is already in the second bank is skipped,
    so the catch-up copy and the dual write cannot duplicate. A failed (or
    queue-overflowing) write is appended to a JSONL retry file; the file is
    replayed by the worker right after start.
    """

    _label = "dual-write"

    def __init__(self, memory, *, bank_id: str, retry_path: Path) -> None:
        self._bank_id = bank_id
        self._retry_path = Path(retry_path)
        self._retry_lock = threading.Lock()
        super().__init__(memory)
        self.submit({"replay": True})

    def _describe(self, job: Dict[str, Any]) -> str:
        return f"{job.get('source')!r}"

    def _handle(self, job: Dict[str, Any]) -> None:
        if job.get("replay"):
            self._replay()
        else:
            self._write(job)

    def _write(self, job: Dict[str, Any]) -> None:
        try:
            content = job["content"]
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            with self._owned._pool.connection() as conn:
                found = conn.execute(
                    "SELECT 1 FROM documents WHERE bank_id = %s AND content_hash = %s",
                    (self._bank_id, digest)).fetchone()
            if found:
                logger.debug("prospecta dual-write: %s already in %s", job.get("source"), self._bank_id)
                return
            kw = {k: job[k] for k in ("source", "tags", "index_text", "metadata") if job.get(k) is not None}
            self._owned.retain(content=content, child_chunks=True, **kw)
        except Exception as e:
            logger.exception("prospecta dual-write to %s failed for source %r; queued for retry in %s",
                             self._bank_id, job.get("source"), self._retry_path)
            self._to_retry(job, f"{type(e).__name__}: {e}")

    def _to_retry(self, job: Dict[str, Any], error: str) -> None:
        try:
            line = json.dumps({**job, "error": error, "failed_at": time.time()}, ensure_ascii=False)
            with self._retry_lock:
                self._retry_path.parent.mkdir(parents=True, exist_ok=True)
                with self._retry_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.exception("prospecta dual-write could not save a note to the retry file %s; "
                             "the note was: %r", self._retry_path, job)

    def _on_full(self, job: Dict[str, Any]) -> None:
        logger.error("prospecta dual-write queue full (%d pending); note %r goes to the retry file",
                     _SHADOW_QUEUE_MAX, job.get("source"))
        self._to_retry(job, "queue full")

    def _abandon(self) -> None:
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not _SHADOW_STOP and not job.get("replay"):
                self._to_retry(job, "shutdown before the write")

    def _replay(self) -> None:
        """Retry every saved note once; those that fail again land in a fresh retry file."""
        work = self._retry_path.with_name(self._retry_path.name + ".replaying")
        with self._retry_lock:  # a crash mid-replay leaves .replaying; it is merged, not lost
            if self._retry_path.exists():
                with work.open("a", encoding="utf-8") as w:
                    w.write(self._retry_path.read_text(encoding="utf-8"))
                self._retry_path.unlink()
        if not work.exists():
            return
        lines = [ln for ln in work.read_text(encoding="utf-8").splitlines() if ln.strip()]
        logger.info("prospecta dual-write replaying %d saved note(s) into %s", len(lines), self._bank_id)
        for ln in lines:
            try:
                job = json.loads(ln)
                if not isinstance(job.get("content"), str):
                    raise ValueError("no content")
            except Exception:
                logger.exception("prospecta dual-write retry line is unreadable, skipped: %r", ln)
                continue
            self._write(job)
        work.unlink()


class ProspectaProvider(MemoryProvider):
    """Hermes MemoryProvider wrapping prospecta.Memory."""

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self._memory = None  # prospecta.Memory
        self._database_url: str = ""
        self._session_id: str = ""
        self._hermes_home: Optional[Path] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._compose_project: Optional[str] = None
        self._llm = None  # built LLM callable; None means no-LLM mode
        self._shadow: Optional[_ShadowReader] = None  # None = shadow reads OFF
        self._dual: Optional[_DualWriter] = None  # None = dual write OFF
        self._linked_writes = False
        self._conn_watch: Optional[_ConnectionWatch] = None
        self._date_gate_held = False
        # recall.prefetch (the written recall contract): settings read per session at initialize;
        # per-session state kept from sync_turn; the injectable Jev transport (the library's writer is Memory's llm)
        self._prefetch_cfg: Optional[Any] = None  # _pc.Settings
        self._state_lock = threading.Lock()
        self._turn_sources: "OrderedDict[str, list]" = OrderedDict()  # session -> sources of its last two retained turns
        self._exchanges: "OrderedDict[str, tuple]" = OrderedDict()  # session -> (assistant's last answer, turns)
        self._ahead: Dict[str, Any] = {}  # session -> _pc.AheadEntry
        self._ahead_running: set = set()  # (session, key) pairs being prepared
        self._jev_transport: Optional[Any] = None   # tests / hosts: a (request, timeout) -> response callable
        self._last_brief: Optional[Any] = None      # the last contract run's record (no text is logged)
        self._skill_turn: Optional[str] = None      # the shape of this turn when it is a skill/cron turn (on_turn_start)

    # ---- identity ----

    @property
    def name(self) -> str:
        return "prospecta"

    # ---- availability ----

    def is_available(self) -> bool:
        """No network calls. Check prospecta importable + configured substrate.

        Production/platform installs should provide PROSPECTA_DATABASE_URL
        (preferred) or database_url config. Embedded docker is dev-only and
        requires explicit opt-in via PROSPECTA_ALLOW_EMBEDDED=1.
        """
        try:
            import prospecta  # noqa: F401
        except ImportError:
            return False
        if (
            os.environ.get("PROSPECTA_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
        ):
            return True
        try:
            hermes_home = os.environ.get("HERMES_HOME")
            if hermes_home:
                cfg_path = Path(hermes_home) / "prospecta.json"
                if cfg_path.exists():
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    if cfg.get("database_url"):
                        return True
        except Exception:
            pass
        if os.environ.get("PROSPECTA_ALLOW_EMBEDDED") != "1":
            return False
        import shutil
        return shutil.which("docker") is not None

    def unavailable_reason(self) -> str:
        """Why ``is_available()`` is False, and the fix (shown at ERROR by the host).

        Mirrors ``is_available()`` step by step; empty string when available.
        """
        try:
            import prospecta  # noqa: F401
        except ImportError:
            return (
                "The 'prospecta' Python package is not importable in this venv "
                "(a venv rebuild/sync drops it; it is not on PyPI). Fix: run "
                "ops/prospecta/ensure-venv.sh from the hermes-agent checkout "
                "(installs the pinned 'hermes-agent[prospecta]' extra), then restart "
                "the gateway."
            )
        if self.is_available():
            return ""
        return (
            "No Prospecta database is configured: set PROSPECTA_DATABASE_URL in this "
            "profile's .env (or database_url in $HERMES_HOME/prospecta.json); embedded "
            "docker is dev-only (PROSPECTA_ALLOW_EMBEDDED=1)."
        )

    # ---- config ----

    def _config_path(self) -> Path:
        assert self._hermes_home is not None
        return self._hermes_home / "prospecta.json"

    def _load_config(self) -> Dict[str, Any]:
        if self._hermes_home is None:
            return {}
        path = self._config_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read prospecta config %s: %s", path, e)
            return {}

    def _resolve_database_url(self) -> str:
        """Resolve the Prospecta Postgres URL.

        Prefer Prospecta-specific env over generic DATABASE_URL so host-wide
        platform config does not collide with unrelated app databases.
        Embedded mode is dev-only and must be explicitly enabled.
        """
        database_url = (
            os.environ.get("PROSPECTA_DATABASE_URL")
            or self._config.get("database_url")
            or os.environ.get("DATABASE_URL")
            or ""
        ).strip()
        if database_url:
            return database_url
        if os.environ.get("PROSPECTA_ALLOW_EMBEDDED") == "1":
            return self._start_embedded_substrate()
        raise RuntimeError(
            "Prospecta requires PROSPECTA_DATABASE_URL (preferred), "
            "database_url in prospecta.json, or DATABASE_URL. Embedded "
            "docker fallback is dev-only; set PROSPECTA_ALLOW_EMBEDDED=1 "
            "to opt in."
        )

    def _resolve_bank_id(self) -> str:
        """Resolve bank id with platform env taking precedence."""
        return (
            os.environ.get("PROSPECTA_BANK_ID")
            or self._config.get("bank_id")
            or "default"
        )

    def _resolve_shadow_bank_id(self) -> str:
        """Shadow bank id; empty means OFF. PROSPECTA_SHADOW_BANK (even set to
        empty, which forces OFF) wins over config."""
        return self._resolve_setting("PROSPECTA_SHADOW_BANK", "shadow_bank_id")

    def _resolve_setting(self, env_name: str, key: str) -> str:
        """Env (even set to empty) wins over config."""
        env = os.environ.get(env_name)
        if env is not None:
            return env.strip()
        return str(self._config.get(key) or "").strip()

    def _resolve_max_connections(self) -> int:
        """Connections each Memory may hold open (default 4)."""
        try:
            n = int(self._config.get("max_connections") or _DEFAULT_MAX_CONNECTIONS)
        except (TypeError, ValueError):
            n = _DEFAULT_MAX_CONNECTIONS
        return n if n >= 1 else _DEFAULT_MAX_CONNECTIONS

    def _resolve_pool_idle_seconds(self) -> int:
        """Seconds a pooled connection may sit unused before it is closed
        (default 300). PROSPECTA_POOL_IDLE_SECONDS env wins over config."""
        raw = os.environ.get("PROSPECTA_POOL_IDLE_SECONDS")
        if raw is None or not raw.strip():
            raw = self._config.get("pool_idle_seconds")
        try:
            n = int(raw or _DEFAULT_POOL_IDLE_SECONDS)
        except (TypeError, ValueError):
            n = _DEFAULT_POOL_IDLE_SECONDS
        return n if n >= 1 else _DEFAULT_POOL_IDLE_SECONDS

    def _connection_watch_enabled(self) -> bool:
        """Watch the database's connection count? Default on."""
        raw = os.environ.get("PROSPECTA_CONNECTION_WATCH")
        if raw is None or not raw.strip():
            raw = self._config.get("connection_watch")
        return ("" if raw is None else str(raw).strip().lower()) not in ("0", "false", "no", "off")

    def _resolve_connection_watch_percent(self) -> int:
        """Share of max_connections (1-100) at which the watch warns (default 70)."""
        try:
            n = int(self._config.get("connection_watch_percent") or _DEFAULT_CONNECTION_WATCH_PERCENT)
        except (TypeError, ValueError):
            n = _DEFAULT_CONNECTION_WATCH_PERCENT
        return n if 1 <= n <= 100 else _DEFAULT_CONNECTION_WATCH_PERCENT

    # ---- recall options ----

    def _recall_setting(self, env_name: str, key: str) -> Any:
        """A ``recall.*`` option: env (when non-empty) wins; then the nested
        ``{"recall": {key: ...}}``; then the flat ``"recall.key"`` that
        ``hermes memory setup`` writes. None when unset."""
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            return raw.strip()
        nested = self._config.get("recall")
        if isinstance(nested, dict) and nested.get(key) not in (None, ""):
            return nested[key]
        flat = self._config.get(f"recall.{key}")
        return None if flat in (None, "") else flat

    def _recall_date_cue_gate(self) -> bool:
        """Skip the filter-extraction model call for a question with no date cue?
        Default on; PROSPECTA_DATE_CUE_GATE env or ``recall.date_cue_gate`` turns it off."""
        raw = self._recall_setting("PROSPECTA_DATE_CUE_GATE", "date_cue_gate")
        return ("" if raw is None else str(raw).strip().lower()) not in ("0", "false", "no", "off")

    def _recall_date_timeout(self) -> float:
        """Hard cap (s) on the date-extraction model call. Default 6; 0 disables.
        PROSPECTA_DATE_TIMEOUT env or ``recall.date_timeout_seconds``."""
        raw = self._recall_setting("PROSPECTA_DATE_TIMEOUT", "date_timeout_seconds")
        if raw is None:
            return _dcg.DEFAULT_TIMEOUT
        try:
            v = float(str(raw).strip())
            if v < 0 or v != v:
                raise ValueError(raw)
            return v
        except ValueError:
            logger.warning("prospecta: recall.date_timeout_seconds %r is not a number >= 0; using %s",
                           raw, _dcg.DEFAULT_TIMEOUT)
            return _dcg.DEFAULT_TIMEOUT

    # ---- recall.prefetch (the written recall contract) ----

    def _prefetch_config(self, key: str) -> Any:
        """A ``recall.prefetch.*`` option from the profile only: ``{"recall": {"prefetch": {key: ...}}}``,
        else the flat ``"recall.prefetch.key"`` that ``hermes memory setup`` writes. None when unset."""
        nested = self._config.get("recall")
        nested = nested.get("prefetch") if isinstance(nested, dict) else None
        if isinstance(nested, dict) and nested.get(key) not in (None, ""):
            return nested[key]
        flat = self._config.get(f"recall.prefetch.{key}")
        return None if flat in (None, "") else flat

    def _prefetch_setting(self, env_name: str, key: str) -> Any:
        """A ``recall.prefetch.*`` option: env (when non-empty) wins; then the profile
        (``_prefetch_config``). None when unset."""
        raw = os.environ.get(env_name)
        if raw is not None and raw.strip():
            return raw.strip()
        return self._prefetch_config(key)

    def _prefetch_number(self, key: str, env_name: str, default: float, *, whole: bool = False,
                         positive: bool = False) -> float:
        raw = self._prefetch_setting(env_name, key)
        if raw is None or isinstance(raw, bool):
            return default
        try:
            v = float(str(raw).strip())
            if v < 0 or v != v or v == float("inf") or (whole and v != int(v)) or (positive and v == 0):
                raise ValueError(raw)
            return v
        except (TypeError, ValueError):
            logger.warning("prospecta: recall.prefetch.%s %r is not a %s %s 0; using %s",
                           key, raw, "whole number" if whole else "number", ">" if positive else ">=", default)
            return default

    def _profile_model(self, *values: Any) -> Optional[str]:
        """The first of ``values`` (a profile setting) that is a non-blank string, stripped; else None.
        The plugin holds the profile's model ids only: when it has none, the library resolves the model
        (the bank's recall_config, then the PROSPECTA_*_MODEL environment variable, then its own default),
        so the order and the defaults live in one place."""
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _decision_model(self) -> Optional[str]:
        """The profile's decision model (Stage 0 of the contract prefetch, and the library's Stage 3
        gate), or None: the gate then uses the library's (``prospecta.stages.JEV_MODEL``)."""
        return self._profile_model(self._config.get("decision_model"))

    def _reranker_model(self) -> Optional[str]:
        """The profile's Stage 2 cross-encoder, given to the library, or None (the library resolves it)."""
        return self._profile_model(self._config.get("reranker_model"))

    def _query_writer_model(self) -> Optional[str]:
        """The profile's query-writer model for the contract prefetch, or None (the library resolves it):
        ``recall.prefetch.query_writer_model``, else the profile-wide ``query_writer_model``, else the
        deprecated ``recall.prefetch.question_model`` (warned about at initialise)."""
        return self._profile_model(self._prefetch_config("query_writer_model"), self._config.get("query_writer_model"),
                                   self._prefetch_config("question_model"))

    def _recall_stage3(self) -> bool:
        """Answer ``prospecta_recall`` through the library's ``ask()`` with the Stage 3 sufficiency
        gate? Default off. PROSPECTA_RECALL_STAGE3 env or ``recall.stage3``."""
        raw = self._recall_setting("PROSPECTA_RECALL_STAGE3", "stage3")
        return ("" if raw is None else str(raw).strip().lower()) in ("1", "true", "yes", "on")

    def _warn_renamed_settings(self) -> None:
        """Old contract-prefetch settings, one warning line each, once a session (at initialise).
        ``recall.prefetch.question_model`` is a deprecated alias of ``recall.prefetch.query_writer_model``
        and still works. The others no longer apply: the decision-model keys moved to ``decision_model``,
        and the writer and search time limits went with the stages the library owns now."""
        if self._prefetch_config("question_model") is not None:
            logger.warning("prospecta: recall.prefetch.question_model is deprecated; set "
                           "recall.prefetch.query_writer_model (the value is used for now)")
        old = {"recall.prefetch.decision_model": "decision_model",
               "recall.prefetch.writer_timeout_seconds": None, "recall.prefetch.search_timeout_seconds": None}
        old_env = {"PROSPECTA_PREFETCH_DECISION_MODEL": "PROSPECTA_DECISION_MODEL",
                   "PROSPECTA_PREFETCH_QUESTION_MODEL": "PROSPECTA_QUERY_WRITER_MODEL",
                   "PROSPECTA_PREFETCH_WRITER_TIMEOUT": None, "PROSPECTA_PREFETCH_SEARCH_TIMEOUT": None}
        for key, new in old.items():
            if self._prefetch_config(key.rsplit(".", 1)[1]) is not None:
                logger.warning("prospecta: %s is ignored now; %s", key, f"set {new}" if new else
                               "the library owns that stage and runs it inside recall.prefetch.deadline_seconds")
        for name, new in old_env.items():
            if os.environ.get(name, "").strip():
                logger.warning("prospecta: %s is ignored now; %s", name, f"set {new}" if new else
                               "the library owns that stage and runs it inside PROSPECTA_PREFETCH_DEADLINE")

    def _resolve_prefetch_settings(self) -> Any:
        """The ``recall.prefetch.*`` options. Read at initialize (each session), so flipping the
        mode in prospecta.json or the environment needs no gateway restart."""
        raw_mode = self._prefetch_setting("PROSPECTA_PREFETCH_MODE", "mode")
        mode = _pc.DEFAULT_MODE if raw_mode is None else str(raw_mode).strip().lower()
        if mode not in _pc.MODES:
            logger.warning("prospecta: recall.prefetch.mode %r is not one of %s; using %r",
                           raw_mode, list(_pc.MODES), _pc.DEFAULT_MODE)
            mode = _pc.DEFAULT_MODE
        contract = self._prefetch_setting("PROSPECTA_PREFETCH_CONTRACT", "contract_path")
        # below core's own wait (8 s) so a slow turn is ours to end, not core's to time out; the library
        # is given what is left of it for the whole expansion (the writer, every search, every rerank)
        deadline = self._prefetch_number("deadline_seconds", "PROSPECTA_PREFETCH_DEADLINE", _pc.DEADLINE_S)
        return _pc.Settings(
            mode=mode,
            decision_model=self._decision_model(),
            query_writer_model=self._query_writer_model(),
            contract_path=str(contract) if contract else "",
            max_tokens=int(self._prefetch_number("max_tokens", "PROSPECTA_PREFETCH_MAX_TOKENS", 0, whole=True)),
            deadline_s=min(deadline, _pc.MAX_DEADLINE_S) if deadline > 0 else _pc.DEADLINE_S,
            # the one stage limit the plugin still owns (also capped by what is left of the deadline)
            decision_timeout_s=self._prefetch_number("decision_timeout_seconds", "PROSPECTA_PREFETCH_DECISION_TIMEOUT",
                                                     _pc.DECISION_TIMEOUT_S, positive=True),
            paste_chars=int(self._prefetch_number("paste_chars", "PROSPECTA_PREFETCH_PASTE_CHARS",
                                                  _pc.PASTE_CHARS, whole=True)) or _pc.PASTE_CHARS,
            drop_echoes=str(self._prefetch_setting("PROSPECTA_PREFETCH_DROP_ECHOES", "drop_echoes")
                            ).strip().lower() not in ("false", "0", "no", "off"),
        )

    def _prefetch_settings(self) -> Any:
        return self._prefetch_cfg if self._prefetch_cfg is not None else self._resolve_prefetch_settings()

    def _prefetch_enabled(self) -> bool:
        """The master switch (default OFF): ``prefetch_enabled`` or the PROSPECTA_PREFETCH env."""
        return (
            bool(self._config.get("prefetch_enabled"))
            and str(self._config.get("prefetch_enabled")).lower() not in ("false", "0", "")
        ) or bool(os.environ.get("PROSPECTA_PREFETCH"))

    def _start_connection_watch(self) -> None:
        if not self._connection_watch_enabled():
            return
        try:
            self._conn_watch = _ConnectionWatch(
                self._database_url, self._hermes_home, self._resolve_connection_watch_percent())
            self._conn_watch.start()
        except Exception:
            self._conn_watch = None
            logger.exception("Prospecta connection watch could not start")

    def _resolve_linked_writes(self) -> bool:
        return self._resolve_setting("PROSPECTA_LINKED_WRITES", "linked_writes").lower() in (
            "1", "true", "yes", "on")

    def _resolve_dual_write_bank_id(self) -> str:
        """Dual-write bank id; empty means OFF. PROSPECTA_DUAL_WRITE_BANK (even set
        to empty, which forces OFF) wins over config."""
        return self._resolve_setting("PROSPECTA_DUAL_WRITE_BANK", "dual_write_bank_id")

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "database_url",
                "description": "Postgres connection URL. Prefer PROSPECTA_DATABASE_URL env for host/prod deployments; leave empty only when explicitly enabling embedded dev mode.",
                "default": "",
                "secret": False,
            },
            {
                "key": "bank_id",
                "description": "Bank identifier (multi-tenant key). Default 'default'.",
                "default": "default",
            },
            {
                "key": "max_connections",
                "description": "Most database connections each Memory (main, shadow, dual write) keeps open. Default 4.",
                "default": "4",
            },
            {
                "key": "pool_idle_seconds",
                "description": "Seconds a pooled database connection may sit unused before it is closed (a new one opens on next use). Default 300. PROSPECTA_POOL_IDLE_SECONDS env overrides.",
                "default": "300",
            },
            {
                "key": "connection_watch",
                "description": "Warn loudly (ERROR log and a notice on the next turn) when the database nears its connection limit: 'true' / 'false'. Default on. PROSPECTA_CONNECTION_WATCH env overrides.",
                "default": "true",
            },
            {
                "key": "connection_watch_percent",
                "description": "Share of the database's max_connections (1-100) in use at which connection_watch warns. Default 70.",
                "default": "70",
            },
            {
                "key": "embedder_kind",
                "description": "Embedder provider: 'litellm', 'sentence_transformers', 'openai'.",
                "default": "litellm",
                "choices": ["litellm", "sentence_transformers", "openai"],
            },
            {
                "key": "embedder_model",
                "description": "Embedding model id. Provider-specific default if empty.",
                "default": "",
            },
            {
                "key": "llm_model",
                "description": "LiteLLM model id for recall synthesis and query formulation (e.g. 'anthropic/claude-haiku-4-5'). Falls back to PROSPECTA_LLM_MODEL env, then prospecta.defaults default.",
                "default": "",
            },
            {
                "key": "embedding_dim",
                "description": "Vector dimensionality. Must match embedder (1536 for text-embedding-3-small; 384 for all-MiniLM-L6-v2).",
                "default": "1536",
            },
            {
                "key": "prefetch_enabled",
                "description": "Recall before each turn (see recall.prefetch.mode for what runs). Default off.",
                "default": "false",
            },
            {
                "key": "shadow_bank_id",
                "description": "Shadow-read bank: every search/recall is also run against it and the old/new pair is logged to recall_events. Empty = OFF (default). PROSPECTA_SHADOW_BANK env overrides (empty forces off).",
                "default": "",
            },
            {
                "key": "shadow_embed_model",
                "description": "Embedding model for the shadow bank.",
                "default": _SHADOW_DEFAULT_EMBED_MODEL,
            },
            {
                "key": "shadow_embedding_dim",
                "description": "Embedding dimensionality of the shadow bank.",
                "default": "1536",
            },
            {
                "key": "linked_writes",
                "description": "New notes get entities and links like the migrated bank (the library's linker, plus chunk children). Needs OPENROUTER_API_KEY. Default off. PROSPECTA_LINKED_WRITES env overrides.",
                "default": "false",
            },
            {
                "key": "dual_write_bank_id",
                "description": "Second bank that every note is also written to (own embedder, linker on), in the background. Must already exist. Empty = OFF (default). PROSPECTA_DUAL_WRITE_BANK env overrides (empty forces off).",
                "default": "",
            },
            {
                "key": "dual_write_embed_model",
                "description": "Embedding model for the dual-write bank.",
                "default": _SHADOW_DEFAULT_EMBED_MODEL,
            },
            {
                "key": "dual_write_embedding_dim",
                "description": "Embedding dimensionality of the dual-write bank.",
                "default": "1536",
            },
            {
                "key": "dual_write_retry_path",
                "description": "JSONL file for notes whose second write failed; replayed at the next initialize. Default: prospecta-dual-write-retry.jsonl beside prospecta.json.",
                "default": "",
            },
            {
                "key": "timezone",
                "description": "IANA timezone (e.g. America/Los_Angeles) whose calendar day the library saves as each new note's date (documents.created_on; handed to it as PROSPECTA_TIMEZONE) and the day the contract prefetch counts time scopes back from. Default America/Los_Angeles.",
                "default": _DEFAULT_TIMEZONE,
            },
            {
                "key": "hash_precheck",
                "description": "Before saving a note, look up whether the bank already holds the same text, and skip the write (and its embedding calls) if so: 'true' / 'false'. Empty (default) = on. PROSPECTA_HASH_PRECHECK env overrides.",
                "default": "",
            },
            {
                "key": "tool_guide",
                "description": "Add a short guide to the memory tools to the system prompt: 'true' / 'false'. Empty (default) = on. PROSPECTA_TOOL_GUIDE env overrides.",
                "default": "",
            },
            {
                "key": "recall.date_cue_gate",
                "description": "Call the filter-extraction model on a search only when the question has a date cue (a month, weekday, 'ago', 'last week', a year, an ISO date, ...); otherwise search with no date filter and no model call. 'true' / 'false'. Default on. PROSPECTA_DATE_CUE_GATE env overrides.",
                "default": "true",
            },
            {
                "key": "recall.date_timeout_seconds",
                "description": "Hard wall-clock cap in seconds on the date-extraction model call; on timeout or error the search uses the regex date filter. Default 6. 0 disables. PROSPECTA_DATE_TIMEOUT env overrides.",
                "default": "6",
            },
            {
                "key": "decision_model",
                "description": "The Stage 0 decision model: a typed System One model called through OpenRouter with the profile's key. It decides whether memory is needed and for what period of time (contract prefetch), and is the library's Stage 3 gate model. Unset (default): the library's choice, PROSPECTA_DECISION_MODEL else typesafe/jev-1.14. The profile value wins over the environment.",
                "default": "",
            },
            {
                "key": "query_writer_model",
                "description": "The small text model that expands the contract prefetch's message into searches (LiteLLM id, called with the profile's key; also read as recall.prefetch.query_writer_model, which wins). Unset (default): the library's choice, the bank's recall_config.models.query_writer_model, else PROSPECTA_QUERY_WRITER_MODEL, else openrouter/google/gemini-2.5-flash-lite. The profile value wins over the environment.",
                "default": "",
            },
            {
                "key": "reranker_model",
                "description": "The Stage 2 cross-encoder the library reranks each search with (OpenRouter's rerank endpoint, the profile's key); used where the bank's recall_config sets rerank.stage to cross_encoder. Unset (default): the library's choice, the bank's recall_config.rerank.model, else PROSPECTA_RERANKER_MODEL, else qwen/qwen3-reranker-8b. The profile value wins over the environment.",
                "default": "",
            },
            {
                "key": "recall.stage3",
                "description": "Answer prospecta_recall through the library's ask(): Stage 1 + Stage 2, a grounded answer with [n] citations, and the Stage 3 sufficiency gate (the decision model checks the evidence and one broadened search runs when it falls short). 'true' / 'false'. Default off (prospecta_recall uses recall_synth). PROSPECTA_RECALL_STAGE3 env overrides.",
                "default": "false",
            },
            {
                "key": "recall.prefetch.mode",
                "description": "The pre-turn recall (only when prefetch_enabled is true): 'off' (default: the old prefetch, one model-written answer) or 'contract' (the written recall contract: one decision per message, then the library's query expansion into parallel searches through Stage 1 + Stage 2, one cited block, no synthesis). Read when a session starts. PROSPECTA_PREFETCH_MODE env overrides.",
                "default": _pc.DEFAULT_MODE,
                "choices": list(_pc.MODES),
            },
            {
                "key": "recall.prefetch.contract_path",
                "description": "The recall contract file (the Jev questions and the block's words). Default: the bundled contract/recall-contract-v1.md. PROSPECTA_PREFETCH_CONTRACT env overrides.",
                "default": "",
            },
            {
                "key": "recall.prefetch.max_tokens",
                "description": "Safety valve on the contract prefetch block (the library's): leave out (whole, never cut) the entries that would take it past this many estimated tokens. 0 (default) = not set: the block is sized by relevance alone, unless the bank's recall_config.expansion.max_tokens sets a valve. PROSPECTA_PREFETCH_MAX_TOKENS env overrides.",
                "default": "0",
            },
            {
                "key": "shadow_mode",
                "description": "'search' (default): shadow retrieval only, embedding cost. 'full': also shadow recall_synth (LLM, ~0.2-0.4 USD per recall).",
                "default": "search",
                "choices": ["search", "full"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "prospecta.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        existing.update(values)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    # ---- substrate ----

    def _start_embedded_substrate(self) -> str:
        """Spin up bundled docker-compose Postgres. Returns DATABASE_URL."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            raise RuntimeError(
                "Prospecta requires DATABASE_URL set OR docker installed for embedded substrate. "
                "Install docker or set DATABASE_URL=postgres://..."
            )

        compose_file = Path(__file__).parent / "docker-compose.yml"
        if not compose_file.exists():
            raise RuntimeError(f"Bundled docker-compose.yml missing at {compose_file}")

        assert self._hermes_home is not None
        profile_slug = self._hermes_home.name or "default"
        project = f"prospecta-{profile_slug}"
        self._compose_project = project

        # Pass POSTGRES_PASSWORD via env if compose file references it
        env = os.environ.copy()
        env.setdefault("POSTGRES_PASSWORD", "prospecta")
        env.setdefault("PROSPECTA_EMBEDDED_PORT", os.environ.get("PROSPECTA_EMBEDDED_PORT", "5432"))

        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "-p", project, "up", "-d"],
            capture_output=True, text=True, timeout=120, env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker compose up failed: {result.stderr}")

        # Wait briefly for ps to report running.
        deadline = time.time() + 60
        while time.time() < deadline:
            check = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "-p", project,
                 "ps", "--status=running", "-q"],
                capture_output=True, text=True, env=env,
            )
            if check.stdout.strip():
                break
            time.sleep(1)

        port = os.environ.get("PROSPECTA_EMBEDDED_PORT", "5432")
        return f"postgresql://prospecta:prospecta@localhost:{port}/prospecta"

    # ---- embedder + llm ----

    def _build_embedder(self):
        kind = (self._config.get("embedder_kind") or "litellm").strip().lower()
        model = (self._config.get("embedder_model") or "").strip() or None

        if kind == "litellm":
            try:
                from prospecta.defaults import make_default_embedder
            except ImportError as e:
                raise RuntimeError(
                    "embedder_kind='litellm' requires `pip install 'prospecta[defaults]'`"
                ) from e
            return make_default_embedder(model)
        if kind == "sentence_transformers":
            from prospecta.embed import sentence_transformers
            return sentence_transformers(model or "all-MiniLM-L6-v2")
        if kind == "openai":
            from prospecta.embed import openai as openai_embed
            return openai_embed(model=model or "text-embedding-3-small")
        raise RuntimeError(f"Unknown embedder_kind: {kind!r}")

    def _build_shadow_embedder(self, model: str, dimensions: int):
        from prospecta.defaults import make_default_embedder
        return make_default_embedder(model, dimensions=dimensions)

    def _start_shadow(self, bank_id: str) -> None:
        """Create the shadow reader once; any problem leaves shadow OFF, loudly."""
        shadow_bank = self._resolve_shadow_bank_id()
        if not shadow_bank:
            return
        try:
            if shadow_bank == bank_id:
                raise ValueError(f"shadow_bank_id {shadow_bank!r} must differ from bank_id")
            mode = str(self._config.get("shadow_mode") or "search").strip().lower()
            if mode not in ("search", "full"):
                raise ValueError(f"shadow_mode must be 'search' or 'full', got {mode!r}")
            if mode == "full" and self._llm is None:
                logger.error("prospecta shadow_mode=full needs an LLM; using 'search'")
                mode = "search"
            model = (str(self._config.get("shadow_embed_model") or "").strip()
                     or _SHADOW_DEFAULT_EMBED_MODEL)
            try:
                dim = int(self._config.get("shadow_embedding_dim") or 1536)
            except (TypeError, ValueError):
                logger.warning("invalid shadow_embedding_dim %r; using 1536",
                               self._config.get("shadow_embedding_dim"))
                dim = 1536
            from prospecta._tracer import NoOpTracer
            shadow_memory = _build_memory(
                self._resolve_max_connections(),
                self._resolve_pool_idle_seconds(),
                database_url=self._database_url,
                embed=self._build_shadow_embedder(model, dim),
                llm=self._llm if mode == "full" else None,
                bank_id=shadow_bank,
                tracer=NoOpTracer(),
            )
            self._shadow = _ShadowReader(
                self._memory, shadow_memory,
                bank_id=bank_id, shadow_bank_id=shadow_bank, mode=mode,
            )
            logger.info("Prospecta shadow reads ON: %s -> %s (%s, dim=%s, mode=%s)",
                        bank_id, shadow_bank, model, dim, mode)
        except Exception:
            self._shadow = None
            logger.exception("Prospecta shadow reads could not start; shadow is OFF")

    def _stage_kwargs(self) -> Dict[str, Any]:
        """The library CLI's linker/jev/rerank/synth stages (built when
        OPENROUTER_API_KEY is set). Any problem yields none, loudly."""
        try:
            from prospecta.cli._common import _stage_kwargs
            kw = _stage_kwargs()
        except Exception:
            logger.exception("prospecta could not build the linker stages; none will be used")
            return {}
        if "linker" not in kw:
            logger.warning("prospecta: no linker built (OPENROUTER_API_KEY unset or PROSPECTA_LINKER=off)")
        return kw

    def _recall_stage_kwargs(self) -> Dict[str, Any]:
        """Memory arguments for the library's Stage 2 (the cross-encoder, through OpenRouter's rerank
        endpoint with the profile's ``OPENROUTER_API_KEY``, read per call and never stored) and, only
        with ``recall.stage3`` on, the Stage 3 gate's Jev. They take effect where the bank's
        ``recall_config`` asks for them; a failure to build them leaves the library's own order."""
        # the profile's model ids only (None: the library resolves the bank's config, the environment, the default)
        kw: Dict[str, Any] = {"reranker_model": self._reranker_model(), "decision_model": self._decision_model()}
        try:
            from prospecta.stages import openrouter_rerank_transport
            kw["rerank_transport"] = openrouter_rerank_transport()
            if self._recall_stage3():
                from prospecta.stages import JevScore, openrouter_jev_transport
                model = self._decision_model()
                kw["jev"] = JevScore(openrouter_jev_transport(), **({"model": model} if model else {}))
        except Exception:
            logger.exception("prospecta could not build the Stage 2 / Stage 3 transports; the library's own order is used")
        return kw

    def _start_dual_write(self, bank_id: str) -> None:
        """Create the dual writer once; any problem leaves dual write OFF, loudly."""
        target = self._resolve_dual_write_bank_id()
        if not target:
            return
        try:
            if target == bank_id:
                raise ValueError(f"dual_write_bank_id {target!r} must differ from bank_id")
            model = (str(self._config.get("dual_write_embed_model") or "").strip()
                     or _SHADOW_DEFAULT_EMBED_MODEL)
            try:
                dim = int(self._config.get("dual_write_embedding_dim") or 1536)
            except (TypeError, ValueError):
                logger.warning("invalid dual_write_embedding_dim %r; using 1536",
                               self._config.get("dual_write_embedding_dim"))
                dim = 1536
            retry = (str(self._config.get("dual_write_retry_path") or "").strip()
                     or str(self._hermes_home / "prospecta-dual-write-retry.jsonl"))
            memory = _build_memory(
                self._resolve_max_connections(),
                self._resolve_pool_idle_seconds(),
                database_url=self._database_url,
                embed=self._build_shadow_embedder(model, dim),
                llm=self._llm,
                bank_id=target,
                **self._stage_kwargs(),
            )
            try:
                memory.bank_stats(target)  # the bank must already exist
            except Exception:
                memory.close()
                raise
            self._dual = _DualWriter(memory, bank_id=target, retry_path=Path(retry))
            logger.info("Prospecta dual write ON: %s -> %s (%s, dim=%s, retry file %s)",
                        bank_id, target, model, dim, retry)
        except Exception:
            self._dual = None
            logger.exception("Prospecta dual write could not start; dual write is OFF")

    def _timezone_name(self) -> str:
        """The IANA zone of the calendar day: the profile's ``timezone``, else PROSPECTA_TIMEZONE, else
        America/Los_Angeles; an unknown name falls back to the default (logged)."""
        name = (str(self._config.get("timezone") or "").strip()
                or os.environ.get("PROSPECTA_TIMEZONE", "").strip() or _DEFAULT_TIMEZONE)
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(name)
        except Exception:
            logger.warning("prospecta: unknown timezone %r for note dates; using %s", name, _DEFAULT_TIMEZONE)
            return _DEFAULT_TIMEZONE
        return name

    def _apply_timezone(self) -> None:
        """The library stamps each new note's day (``documents.created_on``) in PROSPECTA_TIMEZONE
        (UTC when unset); hand it this profile's zone. Never raises."""
        try:
            os.environ["PROSPECTA_TIMEZONE"] = self._timezone_name()
        except Exception:
            logger.exception("prospecta: could not set the timezone for note dates")

    def _today(self):
        """Today's calendar day in the profile's zone (what the contract's time scopes count back from)."""
        from datetime import date
        return date.fromisoformat(local_date_iso(self._timezone_name()))

    def _hash_precheck_enabled(self) -> bool:
        """Look for an existing note before paying for a save? PROSPECTA_HASH_PRECHECK
        env (even empty = unset) wins over config ``hash_precheck``. Default on."""
        raw = os.environ.get("PROSPECTA_HASH_PRECHECK")
        if raw is None or not raw.strip():
            raw = self._config.get("hash_precheck")
        return ("" if raw is None else str(raw).strip().lower()) not in ("0", "false", "no", "off")

    def _existing_note(self, content: str, kw: Dict[str, Any]) -> Optional[str]:
        """The id of the note already saved for this text and source, so the caller can
        skip the write. Returns None to write as usual: no such note, the check is off,
        or the lookup failed. A note with the same text under another source is
        logged and skipped too (the library would refuse it): returns "" then."""
        try:
            if not self._hash_precheck_enabled():
                return None
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            source = kw.get("source")
            if source is None and kw.get("path") is not None:
                source = str(Path(kw["path"]))
            if source is None:
                source = f"retain://{digest[:12]}"
            with self._memory._pool.connection() as conn:
                row = conn.execute(
                    "SELECT id, source FROM documents WHERE bank_id = %s AND content_hash = %s",
                    (self._memory._default_bank_id, digest)).fetchone()
            if not row:
                return None
            doc_id, existing_source = row
            if existing_source == source:
                logger.debug("prospecta: note %r already saved; skipping the write", source)
                return str(doc_id)
            logger.info("prospecta: same text already saved under source %r; not saving it again as %r",
                        existing_source, source)
            return ""
        except Exception:
            logger.debug("prospecta: hash pre-check failed; saving the note normally", exc_info=True)
            return None

    def _retain(self, content: str, **kw: Any) -> str:
        """Write to the main bank (linked when on), then queue the same note for
        the dual-write bank. The queueing never raises and never waits. The library
        indexes the note as chunks with no model call (``index_text`` only when the caller
        gives it) and stamps its day. A note the bank already holds is skipped before
        any embedding call."""
        existing = self._existing_note(content, kw)
        if existing is not None:
            return existing
        # the library's Stage 1 reads chunk items only: a note given question strings (index_text) is
        # also written as chunks, so it stays findable by it
        main_kw = dict(kw, child_chunks=True) if self._linked_writes or "index_text" in kw else kw
        document_id = self._memory.retain(content=content, **main_kw)
        dual = self._dual
        if dual is not None:
            try:
                dual.submit({"content": content, **kw})
            except Exception:
                logger.exception("prospecta dual-write submit failed for %r", kw.get("source"))
        return document_id

    def _shadow_submit(self, tool: str, message: str, *, old_ms: int, mode: str, limit: int,
                       depth: Optional[str], items: Any = None, result: Any = None) -> None:
        """Hand the finished OLD answer (search ``items`` or recall_synth ``result``)
        to the shadow worker. Never raises, never waits."""
        shadow = self._shadow
        if shadow is None:
            return
        try:
            synthesis = None
            queries = [message]
            if result is not None:
                items = result.sources
                synthesis = result.synthesis
                queries = [q.text for q in (getattr(result, "queries", None) or [])] or queries
            job = {
                "tool": tool, "message": message, "queries": queries,
                "results": list(items or []), "synthesis": synthesis,
                "old_ms": old_ms, "mode": mode, "limit": limit, "depth": depth,
            }
            shadow.submit(job)
        except Exception:
            logger.exception("prospecta shadow submit failed")

    def _resolve_llm_model(self) -> Optional[str]:
        """Resolve LLM model id from config or env.

        Priority: config llm_model > PROSPECTA_LLM_MODEL env > None (defaults module picks).
        """
        return (
            (self._config.get("llm_model") or "").strip()
            or os.environ.get("PROSPECTA_LLM_MODEL", "").strip()
            or None
        )

    def _build_llm(self):
        """Build LLMCallable via prospecta.defaults.make_default_llm.

        Mirrors Hindsight's pattern: reads model from config/env,
        delegates to the library's own LiteLLM-backed factory.
        Returns None if prospecta[defaults] is not installed (no LiteLLM).
        """
        # No hidden defaults: prospecta.defaults would silently fall back to
        # openai/gpt-4o-mini. An unset model is a loud initialize() failure (the
        # host logs it at ERROR and notifies the being).
        if not self._resolve_llm_model():
            raise RuntimeError(
                "Prospecta LLM model is not set: set PROSPECTA_LLM_MODEL in the "
                "profile .env (or llm_model in prospecta.json), e.g. "
                "'openrouter/anthropic/claude-sonnet-5.5'. Refusing the library's "
                "built-in default model."
            )
        try:
            import logging
            # Silence LiteLLM loggers by name BEFORE importing the module —
            # it emits DEBUG output during import-time initialization.
            for _name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router"):
                logging.getLogger(_name).setLevel(logging.WARNING)
            import litellm
            litellm.suppress_debug_info = True
            litellm.verbose_logger.setLevel(logging.WARNING)
            from prospecta.defaults import make_default_llm
        except ImportError:
            logger.warning(
                "prospecta.defaults not available (missing LiteLLM extra). "
                "recall_synth disabled; retain/search still work."
            )
            return None
        return make_default_llm(model=self._resolve_llm_model())

    # ---- lifecycle ----

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME")
        if not hermes_home:
            raise RuntimeError("HERMES_HOME not provided; cannot resolve plugin storage")
        self._hermes_home = Path(hermes_home)

        self._config = self._load_config()

        database_url = self._resolve_database_url()
        self._database_url = database_url

        from prospecta.db.migrate import run_migrations

        run_migrations(database_url=database_url)

        bank_id = self._resolve_bank_id()
        try:
            embedding_dim = int(self._config.get("embedding_dim", 1536))
        except (TypeError, ValueError):
            embedding_dim = 1536

        embed = self._build_embedder()
        llm = self._build_llm()
        self._llm = llm  # store on provider so callers use self._llm, not Memory internals

        self._linked_writes = self._resolve_linked_writes()
        self._apply_timezone()
        self._memory = _build_memory(
            self._resolve_max_connections(),
            self._resolve_pool_idle_seconds(),
            database_url=database_url,
            bank_id=bank_id,
            llm=llm,
            embed=embed,
            **{**self._recall_stage_kwargs(), **(self._stage_kwargs() if self._linked_writes else {})},
        )
        try:
            self._memory.create_bank(bank_id, embedding_dim=embedding_dim)
        except Exception as e:
            # idempotent — already exists with matching config is fine
            msg = str(e).lower()
            if "already exists" not in msg:
                # prospecta raises BankConfigConflict for dim mismatch; let that propagate
                raise

        if self._recall_date_cue_gate() and not self._date_gate_held:
            self._date_gate_held = _dcg.install()
            _dcg.set_timeout(self._recall_date_timeout())

        self._prefetch_cfg = self._resolve_prefetch_settings()
        self._warn_renamed_settings()
        self._check_recall_stages(bank_id)
        self._start_shadow(bank_id)
        self._start_dual_write(bank_id)
        self._start_connection_watch()

        logger.info(
            "Prospecta initialized: bank=%s dim=%s mode=%s",
            bank_id, embedding_dim,
            "embedded" if self._compose_project else "byo",
        )

    def _check_recall_stages(self, bank_id: str) -> None:
        """With the contract prefetch on, say once a session (read-only, never raises) what stops the
        library's query expansion from working: a library without ``Memory.expand_and_recall``, a bank
        with no ``hybrid_chunk`` channel (Stage 1: without it every turn goes without memory) or with no
        ``cross_encoder`` rerank (each search keeps Stage 1's own order)."""
        try:
            if not (self._prefetch_enabled() and self._prefetch_cfg is not None and self._prefetch_cfg.active):
                return
            if not hasattr(self._memory, "expand_and_recall"):
                logger.warning(
                    "prospecta: the installed library has no Memory.expand_and_recall: the contract prefetch "
                    "needs prospecta from main@736bd28 (PR 85) or later, so every turn goes without memory")
                return
            from prospecta.channels.recall import read_channel_config
            from prospecta.stages import read_recall_config
            with self._memory._pool.connection() as conn:
                channels = read_channel_config(conn, bank_id)
                rerank = read_recall_config(conn, bank_id).get("rerank") or {}
                conn.rollback()
            if not any(c.get("name") == "hybrid_chunk" and c.get("enabled", True) for c in channels):
                logger.warning(
                    "prospecta: bank %r has no hybrid_chunk channel in its channel_config: the contract "
                    "prefetch (query expansion) runs on Stage 1, so every turn goes without memory "
                    "(README, 'Recall: Stage 1 + Stage 2, cited')", bank_id)
            if not (rerank.get("enabled") and rerank.get("stage") == "cross_encoder"):
                logger.warning(
                    "prospecta: bank %r does not rerank with the cross_encoder stage (recall_config.rerank): "
                    "the contract prefetch keeps Stage 1's own order for each search, the candidates within "
                    "half of its best score (README, 'Recall: Stage 1 + Stage 2, cited')", bank_id)
        except Exception:
            logger.debug("prospecta: could not check the bank's recall stages", exc_info=True)

    def shutdown(self) -> None:
        if self._date_gate_held:
            self._date_gate_held = False
            _dcg.uninstall()
        # Wait briefly for in-flight sync_turn
        t = self._sync_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        watch, self._conn_watch = self._conn_watch, None
        if watch is not None:
            watch.stop()
        dual, self._dual = self._dual, None
        if dual is not None:  # drains the queued second writes
            dual.stop()
        shadow, self._shadow = self._shadow, None
        if shadow is not None:  # before the old Memory: it writes the pair events
            shadow.stop()
        if self._memory is not None:
            try:
                self._memory.shutdown()
            except Exception as e:
                logger.warning("Prospecta memory.shutdown raised: %s", e)
            self._memory = None

    # ---- tools ----

    def _tool_guide_enabled(self) -> bool:
        """Add the tool guide to the prompt block? PROSPECTA_TOOL_GUIDE env (even
        empty = unset) wins over config ``tool_guide``. Unset means on."""
        raw = os.environ.get("PROSPECTA_TOOL_GUIDE")
        if raw is None or not raw.strip():
            raw = self._config.get("tool_guide")
        text = "" if raw is None else str(raw).strip().lower()
        return text not in ("0", "false", "no", "off")

    def system_prompt_block(self) -> str:
        block = self._base_prompt_block()
        if self._tool_guide_enabled():
            block += "\n\n" + _TOOL_GUIDE
        return block

    def _base_prompt_block(self) -> str:
        return (
            "You have access to prospecta memory — bilateral question-space retrieval. "
            "Use `prospecta_retain` to store memorable facts or decisions; supply a clear `source` "
            "for replace-on-source-match. Use `prospecta_recall` to retrieve a synthesized answer "
            "about a topic. Use `prospecta_search` for raw chunks when you need to see matches. "
            "Memory persists across sessions."
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "prospecta_retain",
                "description": (
                    "Store a document in prospecta memory. It is indexed as chunks, with no model "
                    "call, and is searchable at once. Use to record facts, decisions, "
                    "conversation summaries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content to remember (the body).",
                        },
                        "source": {
                            "type": "string",
                            "description": (
                                "An identifier for this memory (e.g. 'meeting-2026-05-19', "
                                "'fact:kelly-birthday'). Used for replace-on-source-match."
                            ),
                        },
                        "index_text": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": (
                                "Optional anticipated questions, kept as extra search keys next "
                                "to the chunks. Pass a single string or list of strings."
                            ),
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for metadata filtering.",
                        },
                    },
                    "required": ["content", "source"],
                },
            },
            {
                "name": "prospecta_recall",
                "description": (
                    "Retrieve and synthesize from prospecta memory: hybrid retrieval and "
                    "cross-encoder rerank, then an LLM answer. Returns the answer and the cited "
                    "sources (source, date, kind, match, relevance, exact_snippet)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The question or topic to recall about.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results per formulated query.",
                            "default": 10,
                        },
                        "depth": {
                            "type": "string",
                            "enum": ["standard", "deep"],
                            "description": "Retrieval depth. 'standard' (default) is right for ordinary questions. Use 'deep' for discovery and completeness questions ('everything about X', 'all the ...', 'every time ...'): it reads a wider pool for about 3x the cost.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "prospecta_recall_set",
                "description": (
                    "Complete answer to a SET or discovery question about one person or "
                    "entity ('what are X's nicknames', 'every project Y mentioned'): reads "
                    "every note tied to the entity and returns a deduplicated cited list. "
                    "Slower and costlier than prospecta_recall; use it when completeness "
                    "matters more than speed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The set question."},
                        "entity": {"type": "string", "description": "The person or entity the set is about."},
                        "aliases": {"type": "array", "items": {"type": "string"}, "description": "Other names for the entity."},
                    },
                    "required": ["query", "entity"],
                },
            },
            {
                "name": "prospecta_search",
                "description": (
                    "Hybrid retrieval from prospecta memory without LLM synthesis: semantic, "
                    "stemmed and exact-phrase matching, then a cross-encoder rerank. Returns cited "
                    "hits: source, date, kind, how it matched, relevance, the matched words in "
                    "context (exact_snippet) and the whole chunk. Put an \"exact phrase\" in quotes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["hybrid", "semantic", "lexical"],
                            "default": "hybrid",
                        },
                        "limit": {"type": "integer", "default": 10},
                        "depth": {
                            "type": "string",
                            "enum": ["standard", "deep"],
                            "description": "Retrieval depth. 'standard' (default) is right for ordinary questions. Use 'deep' for discovery and completeness questions ('everything about X', 'all the ...', 'every time ...'): it reads a wider pool for about 3x the cost.",
                        },
                        "date_from": {
                            "type": "string",
                            "description": "Only notes dated on or after this day (YYYY-MM-DD).",
                        },
                        "date_to": {
                            "type": "string",
                            "description": "Only notes dated on or before this day (YYYY-MM-DD).",
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._memory is None:
            return json.dumps({"error": "prospecta not initialized"})

        try:
            if tool_name == "prospecta_retain":
                content = args.get("content", "")
                source = args.get("source", "")
                if not content or not source:
                    return json.dumps({"error": "content and source are required"})
                kw: Dict[str, Any] = {"source": source}
                if "index_text" in args and args["index_text"] is not None:
                    kw["index_text"] = args["index_text"]
                if "tags" in args and args["tags"]:
                    kw["tags"] = list(args["tags"])
                document_id = self._retain(content, **kw)
                return json.dumps({"document_id": str(document_id), "source": source})

            if tool_name == "prospecta_recall":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "query is required"})
                if self._llm is None:
                    return json.dumps({"error": "prospecta_recall requires an LLM. Set PROSPECTA_LLM_MODEL env or llm_model in prospecta.json and ensure 'prospecta[defaults]' is installed."})
                limit = int(args.get("limit", 10))
                t0 = time.monotonic()
                if self._recall_stage3():
                    # the library's ask(): Stage 1 + Stage 2, a grounded answer with [n] citations, and the
                    # Stage 3 gate (one broadened search when the evidence falls short)
                    asked = self._memory.ask(query, limit=limit, sufficiency=True, **_depth_kw(args))
                    self._shadow_submit(
                        "prospecta_recall", query, items=asked.items, mode="hybrid", limit=limit,
                        depth=args.get("depth"), old_ms=int((time.monotonic() - t0) * 1000))
                    return json.dumps({
                        "synthesis": asked.rendered or "",
                        "answer": asked.answer or "",
                        "citations": asked.citations,
                        "sufficient": asked.sufficient,
                        "retried": asked.retried,
                        "gate": asked.gate,
                        "sources": [_cited_hit(i + 1, it, text=False) for i, it in enumerate(asked.items or [])],
                    })
                result = self._memory.recall_synth(query, limit=limit, **_depth_kw(args))
                self._shadow_submit(
                    "prospecta_recall", query, result=result, mode="hybrid", limit=limit,
                    depth=args.get("depth"), old_ms=int((time.monotonic() - t0) * 1000))
                sources = []
                for i, hit in enumerate(result.sources or []):
                    sources.append({
                        **_cited_hit(i + 1, hit, text=False),
                        "rank": i + 1,
                        "rrf_score": (hit.scores or {}).get("rrf") if hasattr(hit, "scores") else None,
                    })
                return json.dumps({
                    "synthesis": result.synthesis or "",
                    "sources": sources,
                })

            if tool_name == "prospecta_recall_set":
                query = args.get("query", "")
                entity = args.get("entity", "")
                if not query or not entity:
                    return json.dumps({"error": "query and entity are required"})
                if self._llm is None:
                    return json.dumps({"error": "prospecta_recall_set requires an LLM. Set PROSPECTA_LLM_MODEL env or llm_model in prospecta.json and ensure 'prospecta[defaults]' is installed."})
                kw = {"entity": entity}
                if args.get("aliases"):
                    kw["aliases"] = list(args["aliases"])
                result = self._memory.recall_mapreduce(query, **kw)
                return json.dumps({
                    "synthesis": result.synthesis or "",
                    "citations": [c if isinstance(c, (str, int, float, dict, list)) else str(c)
                                  for c in (getattr(result, "citations", None) or [])],
                })

            if tool_name == "prospecta_search":
                query = args.get("query", "")
                if not query:
                    return json.dumps({"error": "query is required"})
                return self._search_tool(query, args)

            return json.dumps({"error": f"unknown tool: {tool_name}"})
        except Exception as e:
            logger.warning("prospecta tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ---- prospecta_search: the library's Stage 1 + Stage 2, every hit cited ----

    def _search_tool(self, query: str, args: Dict[str, Any]) -> str:
        """The library's search (Stage 1 hybrid candidates, Stage 2 cross-encoder rerank where the
        bank has them), as cited hits: source, date, kind, how it matched, relevance, the matched words
        in context (``exact_snippet``) and the whole chunk. The shadow compares the same list."""
        mode = args.get("mode", "hybrid")
        limit = int(args.get("limit", 10))
        window = {k: _iso_date(args.get(k), k) for k in ("date_from", "date_to") if args.get(k) not in (None, "")}
        trace: List[Dict[str, Any]] = []
        t0 = time.monotonic()
        items = list(self._memory.search(query, mode=mode, limit=limit, _trace=trace,
                                         **_depth_kw(args), **window))
        old_ms = int((time.monotonic() - t0) * 1000)
        self._shadow_submit("prospecta_search", query, items=items, mode=mode, limit=limit,
                            depth=args.get("depth"), old_ms=old_ms)
        return json.dumps({
            "results": [_cited_hit(i + 1, it) for i, it in enumerate(items)],
            "count": len(items),
            "reranked": _stage2_graded(trace),
        })

    # ---- hooks ----

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._memory is None:
            return

        sid = session_id or self._session_id
        # the turn's record is named now, so the next recall can leave it out even while it is being written
        source = f"{_pc.TURN_SOURCE_PREFIX}{sid}:{int(time.time() * 1000)}"
        self._keep_exchange(sid, assistant_content, source)

        def _sync() -> None:
            try:
                # chunk-only: the library indexes the turn as chunks with no model call and stamps its day
                content = f"<user>: {user_content}\n\n<assistant>: {assistant_content}"
                self._retain(content, source=source, tags=["conversation"])
            except Exception:
                logger.exception("prospecta sync_turn failed")

        # Bound any pre-existing in-flight sync
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="prospecta-sync")
        self._sync_thread.start()

    def _keep_exchange(self, sid: str, assistant_content: Any, turn_source: str = "") -> None:
        """The agent's last answer of each session (whole) and how many turns it has had: the
        contract's ``agent_last``; and the sources of its last two retained turns, which the next
        recall leaves out. Bounded to the 64 most recent sessions."""
        try:
            text = assistant_content if isinstance(assistant_content, str) else str(assistant_content or "")
            with self._state_lock:
                turns = self._exchanges.pop(sid, ("", 0))[1] + 1
                self._exchanges[sid] = (text, turns)
                while len(self._exchanges) > 64:
                    self._exchanges.popitem(last=False)
                if turn_source:
                    self._turn_sources[sid] = (self._turn_sources.pop(sid, []) + [turn_source])[-2:]
                    while len(self._turn_sources) > 64:
                        self._turn_sources.popitem(last=False)
        except Exception:
            logger.debug("prospecta: could not keep the last answer", exc_info=True)

    def _recover_last_answer(self, sid: str) -> None:
        """When this session has no kept answer (after a restart or with a new session id), read
        the session's last assistant message once from the profile's session store and keep it.
        Never raises; on any trouble the session just has no last answer."""
        try:
            with self._state_lock:
                if sid in self._exchanges:
                    return
            text = self._read_last_assistant(sid)
            try:   # the bank, not the session store: bounded, so a busy pool never holds the turn
                sources = _pc.run_bounded(lambda: self._read_turn_sources(sid), _RECOVER_SOURCES_TIMEOUT_S)
            except Exception:
                sources = []
            with self._state_lock:
                if sid not in self._exchanges:
                    self._exchanges[sid] = (text, 0)
                    while len(self._exchanges) > 64:
                        self._exchanges.popitem(last=False)
                    if sources and sid not in self._turn_sources:
                        self._turn_sources[sid] = sources
        except Exception:
            logger.debug("prospecta: could not recover the last answer", exc_info=True)

    def _read_last_assistant(self, sid: str) -> str:
        """The whole latest non-empty assistant message of ``sid`` from ``state.db`` in the
        profile home, through a separate read-only connection. '' on any error or no row."""
        if self._hermes_home is None or not sid:
            return ""
        db = Path(self._hermes_home) / "state.db"
        if not db.is_file():
            return ""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True, timeout=0.2)
            conn.execute("PRAGMA busy_timeout = 200")
            row = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND role = 'assistant' "
                "AND content IS NOT NULL AND content != '' ORDER BY timestamp DESC LIMIT 1", (sid,)).fetchone()
            text = row[0] if row else ""
            return text if isinstance(text, str) else ""
        except Exception:
            return ""
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _read_turn_sources(self, sid: str) -> List[str]:
        """The sources of the last two turns the plugin retained for ``sid`` (oldest first), read from
        the bank (a read-only query). [] on any error. A restart or a new gateway process has no
        memory of them; the bank has."""
        if not sid or self._memory is None:
            return []
        try:
            with self._memory._pool.connection() as conn:
                rows = conn.execute(
                    "SELECT source FROM documents WHERE bank_id = %s AND starts_with(source, %s) "
                    "ORDER BY created_at DESC, source DESC LIMIT 2",
                    (self._memory._default_bank_id, f"{_pc.TURN_SOURCE_PREFIX}{sid}:")).fetchall()
                conn.rollback()
            return [r[0] for r in reversed(rows)]
        except Exception:
            logger.debug("prospecta: could not read the session's last turns", exc_info=True)
            return []

    def _echo_sources(self, sid: str) -> List[str]:
        with self._state_lock:
            return list(self._turn_sources.get(sid, []))

    def _last_exchange(self, sid: str) -> "tuple":
        with self._state_lock:
            return self._exchanges.get(sid, ("", 0))

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Core calls this with the WHOLE user message (skill scaffolding included) just before ``prefetch_all``,
        which hands ``prefetch`` only the user's instruction of a skill turn. Remember whether this turn is a
        skill/cron invocation; ``prefetch`` and ``queue_prefetch`` read it. Never raises."""
        try:
            self._skill_turn = _pc.skill_turn_shape(message)
        except Exception:
            self._skill_turn = None

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # OFF by default: ``prefetch_enabled`` is the master switch. With it on,
        # ``recall.prefetch.mode`` picks the path: 'off' (default) keeps the old synthesised
        # answer, 'contract' runs the written recall contract. Neither ever fails a turn.
        if not self._prefetch_enabled() or self._memory is None:
            return ""
        try:
            # A skill or cron turn (the on_turn_start flag, or the text's own shape) gets no recall at all: no
            # decision, no search, no writer call, nothing injected. Only the shape's name is ever logged.
            shape = self._skill_turn or _pc.skill_turn_shape(query)
            if shape:
                self._last_brief = _pc.Brief(reason="skill turn")
                logger.debug("prospecta prefetch: skill turn (%s); no recall", shape)
                return ""
            cfg = self._prefetch_settings()
            if not cfg.active:
                return self._prefetch_synth(query)
            return self._prefetch_contract(query, session_id or self._session_id, cfg)
        except Exception as e:
            # one line, no text: the type, and the contract's own message when the contract is the problem
            logger.warning("prospecta prefetch failed (this turn goes without memory): %s%s",
                           type(e).__name__, f": {e}" if isinstance(e, _pc.ContractError) else "")
            return ""

    def _prefetch_synth(self, query: str) -> str:
        """The old prefetch (``recall.prefetch.mode: off``): the model-written answer."""
        try:
            t0 = time.monotonic()
            result = self._memory.recall_synth(query)
            self._shadow_submit(
                "prefetch", query, result=result, mode="hybrid", limit=10, depth=None,
                old_ms=int((time.monotonic() - t0) * 1000))
            if not result.synthesis:
                return ""
            srcs = [getattr(s, "source", "") for s in (result.sources or [])[:3]]
            srcs_str = ", ".join(s for s in srcs if s) or "no sources"
            return (
                "<prospecta-recall>\n"
                f"{result.synthesis}\n\n"
                f"(sources: {srcs_str})\n"
                "</prospecta-recall>"
            )
        except Exception as e:
            logger.warning("prospecta prefetch failed: %s", e)
            return ""

    # ---- recall.prefetch.mode: contract ----

    def _contract_deps(self) -> Any:
        """The run's outside world: the library's one entry point for automatic recall (its query-writer
        model is the Memory's ``llm``, called with the writer model per call; its searches, reranks and merge
        run inside it) and the Jev transport of the gate. A library without the entry point raises here, which
        ``prefetch`` turns into its one warning line."""
        transport = self._jev_transport
        if transport is None:
            transport = _pc.default_transport()
        return _pc.Deps(expand=self._memory.expand_and_recall, transport=transport, today=self._today)

    def _prefetch_contract(self, query: str, sid: str, cfg: Any) -> str:
        contract = _pc.load_contract(cfg.contract_path or None)
        started = time.monotonic()
        self._recover_last_answer(sid)
        agent_last, turns = self._last_exchange(sid)
        with self._state_lock:
            ahead = self._ahead.get(sid)
        if ahead is not None and started - ahead.at > _pc.AHEAD_TTL_S:
            ahead = None
        deps = self._contract_deps()
        try:
            brief = _pc.run_bounded(lambda: _pc.run_contract(
                query, agent_last=agent_last, turns=turns, deps=deps, settings=cfg, contract=contract,
                ahead=ahead, started=started, excluded_sources=self._echo_sources(sid)),
                cfg.deadline_s + _DEADLINE_GRACE_S)
        except TimeoutError:
            logger.warning("prospecta prefetch: no answer within %.1f s; this turn goes without memory",
                           cfg.deadline_s)
            return ""
        self._last_brief = brief
        if brief.ahead_used:
            with self._state_lock:
                self._ahead.pop(sid, None)
        return brief.text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Ahead of the next turn (contract mode only): when the agent's own last answer ended on a
        proposal, prepare the block its go-ahead will need while the person answers. Keyed on that
        answer (kept from sync_turn), not on ``query`` (the person's text). Runs on its own thread
        and never raises."""
        try:
            if not self._prefetch_enabled() or self._memory is None:
                return
            if self._skill_turn or _pc.skill_turn_shape(query):      # a skill/cron turn prepares nothing ahead
                return
            cfg = self._prefetch_settings()
            if not cfg.active:
                return
            sid = session_id or self._session_id
            agent_last, _ = self._last_exchange(sid)
            if not agent_last or not _pc.looks_like_proposal(agent_last):
                return
            key = _pc.ahead_key(agent_last)
            with self._state_lock:
                have = self._ahead.get(sid)
                if (have is not None and have.key == key) or (sid, key) in self._ahead_running:
                    return
                self._ahead_running.add((sid, key))
            threading.Thread(target=self._prepare_ahead, args=(sid, key, agent_last, cfg),
                             daemon=True, name="prospecta-ahead").start()
        except Exception:
            logger.debug("prospecta queue_prefetch failed", exc_info=True)

    def _prepare_ahead(self, sid: str, key: str, agent_last: str, cfg: Any) -> None:
        try:
            contract = _pc.load_contract(cfg.contract_path or None)
            deps = self._contract_deps()
            started = time.monotonic()
            entry = _pc.run_bounded(lambda: _pc.run_ahead(
                agent_last, deps=deps, settings=cfg, contract=contract, started=started,
                excluded_sources=self._echo_sources(sid)), cfg.deadline_s)
            if entry is not None:
                with self._state_lock:
                    self._ahead[sid] = entry
                    while len(self._ahead) > 64:
                        self._ahead.pop(next(iter(self._ahead)))
        except Exception:
            logger.debug("prospecta ahead failed", exc_info=True)
        finally:
            with self._state_lock:
                self._ahead_running.discard((sid, key))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        t = self._sync_thread
        if t is not None and t.is_alive():
            try:
                t.join(timeout=5.0)
            except Exception:
                pass


def register(ctx) -> None:
    """Called by Hermes memory plugin discovery."""
    provider = ProspectaProvider()
    ctx.register_memory_provider(provider)
