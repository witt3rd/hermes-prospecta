"""Prospecta memory provider for Hermes Agent.

Bilateral prospective synthesis: both write-side (LLM-anticipated index_text)
and read-side (LLM-generated multi-query expansion) are LLM-mediated.
Three-channel hybrid index (semantic + content-stem + body-stem) fused via RRF.

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
  - prospecta_recall(query, limit=10) -> synthesis + sources
  - prospecta_search(query, mode="hybrid", limit=10) -> raw chunks
  - prospecta_recall_set(query, entity, aliases=None) -> complete cited list (map-reduce)
  (recall/search take an optional depth: standard | deep)

Shadow reads (OFF by default; see README): with ``shadow_bank_id`` set, every
search / recall / prefetch is also run against that bank on a background
thread and the old/new pair is stored in recall_events. The agent only ever
sees the old bank's answer.

Linked / dual writes (OFF by default; see README): ``linked_writes`` builds the
library's linker for the main Memory; ``dual_write_bank_id`` also retains every
note into a second bank on a background thread, with a JSONL retry file.

Note dates (see README): every new note is saved with its local date
(``metadata={"created_on": ...}``, ``timezone`` config, default
America/Los_Angeles) so the library's date filter can find it.

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
from contextlib import contextmanager
from datetime import datetime, timezone as _utc_tz
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

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
    "- prospecta_search: fast, no LLM calls. Use it for exact names, for lists of notes, and "
    "whenever you want to see the matching notes yourself.\n"
    "- prospecta_recall: reads the matches and writes one synthesized answer. It costs LLM "
    "calls, so use it when you need an answer, not a list.\n"
    "- depth (search and recall): 'standard' suits ordinary questions. 'deep' reads a wider pool "
    "for about 3x the cost; use it for 'everything about X' or 'every time ...' questions.\n"
    "- A date in a question (\"in July\", a year) narrows results only for notes that carry a "
    "date; notes without one are not filtered by it.\n"
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


# ---- shadow reads (embedding migration) ----

_SHADOW_QUEUE_MAX = 16          # pending shadow jobs; beyond this a job is dropped, loudly
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
_POOL_WAIT_S = 30.0


class _BoundedPool:
    """Stands in for a Memory's ``_pool``: at most ``max_connections`` shared
    connections, however many threads use it.

    The library's own pool keeps one connection per thread and never closes it
    when the thread ends, which piles up on a gateway that runs each turn on a
    new thread. Here a thread borrows one connection for the length of a
    ``connection()``/``cursor()`` block (nested blocks on the same thread reuse
    it) and gives it back. A connection that has been lost is dropped and the
    next borrower gets a fresh one. Same methods as the library pool:
    connection(), cursor(), dedicated(), close(), ``_conn``, ``database_url``.
    """

    def __init__(self, database_url: str, max_connections: int = _DEFAULT_MAX_CONNECTIONS) -> None:
        self._database_url = database_url
        self._max = max(1, int(max_connections))
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

    def _acquire(self):
        import psycopg
        deadline = time.monotonic() + _POOL_WAIT_S
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("prospecta connection pool is closed")
                while self._idle:
                    conn = self._idle.pop()
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
            self._cond.notify()

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
            self._cond.notify_all()
        for c in conns:
            try:
                if not c.closed:
                    c.close()
            except Exception:
                logger.debug("prospecta: closing a connection failed", exc_info=True)


def _build_memory(max_connections: int = _DEFAULT_MAX_CONNECTIONS, **kw: Any):
    """``prospecta.Memory(**kw)`` whose connections are a ``_BoundedPool``
    (also used by its default tracer sink), so ``close()`` ends them all."""
    from prospecta.memory import Memory
    memory = Memory(**kw)
    old = getattr(memory, "_pool", None)
    if old is not None:
        pool = _BoundedPool(old.database_url, max_connections)
        memory._pool = pool
        tracer = getattr(memory, "_tracer", None)
        if getattr(tracer, "_pool", None) is old:
            tracer._pool = pool
    return memory


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
        overlap = {
            "old_origin_document_ids": old_origin,
            "new_origin_document_ids": new_origin,
            "jaccard": (len(old_set & new_set) / len(union)) if union else 1.0,
        }
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
                "description": "Run recall_synth before each turn (costs 2 LLM calls). Default off.",
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
                "description": "IANA timezone (e.g. America/Los_Angeles) whose calendar day is saved as each new note's date (documents.created_on). Default America/Los_Angeles.",
                "default": _DEFAULT_TIMEZONE,
            },
            {
                "key": "note_date",
                "description": "Save each new note's date so date filters work: 'true' / 'false'. Empty (default) = on, except off while linked_writes is on. PROSPECTA_NOTE_DATE env overrides.",
                "default": "",
            },
            {
                "key": "hash_precheck",
                "description": "Before saving a note, look up whether the bank already holds the same text, and skip the write (and its LLM call) if so: 'true' / 'false'. Empty (default) = on. PROSPECTA_HASH_PRECHECK env overrides.",
                "default": "",
            },
            {
                "key": "tool_guide",
                "description": "Add a short guide to the memory tools to the system prompt: 'true' / 'false'. Empty (default) = on. PROSPECTA_TOOL_GUIDE env overrides.",
                "default": "",
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

    def _note_dates_enabled(self) -> bool:
        """Save the note's date with new notes? PROSPECTA_NOTE_DATE env (even
        empty = unset) wins over config ``note_date``. Unset means on, except
        while linked writes are on (the linker orders notes by this date, so
        turning it on there is an explicit choice)."""
        raw = os.environ.get("PROSPECTA_NOTE_DATE")
        if raw is None or not raw.strip():
            raw = self._config.get("note_date")
        text = "" if raw is None else str(raw).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return not self._linked_writes

    def _note_date_metadata(self) -> Dict[str, Any]:
        """Document metadata that makes the library store documents.created_on
        (key ``created_on``): the date now, in the configured ``timezone``
        (default America/Los_Angeles). Empty when off; never raises."""
        try:
            if not self._note_dates_enabled():
                return {}
            return {"created_on": local_date_iso(str(self._config.get("timezone") or ""))}
        except Exception:
            logger.exception("prospecta: could not work out the note date; saving the note without it")
            return {}

    def _with_note_date(self, kw: Dict[str, Any]) -> Dict[str, Any]:
        """kw plus the note date in ``metadata``; other metadata is never
        touched, and a date already there (created_on / created / date) wins."""
        existing = kw.get("metadata")
        if existing is not None and not isinstance(existing, dict):
            return kw
        if existing and any(k in existing for k in ("created_on", "created", "date")):
            return kw
        stamp = self._note_date_metadata()
        if not stamp:
            return kw
        return {**kw, "metadata": {**(existing or {}), **stamp}}

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
        the dual-write bank. The queueing never raises and never waits.
        A note the bank already holds is skipped before the LLM call."""
        existing = self._existing_note(content, kw)
        if existing is not None:
            return existing
        kw = self._with_note_date(kw)
        main_kw = dict(kw, child_chunks=True) if self._linked_writes else kw
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
            shadow.submit({
                "tool": tool, "message": message, "queries": queries,
                "results": list(items or []), "synthesis": synthesis,
                "old_ms": old_ms, "mode": mode, "limit": limit, "depth": depth,
            })
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
        self._memory = _build_memory(
            self._resolve_max_connections(),
            database_url=database_url,
            bank_id=bank_id,
            llm=llm,
            embed=embed,
            **(self._stage_kwargs() if self._linked_writes else {}),
        )
        try:
            self._memory.create_bank(bank_id, embedding_dim=embedding_dim)
        except Exception as e:
            # idempotent — already exists with matching config is fine
            msg = str(e).lower()
            if "already exists" not in msg:
                # prospecta raises BankConfigConflict for dim mismatch; let that propagate
                raise

        self._start_shadow(bank_id)
        self._start_dual_write(bank_id)

        logger.info(
            "Prospecta initialized: bank=%s dim=%s mode=%s",
            bank_id, embedding_dim,
            "embedded" if self._compose_project else "byo",
        )

    def shutdown(self) -> None:
        # Wait briefly for in-flight sync_turn
        t = self._sync_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
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
                    "Store a document in prospecta memory. The LLM will generate anticipated "
                    "questions as index_text unless you supply them. Use to record facts, "
                    "decisions, conversation summaries."
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
                                "Optional caller-supplied anticipated questions. If provided, "
                                "skips LLM generation. Pass a single string or list of strings."
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
                    "Retrieve and synthesize from prospecta memory. Multi-query formulation + "
                    "hybrid retrieval + LLM synthesis. Returns a synthesized answer with "
                    "source attribution."
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
                    "Raw hybrid retrieval from prospecta memory without LLM synthesis. "
                    "Returns chunks with scores. Use when you want to see raw matches."
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
                result = self._memory.recall_synth(query, limit=limit, **_depth_kw(args))
                self._shadow_submit(
                    "prospecta_recall", query, result=result, mode="hybrid", limit=limit,
                    depth=args.get("depth"), old_ms=int((time.monotonic() - t0) * 1000))
                sources = []
                for i, s in enumerate(result.sources or []):
                    sources.append({
                        "source": getattr(s, "source", None),
                        "rank": i + 1,
                        "rrf_score": (s.scores or {}).get("rrf") if hasattr(s, "scores") else None,
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
                mode = args.get("mode", "hybrid")
                limit = int(args.get("limit", 10))
                t0 = time.monotonic()
                items = self._memory.search(query, mode=mode, limit=limit, **_depth_kw(args))
                self._shadow_submit(
                    "prospecta_search", query, items=items, mode=mode, limit=limit,
                    depth=args.get("depth"), old_ms=int((time.monotonic() - t0) * 1000))
                results = []
                for it in items:
                    content = it.original_chunk or it.content
                    if content and len(content) > 2000:
                        content = content[:2000] + "...[truncated]"
                    results.append({
                        "content": content,
                        "source": it.source,
                        "scores": dict(it.scores) if it.scores else {},
                    })
                return json.dumps({"results": results})

            return json.dumps({"error": f"unknown tool: {tool_name}"})
        except Exception as e:
            logger.warning("prospecta tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # ---- hooks ----

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._memory is None:
            return

        sid = session_id or self._session_id
        turn_date = self._note_date_metadata()  # the turn's own day, fixed when the turn ends

        def _sync() -> None:
            try:
                content = f"<user>: {user_content}\n\n<assistant>: {assistant_content}"
                source = f"turn:{sid}:{int(time.time() * 1000)}"
                kwargs = {"source": source, "tags": ["conversation"]}
                if self._llm is None:
                    kwargs["index_text"] = [user_content, assistant_content]
                if turn_date:
                    kwargs["metadata"] = dict(turn_date)
                self._retain(content, **kwargs)
            except Exception:
                logger.exception("prospecta sync_turn failed")

        # Bound any pre-existing in-flight sync
        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="prospecta-sync")
        self._sync_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # OFF by default — opt-in via config or env.
        enabled = (
            bool(self._config.get("prefetch_enabled"))
            and str(self._config.get("prefetch_enabled")).lower() not in ("false", "0", "")
        ) or bool(os.environ.get("PROSPECTA_PREFETCH"))
        if not enabled or self._memory is None:
            return ""
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
