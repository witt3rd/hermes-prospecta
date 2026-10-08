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

Lifecycle hooks: sync_turn (non-blocking daemon thread), on_session_end,
                  shutdown. Prefetch OPT-IN via config (default OFF —
                  spine cost should not be a surprise tax).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
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


class _ShadowReader:
    """Replays the old bank's recalls against a second (new) bank, off-path.

    The library's ``Memory(shadow_bank_id=...)`` only shadows ``Memory.recall``
    (this provider calls ``search`` / ``recall_synth``) and opens a fresh
    connection pool per call, so this reuses its pair format (two
    ``recall_events`` rows sharing ``trace.shadow.id``, with ``jaccard`` over
    original document ids) but keeps ONE shadow ``Memory`` for the provider's
    life and runs it on ONE daemon worker thread behind a bounded queue.
    Nothing here raises into, or waits on, the caller.
    """

    def __init__(self, memory, shadow_memory, *, bank_id: str, shadow_bank_id: str,
                 mode: str = "search") -> None:
        self._memory = memory            # old-bank Memory: writes both events, resolves ids
        self._shadow = shadow_memory     # new-bank Memory, created once, reused
        self._bank_id = bank_id
        self._shadow_bank_id = shadow_bank_id
        self._mode = mode                # "search" | "full"
        self._queue: "queue.Queue" = queue.Queue(maxsize=_SHADOW_QUEUE_MAX)
        self._thread = threading.Thread(target=self._run, daemon=True, name="prospecta-shadow")
        self._thread.start()

    def submit(self, job: Dict[str, Any]) -> None:
        """Queue a job; never blocks and never raises."""
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            logger.error(
                "prospecta shadow queue full (%d pending); dropped shadow pair for %s %r",
                _SHADOW_QUEUE_MAX, job.get("tool"), job.get("message"),
            )
        except Exception:
            logger.exception("prospecta shadow submit failed")

    def stop(self) -> None:
        try:
            self._queue.put(_SHADOW_STOP, timeout=_SHADOW_DRAIN_TIMEOUT_S)
        except queue.Full:
            logger.error("prospecta shadow queue stayed full for %ss at shutdown",
                         _SHADOW_DRAIN_TIMEOUT_S)
        self._thread.join(timeout=_SHADOW_DRAIN_TIMEOUT_S)
        if self._thread.is_alive():
            logger.error("prospecta shadow worker still busy after %ss; leaving its "
                         "connection open (daemon thread dies with the process)",
                         _SHADOW_DRAIN_TIMEOUT_S)
            return
        try:
            self._shadow.close()
        except Exception:
            logger.exception("prospecta shadow memory close failed")

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is _SHADOW_STOP:
                return
            try:
                self._record(job)
            except Exception:
                logger.exception("prospecta shadow job failed: %s %r",
                                 job.get("tool"), job.get("message"))

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
        env = os.environ.get("PROSPECTA_SHADOW_BANK")
        if env is not None:
            return env.strip()
        return str(self._config.get("shadow_bank_id") or "").strip()

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
            from prospecta.memory import Memory
            shadow_memory = Memory(
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
        from prospecta.memory import Memory

        run_migrations(database_url=database_url)

        bank_id = self._resolve_bank_id()
        try:
            embedding_dim = int(self._config.get("embedding_dim", 1536))
        except (TypeError, ValueError):
            embedding_dim = 1536

        embed = self._build_embedder()
        llm = self._build_llm()
        self._llm = llm  # store on provider so callers use self._llm, not Memory internals

        self._memory = Memory(
            database_url=database_url,
            bank_id=bank_id,
            llm=llm,
            embed=embed,
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

    def system_prompt_block(self) -> str:
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
                document_id = self._memory.retain(content=content, **kw)
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

        def _sync() -> None:
            try:
                content = f"<user>: {user_content}\n\n<assistant>: {assistant_content}"
                source = f"turn:{sid}:{int(time.time() * 1000)}"
                kwargs = {"source": source, "tags": ["conversation"]}
                if self._llm is None:
                    kwargs["index_text"] = [user_content, assistant_content]
                self._memory.retain(content=content, **kwargs)
            except Exception as e:
                logger.warning("prospecta sync_turn failed: %s", e)

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
