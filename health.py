"""hermes-prospecta-health: console health check for the memory stack.

Exits 0 only when BOTH the Postgres database and the configured embedder
are usable. Otherwise exits 1 and prints a LOUD ``PROSPECTA DOWN`` line on
stderr naming each failed component, so a silent memory outage (dropped
venv, dead database) cannot go unnoticed.

Self-contained on purpose: installed as a top-level module with the
``hermes-prospecta-health`` console script; does not import Hermes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    return Path(home) if home else Path.home() / ".hermes"


def _load_config() -> dict:
    path = _hermes_home() / "prospecta.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_database_url(cfg: dict) -> str:
    return (
        os.environ.get("PROSPECTA_DATABASE_URL")
        or cfg.get("database_url")
        or os.environ.get("DATABASE_URL")
        or ""
    )


def build_embedder(cfg: dict) -> Callable:
    """Mirror of ProspectaProvider._build_embedder (kept in sync by test)."""
    kind = (cfg.get("embedder_kind") or "litellm").strip().lower()
    model = (cfg.get("embedder_model") or "").strip() or None
    if kind == "litellm":
        from prospecta.defaults import make_default_embedder
        return make_default_embedder(model)
    if kind == "sentence_transformers":
        from prospecta.embed import sentence_transformers
        return sentence_transformers(model or "all-MiniLM-L6-v2")
    if kind == "openai":
        from prospecta.embed import openai as openai_embed
        return openai_embed(model=model or "text-embedding-3-small")
    raise RuntimeError(f"Unknown embedder_kind: {kind!r}")


def check_database(url: str) -> Tuple[bool, str]:
    if not url:
        return False, "no database URL (set PROSPECTA_DATABASE_URL or DATABASE_URL)"
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "connected"
    except Exception as e:  # noqa: BLE001 - report anything
        return False, f"{type(e).__name__}: {e}".strip()


def check_embedder(cfg: dict) -> Tuple[bool, str]:
    try:
        embed = build_embedder(cfg)
        vecs = embed(["prospecta health check"])
        dim = len(vecs[0])
        want = int(cfg.get("embedding_dim", 1536))
        if dim != want:
            return False, f"embedder returns {dim} dims but embedding_dim={want}"
        return True, f"ok ({dim} dims)"
    except BaseException as e:  # noqa: BLE001 - ImportError, model load, network
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        return False, f"{type(e).__name__}: {e}".strip()


def run_health(cfg: Optional[dict] = None) -> Tuple[int, List[str], List[str]]:
    """Return (exit_code, ok_lines, failure_lines)."""
    cfg = _load_config() if cfg is None else cfg
    ok: List[str] = []
    bad: List[str] = []
    for name, (good, msg) in (
        ("database", check_database(resolve_database_url(cfg))),
        ("embedder", check_embedder(cfg)),
    ):
        (ok if good else bad).append(f"{name}: {msg}")
    return (1 if bad else 0), ok, bad


def main(argv: Optional[List[str]] = None) -> int:
    argparse.ArgumentParser(
        prog="hermes-prospecta-health",
        description="Exit non-zero (loudly) if the prospecta database or embedder is unavailable.",
    ).parse_args(argv)
    code, ok, bad = run_health()
    for line in ok:
        print(f"prospecta OK  {line}")
    if bad:
        names = ", ".join(b.split(":", 1)[0].upper() for b in bad)
        print(f"!!! PROSPECTA DOWN !!! unavailable: {names}", file=sys.stderr)
        for line in bad:
            print(f"!!! PROSPECTA DOWN !!!   {line}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
