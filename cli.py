"""hermes prospecta {status,stats,sweep-once,config,import} CLI subcommands.

Implementation strategy: shell out to the `prospecta` CLI (which already
exists in prospecta>=0.1.0) for stats/sweep, read the JSON config directly
for status/config. Avoids importing the full plugin module at CLI time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home)
    return Path.home() / ".hermes"


def _load_config() -> dict:
    cfg_path = _hermes_home() / "prospecta.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_database_url() -> str:
    cfg = _load_config()
    return (
        os.environ.get("PROSPECTA_DATABASE_URL")
        or cfg.get("database_url")
        or os.environ.get("DATABASE_URL")
        or ""
    )


def _resolve_bank_id() -> str:
    return os.environ.get("PROSPECTA_BANK_ID") or _load_config().get("bank_id", "default")


def _redact(values: dict) -> dict:
    """Redact secret-shaped fields in config view."""
    out = dict(values)
    if "database_url" in out and out["database_url"]:
        url = str(out["database_url"])
        # naive redaction of user:pass in postgres URL
        if "@" in url and "://" in url:
            scheme, rest = url.split("://", 1)
            if "@" in rest:
                _, host = rest.split("@", 1)
                out["database_url"] = f"{scheme}://***@{host}"
    return out


def _cmd_status(args) -> int:
    cfg = _load_config()
    db_url = _resolve_database_url()
    bank_id = _resolve_bank_id()
    mode = "byo" if db_url else ("embedded" if os.environ.get("PROSPECTA_ALLOW_EMBEDDED") == "1" else "unconfigured")
    print(f"prospecta status")
    print(f"  hermes_home : {_hermes_home()}")
    print(f"  mode        : {mode}")
    print(f"  database_url: {_redact({'database_url': db_url}).get('database_url') or '(unset)'}")
    print(f"  bank_id     : {bank_id}")
    print(f"  embedder    : {cfg.get('embedder_kind', 'litellm')}/{cfg.get('embedder_model') or '(default)'}")
    print(f"  embed_dim   : {cfg.get('embedding_dim', '1536')}")
    print(f"  prefetch    : {cfg.get('prefetch_enabled', 'false')}")
    return 0


def _cmd_config(args) -> int:
    cfg = _load_config()
    print(json.dumps(_redact(cfg), indent=2))
    return 0


def _run_prospecta_cli(extra_args: list[str]) -> int:
    """Shell out to the bundled `prospecta` CLI if available."""
    db_url = _resolve_database_url()
    env = os.environ.copy()
    if db_url and "DATABASE_URL" not in env:
        env["DATABASE_URL"] = db_url
    if db_url and "PROSPECTA_DATABASE_URL" not in env:
        env["PROSPECTA_DATABASE_URL"] = db_url

    exe = shutil.which("prospecta")
    if exe:
        cmd = [exe] + extra_args
    else:
        cmd = [sys.executable, "-m", "prospecta.cli"] + extra_args

    try:
        return subprocess.run(cmd, env=env).returncode
    except FileNotFoundError:
        print("prospecta CLI not available; ensure prospecta>=0.1.0 is installed", file=sys.stderr)
        return 2


def _cmd_stats(args) -> int:
    bank_id = _resolve_bank_id()
    return _run_prospecta_cli(["stats", "--bank", bank_id])


def _cmd_sweep_once(args) -> int:
    bank_id = _resolve_bank_id()
    return _run_prospecta_cli(["sweep", "--once", "--bank", bank_id])


def _build_embedder(cfg: dict):
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


def _build_memory():
    """Memory handle for offline writes (no LLM: imports pass index_text)."""
    from prospecta.db.migrate import run_migrations
    from prospecta.memory import Memory

    cfg = _load_config()
    url = _resolve_database_url()
    if not url:
        raise RuntimeError("no database url (set PROSPECTA_DATABASE_URL)")
    run_migrations(database_url=url)
    bank = _resolve_bank_id()
    mem = Memory(database_url=url, bank_id=bank, embed=_build_embedder(cfg))
    try:
        mem.create_bank(bank, embedding_dim=int(cfg.get("embedding_dim", 1536)))
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
    return mem


def _cmd_import(args) -> int:
    try:
        from .importer import format_report, import_dump, load_dump
    except ImportError:  # loaded as a top-level module / installed py-module
        from importer import format_report, import_dump, load_dump

    try:
        dump = load_dump(args.file)
        mem = _build_memory()
        try:
            report = import_dump(mem, dump, dry_run=args.dry_run)
        finally:
            mem.close()
    except Exception as e:
        print(f"IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0 if report["ok"] else 1


def _dispatch(args) -> int:
    sub = getattr(args, "prospecta_command", None)
    if sub == "status":
        return _cmd_status(args)
    if sub == "stats":
        return _cmd_stats(args)
    if sub == "sweep-once":
        return _cmd_sweep_once(args)
    if sub == "config":
        return _cmd_config(args)
    if sub == "import":
        return _cmd_import(args)
    print("usage: hermes prospecta {status,stats,sweep-once,config,import}", file=sys.stderr)
    return 2


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Hermes plugin CLI hook — wire `hermes prospecta ...` subcommands."""
    subs = parser.add_subparsers(dest="prospecta_command")
    subs.add_parser("status", help="Show provider connection + bank info")
    subs.add_parser("stats", help="Show document and event counters")
    subs.add_parser("sweep-once", help="Run one sweeper pass synchronously")
    subs.add_parser("config", help="Show loaded config (redacted)")
    imp = subs.add_parser("import", help="Import a Hindsight bank dump (JSON) into the bank")
    imp.add_argument("file", help="path to the dump JSON")
    imp.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    imp.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.set_defaults(func=_dispatch)


# Hermes discovery looks for `<plugin>_command` as the dispatch handler.
def prospecta_command(args) -> int:
    return _dispatch(args)
