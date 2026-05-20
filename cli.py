"""hermes prospecta {status,stats,sweep-once,config} CLI subcommands.

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
    return (
        os.environ.get("DATABASE_URL")
        or _load_config().get("database_url")
        or ""
    )


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
    bank_id = cfg.get("bank_id", "default")
    mode = "byo" if (os.environ.get("DATABASE_URL") or cfg.get("database_url")) else "embedded"
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
    cfg = _load_config()
    bank_id = cfg.get("bank_id", "default")
    return _run_prospecta_cli(["stats", "--bank", bank_id])


def _cmd_sweep_once(args) -> int:
    cfg = _load_config()
    bank_id = cfg.get("bank_id", "default")
    return _run_prospecta_cli(["sweep", "--once", "--bank", bank_id])


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
    print("usage: hermes prospecta {status,stats,sweep-once,config}", file=sys.stderr)
    return 2


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Hermes plugin CLI hook — wire `hermes prospecta ...` subcommands."""
    subs = parser.add_subparsers(dest="prospecta_command")
    subs.add_parser("status", help="Show provider connection + bank info")
    subs.add_parser("stats", help="Show document and event counters")
    subs.add_parser("sweep-once", help="Run one sweeper pass synchronously")
    subs.add_parser("config", help="Show loaded config (redacted)")
    parser.set_defaults(func=_dispatch)


# Hermes discovery looks for `<plugin>_command` as the dispatch handler.
def prospecta_command(args) -> int:
    return _dispatch(args)
