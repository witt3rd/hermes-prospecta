"""hermes-prospecta-backup / hermes-prospecta-restore: durable Postgres dumps.

``backup PATH`` runs ``pg_dump`` (custom format, compressed) of the whole
prospecta database into PATH (written to a temp file, then renamed, so a
failed dump never clobbers a good backup). ``restore PATH`` loads such a file
with ``pg_restore`` in a single transaction.

The database URL is resolved like the plugin does (PROSPECTA_DATABASE_URL,
``$HERMES_HOME/prospecta.json``, DATABASE_URL). Credentials are handed to the
PostgreSQL tools through PG* environment variables, never argv.

Requires ``pg_dump`` / ``pg_restore`` on PATH (client version >= server's).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote, urlparse

try:
    from .health import _load_config, resolve_database_url
except ImportError:  # loaded as a top-level module / installed py-module
    from health import _load_config, resolve_database_url


def _pg_env(url: str) -> dict:
    """Split a postgres URL into PG* env vars (keeps the password off argv)."""
    p = urlparse(url)
    if p.scheme not in ("postgres", "postgresql") or not p.path.strip("/"):
        raise ValueError("database URL must look like postgresql://user:pass@host:port/dbname")
    env = os.environ.copy()
    env["PGDATABASE"] = unquote(p.path.lstrip("/"))
    if p.hostname:
        env["PGHOST"] = p.hostname
    if p.port:
        env["PGPORT"] = str(p.port)
    if p.username:
        env["PGUSER"] = unquote(p.username)
    if p.password:
        env["PGPASSWORD"] = unquote(p.password)
    for part in (p.query or "").split("&"):
        if part.startswith("sslmode="):
            env["PGSSLMODE"] = part.split("=", 1)[1]
    return env


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"{name} not found on PATH; install the PostgreSQL client tools")
    return exe


def backup(url: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        res = subprocess.run(
            [_tool("pg_dump"), "--format=custom", "--no-owner", "--no-privileges", "--file", tmp],
            env=_pg_env(url), capture_output=True, text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"pg_dump failed ({res.returncode}): {res.stderr.strip()}")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def restore(url: str, path: Path, clean: bool = False) -> None:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"backup file not found: {path}")
    env = _pg_env(url)
    # pg_restore -> SQL -> psql, rather than pg_restore --dbname: a newer
    # client (pg_dump 17+) emits `SET transaction_timeout`, which older servers
    # reject. We drop only that line, and run psql in one transaction.
    cmd = [_tool("pg_restore"), "--no-owner", "--no-privileges", "--file", "-"]
    if clean:
        cmd += ["--clean", "--if-exists"]
    cmd.append(str(path))
    psql_cmd = [_tool("psql"), "-X", "-q", "-1", "-v", "ON_ERROR_STOP=1", "-o", os.devnull]
    with tempfile.TemporaryFile() as err_r, tempfile.TemporaryFile() as err_p:
        reader = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=err_r)
        psql = subprocess.Popen(psql_cmd, env=env, stdin=subprocess.PIPE, stderr=err_p)
        try:
            for line in reader.stdout:
                if line.strip() == b"SET transaction_timeout = 0;":
                    continue
                psql.stdin.write(line)
        except BrokenPipeError:
            pass  # psql died on error; reported below
        finally:
            reader.stdout.close()
            try:
                psql.stdin.close()
            except BrokenPipeError:
                pass
        rc_r, rc_p = reader.wait(), psql.wait()
        if rc_r != 0 or rc_p != 0:
            err_r.seek(0)
            err_p.seek(0)
            msg = (err_r.read() + err_p.read()).decode(errors="replace").strip()
            raise RuntimeError(f"restore failed (pg_restore={rc_r}, psql={rc_p}): {msg}")


def _url_or_die() -> str:
    url = resolve_database_url(_load_config())
    if not url:
        raise RuntimeError("no database URL (set PROSPECTA_DATABASE_URL or DATABASE_URL)")
    return url


def backup_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="hermes-prospecta-backup",
                                 description="pg_dump the prospecta database to PATH.")
    ap.add_argument("path", type=Path)
    args = ap.parse_args(argv)
    try:
        out = backup(_url_or_die(), args.path)
    except Exception as e:  # noqa: BLE001
        print(f"!!! PROSPECTA BACKUP FAILED !!! {e}", file=sys.stderr)
        return 1
    print(f"prospecta backup OK  {out} ({out.stat().st_size} bytes)")
    return 0


def restore_main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hermes-prospecta-restore",
        description="pg_restore a backup into the prospecta database (single transaction).")
    ap.add_argument("path", type=Path)
    ap.add_argument("--clean", action="store_true",
                    help="drop existing objects first (overwrites current data); without it "
                         "restore fails, changing nothing, if objects already exist")
    args = ap.parse_args(argv)
    try:
        restore(_url_or_die(), args.path, clean=args.clean)
    except Exception as e:  # noqa: BLE001
        print(f"!!! PROSPECTA RESTORE FAILED !!! {e}", file=sys.stderr)
        return 1
    print(f"prospecta restore OK  {args.path}")
    return 0
