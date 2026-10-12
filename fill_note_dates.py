"""hermes-prospecta-fill-note-dates: fill documents.created_on for old notes.

New notes get their date from the plugin (see README, "Note dates"). Notes
saved before that have ``documents.created_on`` empty, so the library's date
filter ("in July", an ISO date, a year) cannot find them. This tool fills the
date from each note's own recorded time (``documents.created_at``), converted
to the calendar day in the profile's timezone.

* DRY RUN by default: reads only (a read-only transaction) and prints counts.
* ``--apply`` writes, and only after it finds a fresh verified dump in
  ``--dump-dir`` (a file less than 24 hours old whose ``.sha256`` matches).
* Additive: only ``documents.created_on`` of notes where it is empty is set.
  Nothing else is changed and nothing is deleted. Every filled id is recorded
  in ``note_date_fill_log`` in the same transaction as the update.
* ``--undo --run-id X`` (or ``--undo-all``) sets ``created_on`` back to empty
  for the ids that run filled and clears their log rows.
* Batched, resumable (re-run to continue) and idempotent.
* Prints counts only: never the database address, never note text.

The database address comes from the environment (PROSPECTA_DATABASE_URL), the
profile ``.env``, ``prospecta.json``, or DATABASE_URL, like the plugin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_TIMEZONE = "America/Los_Angeles"
MAX_DUMP_AGE_SECONDS = 24 * 3600
DEFAULT_BATCH = 500

_DDL = """
CREATE TABLE IF NOT EXISTS note_date_fill_log (
    bank_id      TEXT        NOT NULL,
    document_id  UUID        NOT NULL,
    filled_on    DATE        NOT NULL,
    filled_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id       TEXT        NOT NULL,
    PRIMARY KEY (bank_id, document_id)
);
CREATE INDEX IF NOT EXISTS note_date_fill_log_run_idx ON note_date_fill_log (run_id);
"""

_COUNTS = """
SELECT count(*),
       count(*) FILTER (WHERE created_on IS NOT NULL),
       count(*) FILTER (WHERE created_on IS NULL AND btrim(original_text) <> ''),
       count(*) FILTER (WHERE created_on IS NULL AND btrim(original_text) = '')
FROM documents WHERE bank_id = %s
"""

# One statement = one transaction: the update and its log rows land together.
_FILL_BATCH = """
WITH picked AS (
    SELECT id, (created_at AT TIME ZONE %(tz)s)::date AS day
    FROM documents
    WHERE bank_id = %(bank)s AND created_on IS NULL AND btrim(original_text) <> ''
    ORDER BY id LIMIT %(n)s FOR UPDATE SKIP LOCKED
), upd AS (
    UPDATE documents d SET created_on = p.day
    FROM picked p WHERE d.id = p.id AND d.created_on IS NULL
    RETURNING d.id, d.created_on
), logged AS (
    INSERT INTO note_date_fill_log (bank_id, document_id, filled_on, filled_at, run_id)
    SELECT %(bank)s, id, created_on, now(), %(run)s FROM upd
    ON CONFLICT (bank_id, document_id) DO UPDATE
        SET filled_on = EXCLUDED.filled_on, filled_at = EXCLUDED.filled_at, run_id = EXCLUDED.run_id
    RETURNING 1
)
SELECT (SELECT count(*) FROM upd), (SELECT count(*) FROM logged)
"""

# Only a date still equal to the one this tool wrote is cleared (a date set by
# anything else since is left alone); the log row goes either way.
_UNDO_BATCH = """
WITH picked AS (
    SELECT bank_id, document_id, filled_on FROM note_date_fill_log
    WHERE (%(run)s::text IS NULL OR run_id = %(run)s) AND (%(banks)s::text[] IS NULL OR bank_id = ANY(%(banks)s))
    ORDER BY bank_id, document_id LIMIT %(n)s FOR UPDATE
), rev AS (
    UPDATE documents d SET created_on = NULL
    FROM picked p WHERE d.id = p.document_id AND d.bank_id = p.bank_id AND d.created_on = p.filled_on
    RETURNING d.id
), gone AS (
    DELETE FROM note_date_fill_log f USING picked p
    WHERE f.bank_id = p.bank_id AND f.document_id = p.document_id
    RETURNING 1
)
SELECT (SELECT count(*) FROM rev), (SELECT count(*) FROM gone)
"""

_UNDO_COUNTS = """
SELECT count(*), count(*) FILTER (WHERE d.created_on = f.filled_on)
FROM note_date_fill_log f LEFT JOIN documents d ON d.id = f.document_id AND d.bank_id = f.bank_id
WHERE (%(run)s::text IS NULL OR f.run_id = %(run)s) AND (%(banks)s::text[] IS NULL OR f.bank_id = ANY(%(banks)s))
"""


class Refusal(Exception):
    """A condition under which the tool must not write; the message has no secrets."""


# ---- configuration ---------------------------------------------------------

def _hermes_home() -> Path:
    home = os.environ.get("HERMES_HOME")
    return Path(home) if home else Path.home() / ".hermes"


def read_dotenv(path: Path) -> Dict[str, str]:
    """KEY=VALUE lines (optional ``export``, optional quotes); never raises."""
    out: Dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_database_url(profile: Path) -> str:
    """Environment first, then the profile .env, prospecta.json, DATABASE_URL."""
    env_file = read_dotenv(Path(profile) / ".env")
    cfg = _read_json(Path(profile) / "prospecta.json")
    for value in (
        os.environ.get("PROSPECTA_DATABASE_URL"),
        env_file.get("PROSPECTA_DATABASE_URL"),
        cfg.get("database_url"),
        os.environ.get("DATABASE_URL"),
        env_file.get("DATABASE_URL"),
    ):
        if value and str(value).strip():
            return str(value).strip()
    return ""


def resolve_timezone(profile: Path, given: Optional[str]) -> str:
    name = (given or str(_read_json(Path(profile) / "prospecta.json").get("timezone") or "")).strip()
    name = name or DEFAULT_TIMEZONE
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(name)
    except Exception:
        raise Refusal(f"unknown timezone {name!r}")
    return name


# ---- the dump gate ---------------------------------------------------------

def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_fresh_dump(dump_dir: Path, now: Optional[float] = None) -> Optional[Path]:
    """The newest file in dump_dir that is under 24 hours old, non-empty, and
    whose ``<file>.sha256`` (``sha256sum`` format) matches its contents."""
    now = time.time() if now is None else now
    dump_dir = Path(dump_dir)
    try:
        entries = sorted(dump_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for path in entries:
        try:
            if not path.is_file() or path.name.endswith(".sha256"):
                continue
            st = path.stat()
            age = now - st.st_mtime
            if st.st_size == 0 or age > MAX_DUMP_AGE_SECONDS or age < -300:
                continue
            sidecar = path.with_name(path.name + ".sha256")
            tokens = sidecar.read_text(encoding="utf-8").split()
            if not tokens or tokens[0].lower() != _sha256_of(path):
                continue
            return path
        except OSError:
            continue
    return None


def require_fresh_dump(dump_dir: Optional[str], now: Optional[float] = None) -> Path:
    if not dump_dir:
        raise Refusal("refusing to write: --dump-dir is required with --apply")
    found = find_fresh_dump(Path(dump_dir), now)
    if found is None:
        raise Refusal("refusing to write: no dump under 24 hours old with a matching .sha256 "
                      "in the --dump-dir")
    return found


# ---- the work --------------------------------------------------------------

def bank_counts(conn, bank: str) -> Dict[str, int]:
    if conn.execute("SELECT 1 FROM banks WHERE bank_id = %s", (bank,)).fetchone() is None:
        raise Refusal(f"unknown bank {bank!r}")
    total, dated, fill, no_text = conn.execute(_COUNTS, (bank,)).fetchone()
    return {"total": total, "already_dated": dated, "would_fill": fill, "no_text": no_text}


def fill_bank(conn, bank: str, tz: str, run_id: str, batch: int) -> int:
    """Fill in batches, one transaction each; returns the number of notes filled."""
    done = 0
    while True:
        with conn.transaction():
            n, logged = conn.execute(
                _FILL_BATCH, {"tz": tz, "bank": bank, "n": batch, "run": run_id}).fetchone()
        if n != logged:  # cannot happen (same statement); stop rather than go on
            raise RuntimeError("fill log out of step with the update")
        if n == 0:
            return done
        done += n


def undo_runs(conn, run_id: Optional[str], banks: Optional[List[str]], batch: int) -> Tuple[int, int]:
    """Returns (dates cleared, log rows removed)."""
    cleared = removed = 0
    args = {"run": run_id, "banks": banks or None, "n": batch}
    while True:
        with conn.transaction():
            rev, gone = conn.execute(_UNDO_BATCH, args).fetchone()
        if gone == 0:
            return cleared, removed
        cleared += rev
        removed += gone


def _log_table_exists(conn) -> bool:
    return conn.execute("SELECT to_regclass('note_date_fill_log') IS NOT NULL").fetchone()[0]


def _connect(url: str, *, read_only: bool):
    import psycopg
    conn = psycopg.connect(url, connect_timeout=10, autocommit=not read_only)
    if read_only:
        conn.read_only = True  # the server refuses any write in this session
    return conn


def _new_run_id() -> str:
    return "fill-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hermes-prospecta-fill-note-dates",
        description="Fill documents.created_on for existing notes from their recorded time. "
                    "Dry run unless --apply. Prints counts only.")
    p.add_argument("--bank", action="append", default=[], metavar="BANK",
                   help="bank id to process (repeatable)")
    p.add_argument("--apply", action="store_true",
                   help="write (needs a fresh verified dump in --dump-dir); default is a dry run")
    p.add_argument("--dump-dir", metavar="DIR",
                   help="directory holding a dump under 24 hours old with a matching .sha256")
    p.add_argument("--timezone", metavar="ZONE",
                   help="IANA zone for the local day (default: prospecta.json timezone, else "
                        f"{DEFAULT_TIMEZONE})")
    p.add_argument("--run-id", metavar="ID",
                   help="name of this run in the log (default: generated); with --undo, the run to undo")
    p.add_argument("--undo", action="store_true", help="undo a run: needs --run-id (or use --undo-all)")
    p.add_argument("--undo-all", action="store_true", help="undo every logged fill")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, metavar="N",
                   help=f"notes per transaction (default {DEFAULT_BATCH})")
    p.add_argument("--profile-dir", metavar="DIR",
                   help="profile directory holding .env and prospecta.json (default $HERMES_HOME or ~/.hermes)")
    return p


def run(args: argparse.Namespace, out=None, now: Optional[float] = None) -> int:
    out = out or sys.stdout
    say = lambda s: print(s, file=out)  # noqa: E731
    undo = args.undo or args.undo_all
    if args.batch_size < 1:
        raise Refusal("--batch-size must be at least 1")
    if undo:
        if args.undo and args.undo_all:
            raise Refusal("use --undo --run-id X or --undo-all, not both")
        if args.undo and not args.run_id:
            raise Refusal("--undo needs --run-id (or use --undo-all)")
    elif not args.bank:
        raise Refusal("give at least one --bank")

    profile = Path(args.profile_dir) if args.profile_dir else _hermes_home()
    url = resolve_database_url(profile)
    if not url:
        raise Refusal("no database address: set PROSPECTA_DATABASE_URL (environment or the profile .env)")
    tz = resolve_timezone(profile, args.timezone)

    if args.apply:  # the gate runs before any connection that could write
        dump = require_fresh_dump(args.dump_dir, now)
        say(f"dump: fresh and verified ({dump.name})")
    mode = "APPLY" if args.apply else "DRY RUN (nothing is written)"

    try:
        conn = _connect(url, read_only=not args.apply)
    except Exception as e:
        raise Refusal(f"could not connect to the database ({type(e).__name__})")
    try:
        if undo:
            return _run_undo(conn, args, say, mode)
        return _run_fill(conn, args, tz, say, mode)
    finally:
        conn.close()


def _run_fill(conn, args, tz: str, say, mode: str) -> int:
    banks = list(dict.fromkeys(args.bank))
    say(f"mode: {mode}; timezone: {tz}")
    if not args.apply:
        for bank in banks:
            c = bank_counts(conn, bank)
            say(f"bank={bank} notes={c['total']} already_dated={c['already_dated']} "
                f"would_fill={c['would_fill']} no_text={c['no_text']}")
        conn.rollback()
        return 0
    for bank in banks:  # check every bank before the first write
        bank_counts(conn, bank)
    run_id = args.run_id or _new_run_id()
    conn.execute(_DDL)
    say(f"run_id: {run_id}")
    for bank in banks:
        before = bank_counts(conn, bank)
        filled = fill_bank(conn, bank, tz, run_id, args.batch_size)
        after = bank_counts(conn, bank)
        say(f"bank={bank} notes={before['total']} already_dated={before['already_dated']} "
            f"filled={filled} still_undated={after['would_fill']} no_text={after['no_text']}")
    return 0


def _run_undo(conn, args, say, mode: str) -> int:
    run_id = None if args.undo_all else args.run_id
    banks = list(dict.fromkeys(args.bank)) or None
    say(f"mode: UNDO {'APPLY' if args.apply else 'DRY RUN (nothing is written)'}; "
        f"run: {run_id or 'all'}")
    if not _log_table_exists(conn):
        say("logged=0 would_clear=0")
        return 0
    if not args.apply:
        logged, matching = conn.execute(_UNDO_COUNTS, {"run": run_id, "banks": banks}).fetchone()
        say(f"logged={logged} would_clear={matching}")
        conn.rollback()
        return 0
    cleared, removed = undo_runs(conn, run_id, banks, args.batch_size)
    say(f"cleared={cleared} log_rows_removed={removed}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Refusal as e:
        print(f"fill-note-dates: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # counts only: no exception text (it can carry the address)
        print(f"fill-note-dates: failed ({type(e).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
