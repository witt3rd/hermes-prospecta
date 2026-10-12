"""bin: hermes-prospecta-fill-note-dates against a real Postgres (temporary container)."""
from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone

import psycopg
import pytest

import fill_note_dates as fnd
from tests.conftest import stub_embed

PASSWORD = "s3cretpw"


@pytest.fixture
def db(pg_url, monkeypatch):
    from prospecta.db.migrate import run_migrations
    from prospecta.memory import Memory
    run_migrations(database_url=pg_url)
    for bank in ("alpha", "beta"):
        Memory(database_url=pg_url, bank_id=bank, llm=None, embed=stub_embed).create_bank(
            bank, embedding_dim=32)
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent-profile")
    return pg_url


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def _add(url, bank, n, created_at, text="a note", created_on=None):
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(
            "INSERT INTO documents (bank_id, source, original_text, content_hash, created_at, created_on) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (bank, f"src:{n}", text, hashlib.sha256(f"{bank}{n}{text}".encode()).hexdigest(),
             created_at, created_on))


@pytest.fixture
def seeded(db):
    # 03:30 UTC on 15 July is the evening of 14 July in Los Angeles.
    _add(db, "alpha", 1, _utc(2026, 7, 15, 3, 30))
    _add(db, "alpha", 2, _utc(2026, 7, 15, 20, 0))
    _add(db, "alpha", 3, _utc(2026, 1, 15, 7, 30))
    _add(db, "alpha", 4, _utc(2026, 3, 1, 12, 0), created_on="2020-02-02")   # already dated
    _add(db, "alpha", 5, _utc(2026, 3, 2, 12, 0), text="   ")                # no text
    _add(db, "beta", 1, _utc(2026, 5, 5, 12, 0))
    return db


def _snapshot(url):
    with psycopg.connect(url) as c:
        return c.execute(
            "SELECT id, bank_id, source, original_text, content_hash, tags, document_metadata, "
            "retain_params, created_at, updated_at, person, source_kind FROM documents ORDER BY id").fetchall()


def _dates(url, bank="alpha"):
    with psycopg.connect(url) as c:
        return {r[0]: r[1] and r[1].isoformat() for r in c.execute(
            "SELECT source, created_on FROM documents WHERE bank_id = %s", (bank,))}


def _dump(tmp_path, *, age_hours=1.0, good_sum=True, sidecar=True, body=b"PGDMP fake"):
    d = tmp_path / "dumps"
    d.mkdir(exist_ok=True)
    f = d / "prospecta.dump"
    f.write_bytes(body)
    if sidecar:
        digest = hashlib.sha256(body if good_sum else b"other").hexdigest()
        (d / "prospecta.dump.sha256").write_text(f"{digest}  prospecta.dump\n")
    t = time.time() - age_hours * 3600
    os.utime(f, (t, t))
    return d


def _run(capsys, *argv):
    code = fnd.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


# ---- dry run ----

def test_dry_run_prints_counts_and_writes_nothing(seeded, capsys):
    before = _snapshot(seeded)
    code, out, err = _run(capsys, "--bank", "alpha", "--bank", "beta")
    assert code == 0 and err == ""
    assert "bank=alpha notes=5 already_dated=1 would_fill=3 no_text=1" in out
    assert "bank=beta notes=1 already_dated=0 would_fill=1 no_text=0" in out
    assert _snapshot(seeded) == before
    with psycopg.connect(seeded) as c:
        assert c.execute("SELECT to_regclass('note_date_fill_log')").fetchone()[0] is None


def test_dry_run_session_is_read_only(seeded):
    conn = fnd._connect(seeded, read_only=True)
    try:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("UPDATE documents SET created_on = now()::date")
    finally:
        conn.close()


def test_unknown_bank_is_refused(seeded, capsys):
    code, out, err = _run(capsys, "--bank", "nope")
    assert code == 1 and "unknown bank" in err


def test_needs_a_bank_or_undo(db, capsys):
    code, out, err = _run(capsys)
    assert code == 1 and "--bank" in err
    code, out, err = _run(capsys, "--undo")
    assert code == 1 and "--run-id" in err


# ---- the dump gate ----

@pytest.mark.parametrize("kw", [
    {"age_hours": 25}, {"good_sum": False}, {"sidecar": False}, {"body": b""}])
def test_apply_refuses_without_a_fresh_verified_dump(seeded, tmp_path, capsys, kw):
    before = _snapshot(seeded)
    code, out, err = _run(capsys, "--bank", "alpha", "--apply", "--dump-dir", str(_dump(tmp_path, **kw)))
    assert code == 1 and "refusing to write" in err
    assert _snapshot(seeded) == before


def test_apply_refuses_without_dump_dir_or_with_missing_dir(seeded, tmp_path, capsys):
    assert _run(capsys, "--bank", "alpha", "--apply")[0] == 1
    assert _run(capsys, "--bank", "alpha", "--apply", "--dump-dir", str(tmp_path / "none"))[0] == 1
    assert _dates(seeded)["src:1"] is None


def test_find_fresh_dump_picks_a_verified_recent_file(tmp_path):
    d = _dump(tmp_path, age_hours=23)
    assert fnd.find_fresh_dump(d).name == "prospecta.dump"
    assert fnd.find_fresh_dump(d, now=time.time() + 3 * 3600) is None  # 26 h old by then


# ---- apply ----

def test_apply_fills_local_dates_and_changes_nothing_else(seeded, tmp_path, capsys):
    before = _snapshot(seeded)
    d = _dump(tmp_path)
    code, out, err = _run(capsys, "--bank", "alpha", "--apply", "--dump-dir", str(d),
                          "--run-id", "r1", "--batch-size", "2")
    assert code == 0 and err == ""
    assert "bank=alpha notes=5 already_dated=1 filled=3 still_undated=0 no_text=1" in out
    assert _dates(seeded) == {
        "src:1": "2026-07-14",   # the local evening, not the UTC day
        "src:2": "2026-07-15",
        "src:3": "2026-01-14",   # winter offset
        "src:4": "2020-02-02",   # already dated: untouched
        "src:5": None,           # no text: skipped
    }
    assert _dates(seeded, "beta")["src:1"] is None  # not asked for
    after = _snapshot(seeded)
    assert after == before  # every column but created_on is identical (created_on is not in the snapshot)
    with psycopg.connect(seeded) as c:
        rows = c.execute("SELECT bank_id, run_id, filled_on::text FROM note_date_fill_log "
                         "ORDER BY filled_on").fetchall()
    assert rows == [("alpha", "r1", "2026-01-14"), ("alpha", "r1", "2026-07-14"), ("alpha", "r1", "2026-07-15")]


def test_timezone_option_and_profile_config(seeded, tmp_path, capsys, monkeypatch):
    d = _dump(tmp_path)
    _run(capsys, "--bank", "alpha", "--apply", "--dump-dir", str(d), "--timezone", "UTC")
    assert _dates(seeded)["src:1"] == "2026-07-15"
    assert _run(capsys, "--bank", "alpha", "--timezone", "Not/AZone")[0] == 1


def test_profile_timezone_is_used(seeded, tmp_path, capsys):
    prof = tmp_path / "prof"
    prof.mkdir()
    (prof / "prospecta.json").write_text('{"timezone": "Pacific/Auckland"}')
    code, out, _ = _run(capsys, "--bank", "alpha", "--profile-dir", str(prof))
    assert "timezone: Pacific/Auckland" in out


def test_apply_is_idempotent_and_resumable(seeded, tmp_path, capsys):
    d = _dump(tmp_path)
    args = ["--bank", "alpha", "--apply", "--dump-dir", str(d), "--run-id", "r1", "--batch-size", "1"]
    _run(capsys, *args)
    snap = _dates(seeded)
    _add(seeded, "alpha", 6, _utc(2026, 8, 1, 12, 0))            # a note that arrived meanwhile
    code, out, _ = _run(capsys, *args)
    assert "filled=1 " in out and _dates(seeded)["src:6"] == "2026-08-01"
    assert {k: v for k, v in _dates(seeded).items() if k != "src:6"} == snap
    code, out, _ = _run(capsys, *args)
    assert "filled=0 " in out
    with psycopg.connect(seeded) as c:
        assert c.execute("SELECT count(*) FROM note_date_fill_log").fetchone()[0] == 4


# ---- undo ----

def _fill(capsys, seeded, tmp_path, bank, run):
    assert _run(capsys, "--bank", bank, "--apply", "--dump-dir", str(_dump(tmp_path)), "--run-id", run)[0] == 0


def test_undo_one_run_only(seeded, tmp_path, capsys):
    original = _snapshot(seeded)
    _fill(capsys, seeded, tmp_path, "alpha", "r1")
    _fill(capsys, seeded, tmp_path, "beta", "r2")
    d = str(_dump(tmp_path))
    code, out, _ = _run(capsys, "--undo", "--run-id", "r1")  # dry run
    assert "logged=3 would_clear=3" in out and _dates(seeded)["src:1"] == "2026-07-14"
    code, out, err = _run(capsys, "--undo", "--run-id", "r1", "--apply", "--dump-dir", d, "--batch-size", "2")
    assert code == 0 and "cleared=3 log_rows_removed=3" in out
    assert _dates(seeded) == {"src:1": None, "src:2": None, "src:3": None, "src:4": "2020-02-02", "src:5": None}
    assert _dates(seeded, "beta")["src:1"] == "2026-05-05"
    assert _snapshot(seeded) == original
    with psycopg.connect(seeded) as c:
        assert c.execute("SELECT run_id FROM note_date_fill_log").fetchall() == [("r2",)]
    code, out, _ = _run(capsys, "--undo", "--run-id", "r1", "--apply", "--dump-dir", d)
    assert "cleared=0 log_rows_removed=0" in out  # idempotent


def test_undo_all_and_a_date_changed_since_is_left_alone(seeded, tmp_path, capsys):
    _fill(capsys, seeded, tmp_path, "alpha", "r1")
    _fill(capsys, seeded, tmp_path, "beta", "r2")
    with psycopg.connect(seeded, autocommit=True) as c:
        c.execute("UPDATE documents SET created_on = '2001-01-01' WHERE bank_id = 'beta'")
    code, out, _ = _run(capsys, "--undo-all", "--apply", "--dump-dir", str(_dump(tmp_path)))
    assert "cleared=3 log_rows_removed=4" in out
    assert _dates(seeded, "beta")["src:1"] == "2001-01-01"
    with psycopg.connect(seeded) as c:
        assert c.execute("SELECT count(*) FROM note_date_fill_log").fetchone()[0] == 0


def test_undo_without_a_log_table_is_a_no_op(seeded, capsys):
    code, out, _ = _run(capsys, "--undo-all")
    assert code == 0 and "logged=0" in out


def test_undo_apply_needs_the_dump_too(seeded, tmp_path, capsys):
    _fill(capsys, seeded, tmp_path, "alpha", "r1")
    code, out, err = _run(capsys, "--undo", "--run-id", "r1", "--apply")
    assert code == 1 and _dates(seeded)["src:1"] == "2026-07-14"


# ---- the address stays secret ----

def test_output_never_carries_the_address_or_note_text(db, tmp_path, capsys, monkeypatch):
    bad = f"postgresql://someone:{PASSWORD}@127.0.0.1:1/secretdb"
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", bad)
    code, out, err = _run(capsys, "--bank", "alpha")
    assert code == 1
    for text in (out, err):
        assert PASSWORD not in text and "secretdb" not in text and "127.0.0.1" not in text
    assert "OperationalError" in err


def test_success_output_has_no_address(seeded, capsys):
    code, out, err = _run(capsys, "--bank", "alpha")
    assert code == 0 and seeded not in out + err and "postgresql" not in out + err


def test_address_resolution_order_and_dotenv(tmp_path, monkeypatch):
    prof = tmp_path
    for k in ("PROSPECTA_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)
    assert fnd.resolve_database_url(prof) == ""
    (prof / "prospecta.json").write_text('{"database_url": "postgresql://from-json/db"}')
    assert fnd.resolve_database_url(prof) == "postgresql://from-json/db"
    (prof / ".env").write_text(
        "# comment\nexport OTHER=1\nPROSPECTA_DATABASE_URL='postgresql://from-dotenv/db'\n")
    assert fnd.resolve_database_url(prof) == "postgresql://from-dotenv/db"
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", "postgresql://from-env/db")
    assert fnd.resolve_database_url(prof) == "postgresql://from-env/db"
