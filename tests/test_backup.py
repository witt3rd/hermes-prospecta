"""Backup/restore round trip against a real pgvector Postgres."""
from __future__ import annotations

import psycopg
import pytest

import backup


def _make_db(pg_container, name):
    from urllib.parse import urlparse, urlunparse
    base = pg_container.get_connection_url().replace("+psycopg2", "").replace("+psycopg", "")
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    return urlunparse(urlparse(base)._replace(path=f"/{name}"))


def _seed(url):
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute("CREATE TABLE memo (id int primary key, body text, emb vector(3))")
        c.execute("INSERT INTO memo VALUES (1,'alpha','[1,2,3]'),(2,'beta','[4,5,6]')")


def _rows(url):
    with psycopg.connect(url) as c:
        return c.execute("SELECT id, body, emb::text FROM memo ORDER BY id").fetchall()


def test_backup_restore_round_trip(pg_url, pg_container, tmp_path, monkeypatch):
    _seed(pg_url)
    dump = tmp_path / "sub" / "p.dump"
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert backup.backup_main([str(dump)]) == 0
    assert dump.stat().st_size > 0
    want = _rows(pg_url)

    target = _make_db(pg_container, "restore_target_h4")
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", target)
    assert backup.restore_main([str(dump)]) == 0
    assert _rows(target) == want

    # Second restore without --clean fails atomically; with --clean succeeds.
    with psycopg.connect(target, autocommit=True) as c:
        c.execute("DELETE FROM memo WHERE id=2")
    assert backup.restore_main([str(dump)]) == 1
    assert len(_rows(target)) == 1
    assert backup.restore_main(["--clean", str(dump)]) == 0
    assert _rows(target) == want


def test_backup_failure_is_loud_and_keeps_old_file(tmp_path, monkeypatch, capsys):
    dump = tmp_path / "keep.dump"
    dump.write_bytes(b"good")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nope")
    assert backup.backup_main([str(dump)]) == 1
    assert "!!! PROSPECTA BACKUP FAILED !!!" in capsys.readouterr().err
    assert dump.read_bytes() == b"good"
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".part"] == []


def test_restore_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", "postgresql://u:p@127.0.0.1:1/nope")
    assert backup.restore_main([str(tmp_path / "absent")]) == 1
    assert "RESTORE FAILED" in capsys.readouterr().err


def test_no_url(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert backup.backup_main([str(tmp_path / "x")]) == 1
    assert "no database URL" in capsys.readouterr().err
