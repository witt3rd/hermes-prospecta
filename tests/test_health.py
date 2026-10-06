"""hermes-prospecta-health: both outcomes (healthy / database or embedder down)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import health  # noqa: E402

DEAD_URL = "postgresql://nobody:nopw@127.0.0.1:1/none"


def _stub(texts):
    return [[0.1] * 8 for _ in texts]


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for k in ("PROSPECTA_DATABASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(k, raising=False)


def _cfg(tmp_path, **kw):
    import json
    (tmp_path / "prospecta.json").write_text(json.dumps(kw))


def test_database_down_is_loud_and_nonzero(tmp_path, monkeypatch, capsys):
    _cfg(tmp_path, database_url=DEAD_URL, embedding_dim=8)
    monkeypatch.setattr(health, "build_embedder", lambda cfg: _stub)
    assert health.main([]) == 1
    err = capsys.readouterr().err
    assert "!!! PROSPECTA DOWN !!!" in err and "DATABASE" in err
    assert "EMBEDDER" not in err.splitlines()[0]


def test_no_database_url_is_down(monkeypatch, capsys):
    monkeypatch.setattr(health, "build_embedder", lambda cfg: _stub)
    assert health.main([]) == 1
    assert "no database URL" in capsys.readouterr().err


def test_embedder_down_is_loud_and_nonzero(tmp_path, monkeypatch, capsys):
    _cfg(tmp_path, embedder_kind="bogus", embedding_dim=8)
    monkeypatch.setattr(health, "check_database", lambda url: (True, "connected"))
    assert health.main([]) == 1
    cap = capsys.readouterr()
    assert "!!! PROSPECTA DOWN !!!" in cap.err and "EMBEDDER" in cap.err
    assert "Unknown embedder_kind" in cap.err
    assert "database: connected" in cap.out


def test_embedder_dimension_mismatch_is_down(tmp_path, monkeypatch, capsys):
    _cfg(tmp_path, embedding_dim=1536)
    monkeypatch.setattr(health, "check_database", lambda url: (True, "ok"))
    monkeypatch.setattr(health, "build_embedder", lambda cfg: _stub)
    assert health.main([]) == 1
    assert "8 dims" in capsys.readouterr().err


def test_embedder_import_error_is_down(tmp_path, monkeypatch, capsys):
    _cfg(tmp_path, embedder_kind="sentence_transformers")
    monkeypatch.setattr(health, "check_database", lambda url: (True, "ok"))

    def boom(cfg):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(health, "build_embedder", boom)
    assert health.main([]) == 1
    assert "sentence_transformers" in capsys.readouterr().err


def test_healthy_exits_zero(pg_url, tmp_path, monkeypatch, capsys):
    _cfg(tmp_path, database_url=pg_url, embedding_dim=8)
    monkeypatch.setattr(health, "build_embedder", lambda cfg: _stub)
    assert health.main([]) == 0
    cap = capsys.readouterr()
    assert "prospecta OK  database" in cap.out and "prospecta OK  embedder" in cap.out
    assert "DOWN" not in cap.err
