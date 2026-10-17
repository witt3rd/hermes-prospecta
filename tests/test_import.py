"""Hindsight dump import: synthetic dump, idempotency, no-loss report."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, stub_embed, EMBED_DIM


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, str(REPO_ROOT / filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


importer = _load("importer", "importer.py")

DUMP = {
    "bank_id": "janus",
    "memory_units": [
        {"id": "u1", "text": "Dave prefers uv over pip.", "fact_type": "world",
         "context": "tooling chat", "created_at": "2025-03-01T10:00:00Z",
         "event_date": "2025-02-28T00:00:00Z", "document_id": "d9",
         "entities": ["Dave", "uv"], "tags": ["prefs"], "proof_count": 3,
         "weird_field": {"keep": "me"}},
        {"id": "u2", "text": "Dave is moving to Seattle.", "fact_type": "experience",
         "created_at": "2025-04-01T10:00:00Z", "entities": ["Dave"]},
        {"id": "u3", "text": "Dave likes uv and dislikes pip.", "fact_type": "observation",
         "created_at": "2025-05-01T10:00:00Z", "source_memory_ids": ["u1"]},
        # same text as u1, different id -> merged duplicate, not lost
        {"id": "u4", "text": "Dave prefers uv over pip.", "fact_type": "world",
         "created_at": "2025-06-01T10:00:00Z"},
    ],
    "entities": [
        {"id": "e1", "canonical_name": "Dave", "entity_type": "person",
         "aliases": ["dt"], "first_seen": "2025-01-01T00:00:00Z"},
    ],
    "links": [
        {"id": "l1", "from_id": "u1", "to_id": "e1", "link_type": "entity", "weight": 1.0},
        {"from_id": "u3", "to_id": "u1", "link_type": "derived_from", "weight": 0.5},
    ],
}


@pytest.fixture
def memory(pg_url):
    from prospecta.db.migrate import run_migrations
    from prospecta.memory import Memory
    run_migrations(database_url=pg_url)
    m = Memory(database_url=pg_url, bank_id="b", embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    yield m
    m.close()


def _docs(memory):
    with memory._pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT source, created_at, tags, document_metadata FROM documents "
                    "WHERE bank_id='b' ORDER BY source")
        return cur.fetchall()


def test_import_no_loss_and_fields(memory):
    r = importer.import_dump(memory, copy.deepcopy(DUMP))
    assert r["ok"], r["lost"]
    assert r["imported"] == 4 and r["merged_duplicates"] == 1 and r["failed"] == 0
    assert r["verified_present"] == {"units": 4, "entities": 1, "links": 2}
    docs = {d[0]: d for d in _docs(memory)}
    u1 = docs["hindsight://janus/unit/u1"]
    assert u1[1].isoformat().startswith("2025-03-01T10:00")
    assert "fact_type:world" in u1[2] and "entity:Dave" in u1[2] and "prefs" in u1[2]
    h = u1[3]["hindsight"]
    assert h["extra"] == {"weird_field": {"keep": "me"}}
    assert h["provenance"]["proof_count"] == 3
    assert {l["key"] for l in h["links"]} >= {"l1"}
    assert [a["id"] for a in h["import_aliases"]] == ["u4"]
    assert "hindsight://janus/unit/u4" not in docs
    ent = docs["hindsight://janus/entity/e1"][3]["hindsight"]
    assert ent["aliases"] == ["dt"] and len(ent["links"]) == 1


def _items(memory):
    """(source, kind, content, created_at) of every memory item the import wrote."""
    with memory._pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT d.source, m.kind, m.content, m.created_at FROM memory_items m "
                    "JOIN documents d ON d.id = m.document_id WHERE d.bank_id='b' ORDER BY d.source, m.kind")
        return cur.fetchall()


def test_import_writes_chunk_items_only(memory):
    """Stage 1 reads kind='chunk' items: an imported note is written as chunks, which is how the library
    writes every note now (no question item, no ``index_text``)."""
    importer.import_dump(memory, copy.deepcopy(DUMP))
    by_source: dict = {}
    for source, kind, content, created in _items(memory):
        by_source.setdefault(source, []).append((kind, content, created))
    assert sorted(by_source) == ["hindsight://janus/entity/e1", "hindsight://janus/unit/u1",
                                 "hindsight://janus/unit/u2", "hindsight://janus/unit/u3"]
    for source, rows in by_source.items():
        assert {k for k, _, _ in rows} == {"chunk"}, source
    (kind, content, created), = by_source["hindsight://janus/unit/u1"]
    assert content == "Dave prefers uv over pip."
    # the original timestamp is on the chunk item too (not only on the document)
    assert created.isoformat().startswith("2025-03-01T10:00")


def test_the_import_calls_retain_with_the_chunk_only_arguments(memory, monkeypatch):
    """Exactly what the library's retain takes: content, source, tags, metadata (no index_text, no child_chunks)."""
    from tests.conftest import check_call
    seen = []
    real = memory.retain

    def retain(content, **kw):
        check_call("retain", content, **kw)
        seen.append(set(kw))
        return real(content, **kw)

    monkeypatch.setattr(memory, "retain", retain)
    importer.import_dump(memory, copy.deepcopy(DUMP))
    assert seen and all(kw == {"source", "tags", "metadata"} for kw in seen)


def test_imported_notes_are_found_by_the_librarys_stage_1(memory):
    importer.import_dump(memory, copy.deepcopy(DUMP))
    hits = memory.search("Dave moving Seattle", limit=5)       # an empty channel_config is the hybrid_chunk default
    assert "hindsight://janus/unit/u2" in [h.source for h in hits]


def test_import_idempotent(memory):
    importer.import_dump(memory, copy.deepcopy(DUMP))
    before = _docs(memory)
    r = importer.import_dump(memory, copy.deepcopy(DUMP))
    assert r["ok"]
    assert r["imported"] == 0 and r["unchanged"] == 5
    assert _docs(memory) == before


def test_loss_is_reported(memory):
    dump = copy.deepcopy(DUMP)
    dump["memory_units"].append({"id": "bad", "text": "  "})
    dump["links"].append({"from_id": "ghost1", "to_id": "ghost2", "link_type": "x"})
    r = importer.import_dump(memory, dump)
    assert not r["ok"]
    assert any("unit bad" in x for x in r["lost"])
    assert any("neither endpoint" in x for x in r["lost"])
    assert "LOSS DETECTED" in importer.format_report(r)


def test_dry_run_writes_nothing(memory):
    r = importer.import_dump(memory, copy.deepcopy(DUMP), dry_run=True)
    assert r["imported"] == 5  # dry run cannot see intra-dump duplicates
    assert _docs(memory) == []


def test_cli_import(memory, pg_url, hermes_home, tmp_path, monkeypatch, capsys):
    cli = _load("hp_cli", "cli.py")
    f = tmp_path / "dump.json"
    f.write_text(json.dumps(DUMP))
    monkeypatch.setattr(cli, "_build_memory", lambda: memory)
    monkeypatch.setattr(memory, "close", lambda: None)
    import argparse
    p = argparse.ArgumentParser()
    cli.register_cli(p)
    assert cli._dispatch(p.parse_args(["import", str(f), "--json"])) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    f.write_text(json.dumps({"memory_units": [{"id": "x", "text": ""}]}))
    assert cli._dispatch(p.parse_args(["import", str(f)])) == 1
    assert "LOSS DETECTED" in capsys.readouterr().out
