"""Config schema + save_config tests."""
from __future__ import annotations

import json


def test_get_config_schema_has_expected_fields(plugin_module):
    p = plugin_module.ProspectaProvider()
    schema = p.get_config_schema()
    keys = {f["key"] for f in schema}
    expected = {
        "database_url", "bank_id", "embedder_kind", "embedder_model",
        "llm_model", "embedding_dim", "prefetch_enabled",
        "shadow_bank_id", "shadow_embed_model", "shadow_embedding_dim",
        "dual_write_bank_id", "dual_write_embed_model",
        "dual_write_embedding_dim", "dual_write_retry_path", "timezone", "hash_precheck",
        "max_connections", "tool_guide", "pool_idle_seconds", "connection_watch",
        "connection_watch_percent", "decision_model", "query_writer_model", "reranker_model", "recall.stage3",
        "recall.prefetch.mode", "recall.prefetch.contract_path", "recall.prefetch.max_tokens",
    }
    assert keys == expected
    gone = {"note_date", "recall.fusion", "recall.block_tokens", "recall.keyword", "recall.keyword_unit",
            "recall.keyword_index_warn_mb", "recall.keyword_batch_pause_ms", "recall.prefetch.decision_model",
            "recall.prefetch.question_model",
            "linked_writes", "shadow_mode", "recall.date_cue_gate", "recall.date_timeout_seconds"}
    # the plugin's own lanes, the note-date switch, the old model keys and the settings of removed code are gone
    assert not gone & keys


def test_save_config_writes_json(plugin_module, tmp_path):
    p = plugin_module.ProspectaProvider()
    p.save_config(
        {"bank_id": "myproj", "embedder_kind": "openai", "embedding_dim": "1536"},
        str(tmp_path),
    )
    cfg_path = tmp_path / "prospecta.json"
    assert cfg_path.exists()
    data = json.loads(cfg_path.read_text())
    assert data["bank_id"] == "myproj"
    assert data["embedder_kind"] == "openai"


def test_save_config_merges_existing(plugin_module, tmp_path):
    p = plugin_module.ProspectaProvider()
    cfg_path = tmp_path / "prospecta.json"
    cfg_path.write_text('{"bank_id": "old", "embedding_dim": "384"}')
    p.save_config({"bank_id": "new"}, str(tmp_path))
    data = json.loads(cfg_path.read_text())
    assert data["bank_id"] == "new"
    assert data["embedding_dim"] == "384"  # preserved
