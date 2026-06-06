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
    }
    assert keys == expected


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
