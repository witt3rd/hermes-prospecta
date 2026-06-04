"""Environment/config precedence for platform-shaped Prospecta provider setup."""
from __future__ import annotations

import pytest


def test_database_url_prefers_prospecta_specific_env(plugin_module, tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", "postgresql://prospecta-specific/db")
    monkeypatch.setenv("DATABASE_URL", "postgresql://generic/db")
    cfg_path = tmp_path / "prospecta.json"
    cfg_path.write_text('{"database_url": "postgresql://config/db"}')

    provider = plugin_module.ProspectaProvider()
    provider._hermes_home = tmp_path
    provider._config = provider._load_config()

    assert provider._resolve_database_url() == "postgresql://prospecta-specific/db"


def test_database_url_uses_config_before_generic_env(plugin_module, tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://generic/db")
    cfg_path = tmp_path / "prospecta.json"
    cfg_path.write_text('{"database_url": "postgresql://config/db"}')

    provider = plugin_module.ProspectaProvider()
    provider._hermes_home = tmp_path
    provider._config = provider._load_config()

    assert provider._resolve_database_url() == "postgresql://config/db"


def test_bank_id_prefers_prospecta_specific_env(plugin_module, tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPECTA_BANK_ID", "env-bank")
    cfg_path = tmp_path / "prospecta.json"
    cfg_path.write_text('{"bank_id": "config-bank"}')

    provider = plugin_module.ProspectaProvider()
    provider._hermes_home = tmp_path
    provider._config = provider._load_config()

    assert provider._resolve_bank_id() == "env-bank"


def test_embedded_fallback_requires_explicit_opt_in(plugin_module, tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PROSPECTA_ALLOW_EMBEDDED", raising=False)

    provider = plugin_module.ProspectaProvider()
    provider._hermes_home = tmp_path
    provider._config = {}

    with pytest.raises(RuntimeError, match="PROSPECTA_DATABASE_URL"):
        provider._resolve_database_url()


def test_embedded_fallback_opt_in_calls_embedded_starter(plugin_module, tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PROSPECTA_ALLOW_EMBEDDED", "1")

    provider = plugin_module.ProspectaProvider()
    provider._hermes_home = tmp_path
    provider._config = {}
    monkeypatch.setattr(provider, "_start_embedded_substrate", lambda: "postgresql://embedded/db")

    assert provider._resolve_database_url() == "postgresql://embedded/db"
