"""Availability checks before provider initialize()."""
from __future__ import annotations

import shutil


def test_is_available_with_profile_config_database_url(plugin_module, tmp_path, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PROSPECTA_ALLOW_EMBEDDED", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "prospecta.json").write_text(
        '{"database_url": "postgresql://config/db"}', encoding="utf-8"
    )

    provider = plugin_module.ProspectaProvider()

    assert provider.is_available() is True


def test_is_available_does_not_use_docker_unless_embedded_opted_in(plugin_module, monkeypatch):
    monkeypatch.delenv("PROSPECTA_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("PROSPECTA_ALLOW_EMBEDDED", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/docker")

    provider = plugin_module.ProspectaProvider()

    assert provider.is_available() is False
