"""Provider lifecycle tests."""
from __future__ import annotations

import os
import shutil
import pytest


def test_is_available_no_db_no_docker(plugin_module, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Hide docker by overriding shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None)
    provider = plugin_module.ProspectaProvider()
    # prospecta is importable in test env, so this hinges on docker absence
    assert provider.is_available() is False


def test_is_available_with_database_url(plugin_module, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://example/db")
    provider = plugin_module.ProspectaProvider()
    assert provider.is_available() is True


def test_initialize_byo_creates_default_bank(provider_with_stubs):
    p = provider_with_stubs
    assert p._memory is not None
    stats = p._memory.bank_stats("test")
    assert stats.bank_id == "test"


def test_shutdown_idempotent(provider_with_stubs):
    p = provider_with_stubs
    p.shutdown()
    # Second shutdown must not raise
    p.shutdown()
