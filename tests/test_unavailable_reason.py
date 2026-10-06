"""An unavailable provider must say WHY and HOW TO FIX it (the caller logs it at
ERROR and shows it to the being): 2026-10 Augur ran ~7 weeks without memory
because `prospecta` was missing from a rebuilt venv and the only trace was a
bare 'reports unavailable' warning."""
from __future__ import annotations

import builtins

import pytest


def _no_prospecta_import(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *a, **kw):
        if name == "prospecta" or name.startswith("prospecta."):
            raise ImportError("No module named 'prospecta'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)


def test_reason_names_missing_package_and_the_fix(plugin_module, monkeypatch):
    provider = plugin_module.ProspectaProvider()
    _no_prospecta_import(monkeypatch)
    assert provider.is_available() is False
    reason = provider.unavailable_reason()
    assert "prospecta" in reason and "not importable" in reason
    assert "ensure-venv.sh" in reason


def test_reason_names_missing_database_url(plugin_module, monkeypatch, tmp_path):
    for var in ("PROSPECTA_DATABASE_URL", "DATABASE_URL", "PROSPECTA_ALLOW_EMBEDDED"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = plugin_module.ProspectaProvider()
    assert provider.is_available() is False
    assert "PROSPECTA_DATABASE_URL" in provider.unavailable_reason()


def test_reason_empty_when_available(plugin_module, monkeypatch):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", "postgresql://u:p@localhost:1/x")
    provider = plugin_module.ProspectaProvider()
    assert provider.is_available() is True
    assert provider.unavailable_reason() == ""


def test_unset_llm_model_is_a_loud_error_not_a_library_default(plugin_module, monkeypatch, tmp_path):
    # prospecta.defaults would silently fall back to openai/gpt-4o-mini.
    monkeypatch.delenv("PROSPECTA_LLM_MODEL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = plugin_module.ProspectaProvider()
    provider._config = {}
    with pytest.raises(RuntimeError, match="PROSPECTA_LLM_MODEL"):
        provider._build_llm()


def test_configured_llm_model_still_builds(plugin_module, monkeypatch, tmp_path):
    pytest.importorskip("litellm")
    monkeypatch.setenv("PROSPECTA_LLM_MODEL", "openrouter/anthropic/claude-sonnet-5.5")
    provider = plugin_module.ProspectaProvider()
    provider._config = {}
    assert callable(provider._build_llm())
