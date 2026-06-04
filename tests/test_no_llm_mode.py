"""Prospecta can initialize for retain/search even when Hermes has no ctx.llm."""
from __future__ import annotations

import json

from tests.conftest import stub_embed


class _NoLlmCtx:
    pass


def test_initialize_without_ctx_llm_supports_retain_with_index_text_and_search(
    plugin_module, pg_url, hermes_home, monkeypatch
):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", "no_llm")
    provider = plugin_module.ProspectaProvider()
    provider._ctx = _NoLlmCtx()
    monkeypatch.setattr(provider, "_build_embedder", lambda: stub_embed)
    (hermes_home / "prospecta.json").write_text(
        '{"embedding_dim": "32"}', encoding="utf-8"
    )

    provider.initialize(session_id="no-llm", hermes_home=str(hermes_home))
    try:
        retain = json.loads(provider.handle_tool_call(
            "prospecta_retain",
            {
                "content": "Augur Prospecta live smoke: host-wide Roger PostgreSQL is active.",
                "source": "smoke:no-llm",
                "index_text": "What is Augur using as his Prospecta substrate?",
            },
        ))
        assert retain["source"] == "smoke:no-llm"

        search = json.loads(provider.handle_tool_call(
            "prospecta_search",
            {"query": "Augur Prospecta substrate", "limit": 5},
        ))
        assert search["results"]
        assert search["results"][0]["source"] == "smoke:no-llm"
    finally:
        provider.shutdown()


def test_recall_reports_llm_unavailable_when_ctx_llm_missing(
    plugin_module, pg_url, hermes_home, monkeypatch
):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", "no_llm_recall")
    provider = plugin_module.ProspectaProvider()
    provider._ctx = _NoLlmCtx()
    monkeypatch.setattr(provider, "_build_embedder", lambda: stub_embed)
    (hermes_home / "prospecta.json").write_text(
        '{"embedding_dim": "32"}', encoding="utf-8"
    )

    provider.initialize(session_id="no-llm", hermes_home=str(hermes_home))
    try:
        result = json.loads(provider.handle_tool_call(
            "prospecta_recall", {"query": "anything"}
        ))
        assert "ctx.llm" in result["error"]
    finally:
        provider.shutdown()
