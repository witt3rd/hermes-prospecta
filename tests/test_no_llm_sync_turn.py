"""Prospecta sync_turn should not drop turns when no LLM is configured."""
from __future__ import annotations

import time

from tests.conftest import stub_embed


def test_sync_turn_without_llm_uses_turn_text_as_index_text(
    plugin_module, pg_url, hermes_home, monkeypatch
):
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", "sync_no_llm")
    provider = plugin_module.ProspectaProvider()
    monkeypatch.setattr(provider, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(provider, "_build_llm", lambda: None)
    (hermes_home / "prospecta.json").write_text(
        '{"embedding_dim": "32"}', encoding="utf-8"
    )

    provider.initialize(session_id="sync-no-llm", hermes_home=str(hermes_home))
    try:
        provider.sync_turn(
            "Donald: Are you live on Prospecta?",
            "Augur: Yes, I am live on Prospecta.",
            session_id="sync-no-llm",
        )
        assert provider._sync_thread is not None
        provider._sync_thread.join(timeout=5.0)
        assert not provider._sync_thread.is_alive()

        results = provider._memory.search("live on Prospecta", limit=5)
        assert results
        assert any("live on Prospecta" in r.original_chunk for r in results)
    finally:
        provider.shutdown()
