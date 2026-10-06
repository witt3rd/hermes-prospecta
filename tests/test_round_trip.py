"""End-to-end retain -> search/recall round trips against real Postgres."""
from __future__ import annotations

import json

from tests.conftest import stub_embed, stub_llm


def _call(p, tool, **args):
    return json.loads(p.handle_tool_call(tool, args))


def test_retain_then_search_finds_content(provider_with_stubs):
    p = provider_with_stubs
    _call(p, "prospecta_retain", content="Kelly's birthday is May 14.",
          source="fact:kelly", index_text=["When is Kelly's birthday?"])
    _call(p, "prospecta_retain", content="The capital of France is Paris.",
          source="fact:france", index_text=["What is the capital of France?"])
    res = _call(p, "prospecta_search", query="When is Kelly's birthday?", limit=5)
    assert res["results"], res
    assert res["results"][0]["source"] == "fact:kelly"
    assert "May 14" in res["results"][0]["content"]


def test_retain_then_recall_attributes_source(provider_with_stubs):
    p = provider_with_stubs
    _call(p, "prospecta_retain", content="The capital of France is Paris.",
          source="fact:france", index_text=["What is the capital of France?"])
    res = _call(p, "prospecta_recall", query="What is the capital of France?")
    assert res["synthesis"] == "Synthesized answer based on memory."
    assert "fact:france" in [s["source"] for s in res["sources"]]


def test_memory_survives_provider_restart(plugin_module, provider_with_stubs,
                                          hermes_home, monkeypatch):
    p = provider_with_stubs
    _call(p, "prospecta_retain", content="Persistent fact: the vault code is 4711.",
          source="fact:vault", index_text=["What is the vault code?"])
    p.shutdown()

    p2 = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p2, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p2, "_build_llm", lambda: stub_llm)
    p2.initialize(session_id="second", hermes_home=str(hermes_home))
    try:
        res = _call(p2, "prospecta_search", query="What is the vault code?")
        assert [r["source"] for r in res["results"]][:1] == ["fact:vault"]
    finally:
        p2.shutdown()


def test_banks_are_isolated(plugin_module, provider_with_stubs, hermes_home,
                            monkeypatch):
    p = provider_with_stubs
    _call(p, "prospecta_retain", content="Secret of bank test.",
          source="fact:secret", index_text=["secret"])
    monkeypatch.setenv("PROSPECTA_BANK_ID", "other")
    p2 = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p2, "_build_embedder", lambda: stub_embed)
    monkeypatch.setattr(p2, "_build_llm", lambda: stub_llm)
    p2.initialize(session_id="other", hermes_home=str(hermes_home))
    try:
        assert _call(p2, "prospecta_search", query="secret")["results"] == []
    finally:
        p2.shutdown()
