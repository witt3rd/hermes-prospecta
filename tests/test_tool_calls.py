"""Tool-call dispatch tests."""
from __future__ import annotations

import json


def test_handle_retain_returns_document_id(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call(
        "prospecta_retain",
        {
            "content": "Kelly's birthday is May 14.",
            "source": "fact:kelly-birthday",
            "tags": ["personal"],
        },
    )
    data = json.loads(result)
    assert "document_id" in data
    assert data["source"] == "fact:kelly-birthday"


def test_handle_recall_returns_the_grounded_answer_and_its_cited_sources(provider_with_stubs):
    # Populate one document directly.
    provider_with_stubs._memory.retain(
        content="The capital of France is Paris.",
        source="fact:france",
    )
    result = provider_with_stubs.handle_tool_call(
        "prospecta_recall",
        {"query": "What is the capital of France?", "limit": 5},
    )
    data = json.loads(result)
    assert data["answer"] == "Synthesized answer based on memory."      # the library's ask(): the stub model's reply
    assert "synthesis" in data and isinstance(data["citations"], list)
    assert [s["source"] for s in data["sources"]] == ["fact:france"]


def test_handle_search_returns_chunks(provider_with_stubs):
    provider_with_stubs._retain("The capital of France is Paris.", source="fact:france")   # as the tool writes it
    result = provider_with_stubs.handle_tool_call(
        "prospecta_search",
        {"query": "France", "limit": 5},
    )
    data = json.loads(result)
    assert data["count"] == len(data["results"]) >= 1
    assert any("France" in item["text"] for item in data["results"])     # the whole chunk
    for item in data["results"]:                                         # every hit is cited
        assert {"source", "created_on", "kind", "match_kind", "relevance_score", "exact_snippet"} <= set(item)


def test_unknown_tool_returns_error(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call("prospecta_unknown", {})
    data = json.loads(result)
    assert "error" in data


def test_retain_missing_required_returns_error(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call("prospecta_retain", {"content": "x"})
    data = json.loads(result)
    assert "error" in data
