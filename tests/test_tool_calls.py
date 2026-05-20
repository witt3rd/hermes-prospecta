"""Tool-call dispatch tests."""
from __future__ import annotations

import json


def test_handle_retain_returns_document_id(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call(
        "prospecta_retain",
        {
            "content": "Kelly's birthday is May 14.",
            "source": "fact:kelly-birthday",
            "index_text": ["When is Kelly's birthday?"],  # bypass LLM
            "tags": ["personal"],
        },
    )
    data = json.loads(result)
    assert "document_id" in data
    assert data["source"] == "fact:kelly-birthday"


def test_handle_recall_returns_synthesis(provider_with_stubs):
    # Populate one document directly.
    provider_with_stubs._memory.retain(
        content="The capital of France is Paris.",
        source="fact:france",
        index_text=["What is the capital of France?"],
    )
    result = provider_with_stubs.handle_tool_call(
        "prospecta_recall",
        {"query": "What is the capital of France?", "limit": 5},
    )
    data = json.loads(result)
    assert "synthesis" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_handle_search_returns_chunks(provider_with_stubs):
    provider_with_stubs._memory.retain(
        content="The capital of France is Paris.",
        source="fact:france",
        index_text=["What is the capital of France?"],
    )
    result = provider_with_stubs.handle_tool_call(
        "prospecta_search",
        {"query": "France", "mode": "hybrid", "limit": 5},
    )
    data = json.loads(result)
    assert "results" in data
    for item in data["results"]:
        assert "content" in item
        assert "source" in item
        assert "scores" in item


def test_unknown_tool_returns_error(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call("prospecta_unknown", {})
    data = json.loads(result)
    assert "error" in data


def test_retain_missing_required_returns_error(provider_with_stubs):
    result = provider_with_stubs.handle_tool_call("prospecta_retain", {"content": "x"})
    data = json.loads(result)
    assert "error" in data
