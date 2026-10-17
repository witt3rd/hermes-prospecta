"""Tool schema tests."""
from __future__ import annotations


def _schemas(plugin_module):
    return {s["name"]: s for s in plugin_module.ProspectaProvider().get_tool_schemas()}


def test_three_tools_registered(plugin_module):
    p = plugin_module.ProspectaProvider()
    schemas = p.get_tool_schemas()
    names = [s["name"] for s in schemas]
    assert names == ["prospecta_retain", "prospecta_recall", "prospecta_search"]


def test_retain_schema_takes_content_source_and_tags_only(plugin_module):
    """The library's write path is chunk-only: there is no ``index_text`` to offer."""
    retain = _schemas(plugin_module)["prospecta_retain"]
    assert set(retain["parameters"]["required"]) == {"content", "source"}
    assert set(retain["parameters"]["properties"]) == {"content", "source", "tags"}
    assert "index_text" not in retain["description"] and "question" not in retain["description"].lower()


def test_recall_schema_minimal(plugin_module):
    recall = _schemas(plugin_module)["prospecta_recall"]
    assert recall["parameters"]["required"] == ["query"]
    assert set(recall["parameters"]["properties"]) == {"query", "limit", "depth"}


def test_recall_describes_the_grounded_cited_answer_of_ask(plugin_module):
    """Mode 2B: Stage 1 + Stage 2, then an answer from the notes alone with [n] citations, and the sources."""
    text = _schemas(plugin_module)["prospecta_recall"]["description"]
    for word in ("[n]", "citation", "exact_snippet", "not in memory", "search"):
        assert word in text
    assert "synthesi" not in text.lower() and "formulat" not in text.lower()
    assert "limit" in _schemas(plugin_module)["prospecta_recall"]["parameters"]["properties"]


def test_search_schema_has_no_mode(plugin_module):
    """``mode`` (hybrid / semantic / lexical) is gone from the library: one retrieval path."""
    props = _schemas(plugin_module)["prospecta_search"]["parameters"]["properties"]
    assert "mode" not in props and {"query", "limit", "depth", "date_from", "date_to"} == set(props)
