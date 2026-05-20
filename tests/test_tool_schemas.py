"""Tool schema tests."""
from __future__ import annotations


def test_three_tools_registered(plugin_module):
    p = plugin_module.ProspectaProvider()
    schemas = p.get_tool_schemas()
    names = [s["name"] for s in schemas]
    assert names == ["prospecta_retain", "prospecta_recall", "prospecta_search"]


def test_retain_schema_marks_index_text_optional(plugin_module):
    p = plugin_module.ProspectaProvider()
    retain = next(s for s in p.get_tool_schemas() if s["name"] == "prospecta_retain")
    assert "index_text" not in retain["parameters"]["required"]
    assert set(retain["parameters"]["required"]) == {"content", "source"}


def test_recall_schema_minimal(plugin_module):
    p = plugin_module.ProspectaProvider()
    recall = next(s for s in p.get_tool_schemas() if s["name"] == "prospecta_recall")
    assert recall["parameters"]["required"] == ["query"]
    assert "limit" in recall["parameters"]["properties"]


def test_search_schema_modes(plugin_module):
    p = plugin_module.ProspectaProvider()
    search = next(s for s in p.get_tool_schemas() if s["name"] == "prospecta_search")
    modes = search["parameters"]["properties"]["mode"]["enum"]
    assert set(modes) == {"hybrid", "semantic", "lexical"}
