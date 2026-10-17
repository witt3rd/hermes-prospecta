"""Depth is a thin pass-through to Prospecta: omitted stays omitted."""
from __future__ import annotations

import json
from types import SimpleNamespace


class FakeMemory:
    def __init__(self):
        self.calls = []

    def recall_synth(self, q, **kw):
        self.calls.append(("recall_synth", q, kw))
        return SimpleNamespace(synthesis="s", sources=[])

    def search(self, q, **kw):
        self.calls.append(("search", q, kw))
        return []


def _provider(plugin_module):
    p = plugin_module.ProspectaProvider()
    p._memory = FakeMemory()
    p._llm = object()
    return p


def test_depth_omitted_is_not_passed(plugin_module):
    p = _provider(plugin_module)
    p.handle_tool_call("prospecta_recall", {"query": "x"})
    p.handle_tool_call("prospecta_search", {"query": "x"})
    assert all("depth" not in kw for _, _, kw in p._memory.calls)


def test_depth_deep_reaches_recall_and_search(plugin_module):
    p = _provider(plugin_module)
    p.handle_tool_call("prospecta_recall", {"query": "x", "depth": "deep"})
    p.handle_tool_call("prospecta_search", {"query": "x", "depth": "deep"})
    assert [kw["depth"] for _, _, kw in p._memory.calls] == ["deep", "deep"]


def test_schemas_offer_depth(plugin_module):
    names = {s["name"]: s for s in plugin_module.ProspectaProvider().get_tool_schemas()}
    assert names["prospecta_recall"]["parameters"]["properties"]["depth"]["enum"] == ["standard", "deep"]
    assert names["prospecta_search"]["parameters"]["properties"]["depth"]["enum"] == ["standard", "deep"]
