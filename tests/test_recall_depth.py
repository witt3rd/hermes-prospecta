"""Depth is a thin pass-through to Prospecta (the rerank pool cut: 'standard' | 'deep'): omitted stays omitted.
``prospecta_recall`` hands it to ``ask()``, ``prospecta_search`` to ``search()``."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tests.conftest import check_call


class FakeMemory:
    def __init__(self):
        self.calls = []

    def ask(self, q, **kw):
        check_call("ask", q, **kw)
        self.calls.append(("ask", q, kw))
        return SimpleNamespace(answer="a", rendered="a", citations=[], items=[], sufficient=None, retried=False,
                               gate=None)

    def search(self, q, **kw):
        check_call("search", q, **kw)
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
    assert [name for name, _, _ in p._memory.calls] == ["ask", "search"]
    assert all("depth" not in kw for _, _, kw in p._memory.calls)


def test_depth_deep_reaches_recall_and_search(plugin_module):
    p = _provider(plugin_module)
    p.handle_tool_call("prospecta_recall", {"query": "x", "depth": "deep"})
    p.handle_tool_call("prospecta_search", {"query": "x", "depth": "deep"})
    assert [(name, kw["depth"]) for name, _, kw in p._memory.calls] == [("ask", "deep"), ("search", "deep")]


def test_schemas_offer_depth(plugin_module):
    names = {s["name"]: s for s in plugin_module.ProspectaProvider().get_tool_schemas()}
    assert names["prospecta_recall"]["parameters"]["properties"]["depth"]["enum"] == ["standard", "deep"]
    assert names["prospecta_search"]["parameters"]["properties"]["depth"]["enum"] == ["standard", "deep"]


def test_the_depth_words_describe_the_rerank_pool_not_a_synthesis_cost(plugin_module):
    """'deep' widens the pool the cross-encoder reads; it is not recall_synth's old '3x the cost' depth."""
    for schema in plugin_module.ProspectaProvider().get_tool_schemas():
        depth = schema["parameters"]["properties"].get("depth")
        if depth:
            assert "3x" not in depth["description"] and "pool" in depth["description"]
    assert "3x" not in plugin_module._TOOL_GUIDE
