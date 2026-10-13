"""The memory tool guide in the system prompt block."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PROSPECTA_TOOL_GUIDE", raising=False)


def _p(plugin_module, **cfg):
    p = plugin_module.ProspectaProvider()
    p._config = cfg
    return p


def test_guide_on_by_default(plugin_module):
    block = _p(plugin_module).system_prompt_block()
    for word in ("prospecta_search", "prospecta_recall", "depth", "date", "generated question", "automatic"):
        assert word in block
    assert plugin_module._TOOL_GUIDE in block


@pytest.mark.parametrize("off", ["false", "off", "0", "no", False])
def test_guide_off_by_config(plugin_module, off):
    p = _p(plugin_module, tool_guide=off)
    assert p.system_prompt_block() == p._base_prompt_block()
    assert "Memory tool guide" not in p.system_prompt_block()


def test_env_overrides_config(plugin_module, monkeypatch):
    monkeypatch.setenv("PROSPECTA_TOOL_GUIDE", "off")
    assert "Memory tool guide" not in _p(plugin_module, tool_guide="true").system_prompt_block()
    monkeypatch.setenv("PROSPECTA_TOOL_GUIDE", "on")
    assert "Memory tool guide" in _p(plugin_module, tool_guide="false").system_prompt_block()


def test_block_grows_only_by_the_guide(plugin_module):
    on = _p(plugin_module).system_prompt_block()
    base = _p(plugin_module, tool_guide="false").system_prompt_block()
    assert on.startswith(base)
    assert len(on) == len(base) + 2 + len(plugin_module._TOOL_GUIDE)
    assert len(plugin_module._TOOL_GUIDE.splitlines()) <= 12
    assert base.startswith("You have access to prospecta memory")
