"""The time-scope panel (tests/data/time_scope_panel.json) and its script (tools/time_scope_panel.py), offline: a
fake decision transport stands in for the real one. Invented text only."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest

import prefetch_contract as pc

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "tests" / "data" / "time_scope_panel.json"

_spec = importlib.util.spec_from_file_location("time_scope_panel", ROOT / "tools" / "time_scope_panel.py")
panel_tool = importlib.util.module_from_spec(_spec)
sys.modules["time_scope_panel"] = panel_tool
_spec.loader.exec_module(panel_tool)

TODAY = date(2026, 10, 17)


def answer(scope):
    return {"answers": {"time_scope": {"type": "choice", "choice": scope}}}


def test_the_panel_is_balanced_and_every_expectation_is_a_scope():
    rows = json.loads(PANEL.read_text())["lines"]
    assert len(rows) >= 24 and len({r["id"] for r in rows}) == len(rows)
    request = [r for r in rows if r["class"] == "modifies_request"]
    memory = [r for r in rows if r["class"] == "limits_memory"]
    assert len(request) == len(memory) == len(rows) // 2
    assert {r["expected"] for r in request} == {"none"}
    assert {r["expected"] for r in memory} <= set(pc.TIME_SCOPES) - {"none"}
    assert {"recent", "past_week", "past_month", "named"} <= {r["expected"] for r in memory}


def test_the_panel_has_the_evidence_shape_two_this_weeks_a_name_and_an_acronym():
    rows = {r["id"]: r["line"] for r in json.loads(PANEL.read_text())["lines"]}
    shaped = [t for k, t in rows.items() if k.startswith("evidence_shape")]
    assert shaped
    for t in shaped:
        assert len(re.findall(r"this week", t, re.IGNORECASE)) >= 2
        assert re.search(r"\b[A-Z]{3,}\b(?<!THIS)(?<!WEEK)", re.sub(r"THIS WEEK", "", t))   # an acronym
        assert re.search(r"the [A-Z][a-z]+ theme", t)                                       # a proper name


def test_the_panel_names_no_one_real():
    """Invented text only: nothing here may look like an address, a number or a link."""
    text = PANEL.read_text()
    assert not re.search(r"@|https?://|\d{4,}-\d{2}-\d{2}|\+\d", text.replace("2025", "").replace("2024", ""))


def test_the_script_asks_the_plugins_own_request_and_counts_agreement():
    contract = pc.load_contract()
    seen = []

    def transport(req, timeout):
        seen.append(req)
        # a model that answers a window for every line that says "last" and none otherwise
        return answer("past_week" if " last " in req["state"]["new_lines"][0] else "none")

    rows = panel_tool.load_panel()
    results = panel_tool.run_panel(contract, rows, transport, runs=3, model="m", today=TODAY)
    assert len(seen) == 3 * len(rows)
    assert seen[0]["questions"] is contract.questions and seen[0]["state"]["today"] == "2026-10-17"
    assert seen[0]["model"] == "m"
    c = panel_tool.agreement(results)
    assert c["modifies_request"]["answers"] == c["limits_memory"]["answers"] == 3 * (len(rows) // 2)
    assert c["modifies_request"]["errors"] == 0
    text = panel_tool.report({"before": results, "after": results})
    assert "PER CLASS" in text and "before" in text and "after" in text


def test_a_failed_call_is_an_error_answer_and_the_key_never_reaches_the_output(capsys):
    def transport(req, timeout):
        raise RuntimeError("401 for key sk-secret-123")

    rows = panel_tool.load_panel()[:2]
    results = panel_tool.run_panel(pc.load_contract(), rows, transport, runs=1, model="m", today=TODAY,
                                   secret="sk-secret-123")
    assert [r["answers"] for r in results] == [["error"], ["error"]]
    seen = capsys.readouterr()
    assert "sk-secret-123" not in seen.out + seen.err and "[key]" in seen.err


def test_main_needs_the_key_and_prints_nothing_of_it(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert panel_tool.main([str(pc.DEFAULT_CONTRACT_PATH)]) == 2
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-secret-456")
    monkeypatch.setattr(pc, "default_transport", lambda api_key=None: (lambda req, t: answer("none")))
    monkeypatch.setattr(pc, "library_decision_model", lambda: "fake-model")
    assert panel_tool.main([str(pc.DEFAULT_CONTRACT_PATH), str(pc.DEFAULT_CONTRACT_PATH), "--runs", "1",
                            "--today", "2026-10-17"]) == 0
    seen = capsys.readouterr()
    assert "sk-secret-456" not in seen.out + seen.err
    assert "before" in seen.out and "after" in seen.out and "fake-model" in seen.out
