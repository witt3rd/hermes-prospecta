#!/usr/bin/env python3
"""Measure how the decision model answers the contract's ``time_scope`` question on a panel of invented lines.

    python tools/time_scope_panel.py CONTRACT [CONTRACT_AFTER] [--panel tests/data/time_scope_panel.json]
                                     [--runs 3] [--today YYYY-MM-DD] [--timeout 30]

Needs only a contract path and the environment (``OPENROUTER_API_KEY``; ``PROSPECTA_DECISION_MODEL`` picks
the model, as it does for the plugin). It calls the REAL decision transport the plugin uses
(``prefetch_contract.default_transport``), the same request the plugin builds, ``--runs`` times per line, and
prints per line the answers and per class the agreement with the panel's expectation. With two contracts it
prints both (before, after) side by side. Compare the "before" text with
``git show origin/main:contract/recall-contract-v1.md > /tmp/before.md``.

Nothing here prints, logs or stores the key; error text is scrubbed of it before it is shown. The cost is
``len(lines) * runs`` decision calls per contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prefetch_contract as pc  # noqa: E402

DEFAULT_PANEL = ROOT / "tests" / "data" / "time_scope_panel.json"
ERROR = "error"


def load_panel(path: Any = DEFAULT_PANEL) -> List[Dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lines = data["lines"]
    for row in lines:
        if row["expected"] not in pc.TIME_SCOPES:
            raise ValueError(f"panel line {row.get('id')!r}: expected {row['expected']!r} is not a time scope")
    return lines


def scrub(text: str, secret: Optional[str]) -> str:
    """``text`` without the key (never let it reach the output)."""
    text = str(text)
    if secret:
        text = text.replace(secret, "[key]")
    return text[:200]


def ask_scope(contract: pc.Contract, transport: Callable[[dict, float], dict], line: str, *, model: str,
              today: date, timeout: float, secret: Optional[str] = None) -> str:
    """The time scope the model answers for ``line`` (the plugin's own request and reading), or ``error``."""
    state = pc.build_state(line=line, agent_last="", turns=0, today=today.isoformat())
    req = pc.decision_request(contract, model, state)
    try:
        resp = transport(req, timeout)
        answers = resp.get("answers") if isinstance(resp, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("response has no answers")
        return pc.read_time_scope(contract, answers)
    except Exception as exc:  # noqa: BLE001 - one failed call is one "error", the panel goes on
        print(f"  ! {type(exc).__name__}: {scrub(exc, secret)}", file=sys.stderr)
        return ERROR


def run_panel(contract: pc.Contract, panel: List[Dict[str, str]], transport: Callable[[dict, float], dict], *,
              runs: int = 3, model: str, today: date, timeout: float = 30.0,
              secret: Optional[str] = None) -> List[Dict[str, Any]]:
    """One result per panel line: the line's row plus ``answers`` (a list of ``runs`` scopes)."""
    out = []
    for row in panel:
        answers = [ask_scope(contract, transport, row["line"], model=model, today=today, timeout=timeout,
                             secret=secret) for _ in range(runs)]
        out.append({**row, "answers": answers})
    return out


def agreement(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Per class: ``answers`` (all), ``exact`` (the expected scope), ``limited`` (any window, not none) and
    ``errors`` (no answer; counted as disagreement)."""
    classes: Dict[str, Dict[str, int]] = {}
    for r in results:
        c = classes.setdefault(r["class"], {"answers": 0, "exact": 0, "limited": 0, "errors": 0})
        for a in r["answers"]:
            c["answers"] += 1
            c["exact"] += a == r["expected"]
            c["limited"] += a not in ("none", ERROR)
            c["errors"] += a == ERROR
    return classes


def _summary(a: List[str]) -> str:
    return " ".join(f"{k}x{n}" for k, n in sorted(Counter(a).items()))


def report(results_by_label: Dict[str, List[Dict[str, Any]]]) -> str:
    labels = list(results_by_label)
    first = results_by_label[labels[0]]
    rows = []
    for i, row in enumerate(first):
        cells = []
        for label in labels:
            r = results_by_label[label][i]
            ok = sum(a == r["expected"] for a in r["answers"])
            cells.append(f"{label}: {_summary(r['answers'])} ({ok}/{len(r['answers'])})")
        rows.append(f"[{row['class']}] {row['id']} (expect {row['expected']})\n    {row['line']}\n    "
                    + " | ".join(cells))
    out = ["PER LINE (answers; in brackets: agreeing / runs)", *rows, "", "PER CLASS (agreement with the expectation)"]
    classes = sorted({r["class"] for r in first})
    for cls in classes:
        for label in labels:
            c = agreement(results_by_label[label])[cls]
            n = c["answers"] or 1
            what = "exact" if cls == "limits_memory" else "none"
            extra = (f"; any window {c['limited']}/{c['answers']} ({100 * c['limited'] // n}%)"
                     if cls == "limits_memory" else f"; wrongly windowed {c['limited']}/{c['answers']}")
            out.append(f"  {cls:<17} {label:<7} {what} {c['exact']}/{c['answers']} ({100 * c['exact'] // n}%)"
                       f"{extra}; errors {c['errors']}")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("contract", help="the contract file to measure (the after, or the only one)")
    ap.add_argument("contract_after", nargs="?", help="a second contract: the first is then 'before'")
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--runs", type=int, default=3, help="calls per line (default 3)")
    ap.add_argument("--today", default="", help="the day the question counts back from (default: today)")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per decision call")
    args = ap.parse_args(argv)

    secret = os.environ.get("OPENROUTER_API_KEY") or None
    if not secret:
        print("OPENROUTER_API_KEY is not set in the environment", file=sys.stderr)
        return 2
    today = date.fromisoformat(args.today) if args.today else date.today()
    panel = load_panel(args.panel)
    model = pc.Settings().decision_model or pc.library_decision_model()
    transport = pc.default_transport()          # the key is read from the environment per call, never printed

    paths = [args.contract] + ([args.contract_after] if args.contract_after else [])
    labels = ["before", "after"] if args.contract_after else ["contract"]
    print(f"model {model}; today {today}; {len(panel)} lines x {args.runs} runs per contract", flush=True)
    results = {}
    for label, path in zip(labels, paths):
        print(f"-- {label}: {path}", flush=True)
        results[label] = run_panel(pc.parse_contract(Path(path).read_text(encoding="utf-8")), panel, transport,
                                   runs=args.runs, model=model, today=today, timeout=args.timeout, secret=secret)
    print()
    print(report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
