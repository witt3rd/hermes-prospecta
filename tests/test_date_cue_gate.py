"""Date cue gate: the extraction model runs only for questions with a date cue."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from prospecta.channels.base import Filters
import prospecta._index as idx
import prospecta.channels.extract as ext

import date_cue_gate as g

NOW = datetime(2026, 5, 14, 12, 0)
VOCAB = ["Alice Marlow", "Bob"]

DATE_PHRASES = [
    "what did we ship last week",
    "notes from last month about the router",
    "meetings in the past 3 days",
    "what happened over the previous two weeks",
    "plans for next month",
    "the deploy from 2026-10-03",
    "work between 2026-01-05 and 2026-02-10",
    "ideas from the summer",
    "winter project status",
    "Q3 planning decisions",
    "results in the second quarter",
    "what I did in 2024",
    "what was decided on Tuesday",
    "the call on friday morning",
    "a few days ago I wrote about caching",
    "two weeks ago we discussed pricing",
    "what happened yesterday",
    "tasks due tomorrow",
    "in March I fixed the proxy",
    "since January the build is slow",
    "entries from 10/03",
    "the meeting on the 3rd of June",
    "recently changed settings",
    "earlier this week we met",
]

NO_DATE = [
    "how does the retry loop back off",
    "which port does the proxy listen on, port 8443",
    "show me the top 10 slowest queries",
    "this may help with the cache invalidation",
    "we may need to rewrite the parser",
    "the march of progress in the build system",
    "why does the worker march through every shard",
    "what did I decide about backups",
    "explain the schema of the documents table",
    "list the 5 largest files",
    "how do I configure the nginx upstream timeout",
    "what is the difference between a mutex and a semaphore",
    "which database holds the embeddings",
    "Alice Marlow opinion on the retry design",
    "what does Bob think about the new queue",
    "error 404 from the gateway",
    "version 3.12 of the interpreter",
    "how many workers do we run, 16 or 32",
    "the quick brown fox jumps over a lazy dog",
    "summarize the design of the ranking stage",
    "bash exit code 127 means what",
    "what is the sum of 12 and 30",
]


def _llm(counter):
    def llm(messages, json_mode=False, model=None):
        counter.append(model)
        return json.dumps({"people": [], "date_from": None, "date_to": None, "hard": False})
    return llm


def _gated():
    return g.make_gated(ext.extract_filters)


@pytest.mark.parametrize("text", DATE_PHRASES)
def test_date_phrase_calls_model_once(text):
    calls = []
    _gated()(text, llm=_llm(calls), people_vocab=VOCAB, now=NOW, model="m")
    assert len(calls) == 1


@pytest.mark.parametrize("text", NO_DATE)
def test_no_date_zero_calls_and_library_result(text):
    calls = []
    got = _gated()(text, llm=_llm(calls), people_vocab=[], now=NOW, model="m")
    assert calls == []
    assert got == ext.extract_filters(text, llm=_llm([]), people_vocab=[], now=NOW, model="m")
    assert got == Filters(people=[], date_from=None, date_to=None, hard=False)


def test_people_still_extracted_without_cue():
    calls = []
    got = _gated()("what does alice marlow think of the queue", llm=_llm(calls),
                   people_vocab=VOCAB, now=NOW)
    assert calls == []
    assert got.people == ["Alice Marlow"]
    assert got.date_from is None and got.hard is False


def test_no_llm_unchanged():
    got = _gated()("alice marlow in March 2025", llm=None, people_vocab=VOCAB, now=NOW)
    assert got.date_from == "2025-03-01" and got.people == ["Alice Marlow"]


def test_install_wraps_once_and_restores():
    orig = idx.extract_filters
    assert g.install() and g.install()
    try:
        assert idx.extract_filters._date_cue_gate
        assert idx.extract_filters.__wrapped__ is orig  # not double-wrapped
    finally:
        g.uninstall()
        assert idx.extract_filters._date_cue_gate  # one hold left
        g.uninstall()
    assert idx.extract_filters is orig


def test_provider_option(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    p._config = {}
    monkeypatch.delenv("PROSPECTA_DATE_CUE_GATE", raising=False)
    assert p._recall_date_cue_gate() is True
    p._config = {"recall": {"date_cue_gate": "false"}}
    assert p._recall_date_cue_gate() is False
    p._config = {}
    monkeypatch.setenv("PROSPECTA_DATE_CUE_GATE", "off")
    assert p._recall_date_cue_gate() is False


def test_gate_off_restores_model_call():
    calls = []
    ext.extract_filters("how does the retry loop back off", llm=_llm(calls),
                        people_vocab=[], now=NOW, model="m")
    assert len(calls) == 1
    assert not getattr(idx.extract_filters, "_date_cue_gate", False)
