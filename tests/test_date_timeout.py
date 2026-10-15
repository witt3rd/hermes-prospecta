"""Hard timeout on the date-extraction model call."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import date_cue_gate as g
import prospecta.channels.extract as ext

NOW = datetime(2026, 5, 14, 12, 0)
Q = "what happened in March 2025"
OK = '{"people": [], "date_from": "2024-01-01", "date_to": "2024-01-31", "hard": false}'


def _slow(seconds, ev=None):
    def llm(messages, json_mode=False, model=None):
        (ev or threading.Event()).wait(seconds)
        return OK
    return llm


def test_slow_call_times_out_with_regex_and_warning(caplog):
    gated = g.make_gated(ext.extract_filters, timeout=1)
    ev = threading.Event()
    with caplog.at_level(logging.WARNING, logger="date_cue_gate"):
        t = time.monotonic()
        got = gated(Q, llm=_slow(10, ev), people_vocab=[], now=NOW, model="m")
        dt = time.monotonic() - t
    ev.set()
    assert dt < 1.2
    assert got == ext.regex_filters(Q, [], NOW)
    warns = [r for r in caplog.records if "fell back to regex" in r.getMessage()]
    assert len(warns) == 1


def test_fast_call_returns_model_result():
    got = g.make_gated(ext.extract_filters, timeout=2)(
        Q, llm=_slow(0), people_vocab=[], now=NOW, model="m")
    assert got.date_from == "2024-01-01"


def test_error_falls_back_without_raising():
    def llm(messages, json_mode=False, model=None):
        raise RuntimeError("boom")
    got = g.make_gated(ext.extract_filters, timeout=2)(
        Q, llm=llm, people_vocab=[], now=NOW, model="m")
    assert got == ext.regex_filters(Q, [], NOW)


def test_abandoned_thread_exception_does_not_leak():
    ev = threading.Event()

    def llm(messages, json_mode=False, model=None):
        ev.wait(5)
        raise RuntimeError("late")
    got = g.make_gated(ext.extract_filters, timeout=0.1)(
        Q, llm=llm, people_vocab=[], now=NOW, model="m")
    assert got == ext.regex_filters(Q, [], NOW)
    ev.set()


def test_zero_disables_timeout():
    got = g.make_gated(ext.extract_filters, timeout=0)(
        Q, llm=_slow(0.5), people_vocab=[], now=NOW, model="m")
    assert got.date_from == "2024-01-01"


def test_module_setting_used_by_default():
    g.set_timeout(0.1)
    try:
        ev = threading.Event()
        got = g.make_gated(ext.extract_filters)(
            Q, llm=_slow(5, ev), people_vocab=[], now=NOW, model="m")
        ev.set()
        assert got == ext.regex_filters(Q, [], NOW)
    finally:
        g.set_timeout(g.DEFAULT_TIMEOUT)


def test_no_cue_no_call_and_no_thread():
    calls = []

    def llm(messages, json_mode=False, model=None):
        calls.append(1)
        return OK
    g.make_gated(ext.extract_filters, timeout=1)(
        "how does the retry loop work", llm=llm, people_vocab=[], now=NOW)
    assert calls == []


def test_abandoned_calls_bounded_under_50_concurrent():
    gated = g.make_gated(ext.extract_filters, timeout=0.2)
    ev = threading.Event()
    before = threading.active_count()
    results = []

    def one():
        results.append(gated(Q, llm=_slow(30, ev), people_vocab=[], now=NOW, model="m"))
    ts = [threading.Thread(target=one) for _ in range(50)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(5)
    extra = threading.active_count() - before
    ev.set()
    assert len(results) == 50
    assert all(r == ext.regex_filters(Q, [], NOW) for r in results)
    assert extra <= g.MAX_INFLIGHT
    time.sleep(0.3)


def test_provider_option(plugin_module, monkeypatch):
    p = plugin_module.ProspectaProvider()
    monkeypatch.delenv("PROSPECTA_DATE_TIMEOUT", raising=False)
    p._config = {}
    assert p._recall_date_timeout() == 6.0
    p._config = {"recall": {"date_timeout_seconds": "2.5"}}
    assert p._recall_date_timeout() == 2.5
    p._config = {"recall": {"date_timeout_seconds": "0"}}
    assert p._recall_date_timeout() == 0.0
    monkeypatch.setenv("PROSPECTA_DATE_TIMEOUT", "bad")
    assert p._recall_date_timeout() == 6.0
