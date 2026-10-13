"""The connection watch warns loudly, in time, when the database is close to
its connection limit (no database needed: psycopg.connect is faked)."""
from __future__ import annotations

import json
import logging
import threading
import time

import pytest

URL = "postgresql://fake/db"
HEALTH = "memory-provider-health.json"


class Server:
    """A fake Postgres: how many connections are in use, and the limit."""

    def __init__(self):
        self.used, self.limit, self.fail = 10, 100, None
        self.connects = []


class _Conn:
    def __init__(self, server):
        self.server, self.closed, self._row = server, False, None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def execute(self, sql, *a):
        low = sql.lower()
        if "pg_stat_activity" in low:
            self._row = (self.server.used,)
        elif "max_connections" in low:
            self._row = (str(self.server.limit),)
        return self

    def cursor(self):
        return self

    def fetchone(self):
        return self._row

    def close(self):
        self.closed = True


@pytest.fixture
def server(monkeypatch):
    s = Server()
    import psycopg

    def connect(url, **kw):
        if s.fail:
            raise s.fail
        c = _Conn(s)
        s.connects.append(c)
        return c

    monkeypatch.setattr(psycopg, "connect", connect)
    return s


@pytest.fixture
def watch(plugin_module, tmp_path):
    return plugin_module._ConnectionWatch(URL, tmp_path, percent=70, interval=120)


def _health(tmp_path):
    p = tmp_path / HEALTH
    return json.loads(p.read_text()) if p.exists() else None


def test_below_the_line_is_silent(watch, server, tmp_path, caplog):
    server.used = 69  # 69%
    with caplog.at_level(logging.DEBUG):
        watch.check()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert _health(tmp_path) is None
    assert server.connects and all(c.closed for c in server.connects)  # own short-lived connection


def test_above_the_line_logs_the_numbers_and_writes_the_notice(watch, server, tmp_path, caplog):
    server.used = 70
    with caplog.at_level(logging.ERROR):
        watch.check()
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "70" in errors[0] and "100" in errors[0] and "connection" in errors[0]
    h = _health(tmp_path)
    assert h["status"] == "unavailable" and h["provider"] == "prospecta"
    assert "70" in h["reason"] and "100" in h["reason"]
    assert h["since"] and h["last_checked"]


def test_back_below_the_line_clears_the_notice(watch, server, tmp_path):
    server.used = 95
    watch.check()
    since = _health(tmp_path)["since"]
    watch.check()
    assert _health(tmp_path)["since"] == since  # same episode
    server.used = 20
    watch.check()
    assert _health(tmp_path) is None


def test_clearing_leaves_another_cause_alone(watch, server, tmp_path):
    (tmp_path / HEALTH).write_text(json.dumps(
        {"status": "unavailable", "provider": "prospecta", "kind": "init_failed",
         "reason": "initialize() failed", "since": "t"}))
    watch.check()  # healthy numbers
    assert _health(tmp_path)["kind"] == "init_failed"


def test_a_failing_query_takes_the_loud_path(watch, server, tmp_path, caplog):
    server.fail = RuntimeError("FATAL: sorry, too many clients already")
    with caplog.at_level(logging.ERROR):
        watch.check()  # never raises
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors and "too many clients" in errors[0]
    h = _health(tmp_path)
    assert h["status"] == "unavailable" and "too many clients" in h["reason"]
    server.fail = None
    watch.check()
    assert _health(tmp_path) is None


def test_percent_is_the_line(plugin_module, server, tmp_path):
    w = plugin_module._ConnectionWatch(URL, tmp_path, percent=90, interval=120)
    server.used = 89
    w.check()
    assert _health(tmp_path) is None
    server.used = 90
    w.check()
    assert _health(tmp_path) is not None


def test_thread_checks_at_start_then_repeats_and_stops_at_shutdown(plugin_module, server, tmp_path):
    server.used = 99
    w = plugin_module._ConnectionWatch(URL, tmp_path, percent=70, interval=0.02)
    w.start()
    deadline = time.monotonic() + 5
    while len(server.connects) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(server.connects) >= 3
    assert _health(tmp_path) is not None
    t = w._thread
    assert t.daemon
    w.stop()
    assert not t.is_alive()
    n = len(server.connects)
    time.sleep(0.1)
    assert len(server.connects) == n


def test_start_does_not_block_on_a_slow_database(plugin_module, tmp_path, monkeypatch):
    import psycopg
    gate = threading.Event()
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: gate.wait(5))
    w = plugin_module._ConnectionWatch(URL, tmp_path, percent=70, interval=120)
    t0 = time.monotonic()
    w.start()
    assert time.monotonic() - t0 < 1
    gate.set()
    w.stop()


def test_settings(plugin_module, monkeypatch):
    monkeypatch.delenv("PROSPECTA_CONNECTION_WATCH", raising=False)
    p = plugin_module.ProspectaProvider()
    assert p._connection_watch_enabled() is True
    for raw, want in (("false", False), (False, False), ("off", False), ("true", True), ("", True)):
        p._config = {"connection_watch": raw}
        assert p._connection_watch_enabled() is want
    assert p._resolve_connection_watch_percent() == 70
    for raw, want in (("85", 85), (50, 50), ("0", 70), ("101", 70), ("junk", 70), (-1, 70)):
        p._config = {"connection_watch_percent": raw}
        assert p._resolve_connection_watch_percent() == want


def test_provider_starts_and_stops_the_watch(plugin_module, server, tmp_path):
    p = plugin_module.ProspectaProvider()
    p._hermes_home, p._database_url, p._config = tmp_path, URL, {}
    p._start_connection_watch()
    t = p._conn_watch._thread
    assert t.is_alive()
    p.shutdown()
    assert not t.is_alive() and p._conn_watch is None


def test_provider_with_the_watch_off_starts_no_thread(plugin_module, server, tmp_path):
    p = plugin_module.ProspectaProvider()
    p._hermes_home, p._database_url, p._config = tmp_path, URL, {"connection_watch": "false"}
    p._start_connection_watch()
    assert p._conn_watch is None
