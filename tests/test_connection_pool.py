"""The plugin keeps a bounded number of Postgres connections per Memory and
closes every one it opens (no database needed: psycopg.connect is faked)."""
from __future__ import annotations

import threading
import time

import pytest

from tests.conftest import stub_embed

URL = "postgresql://fake/db"


class FakeConn:
    def __init__(self, world):
        self.world = world
        self.closed = False
        self.broken = False
        self.dead = False  # the server dropped it, not yet noticed
        self.commits = self.rollbacks = 0
        self.statuses = []
        world.all.append(self)

    class _Info:
        transaction_status = 0  # IDLE

    info = _Info()

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self):
                if conn.dead:
                    raise RuntimeError("server closed the connection")
                return self

            def __exit__(self, *a):
                return False

        return Cur()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class World:
    def __init__(self):
        self.all = []

    @property
    def open(self):
        return [c for c in self.all if not c.closed]


@pytest.fixture
def world(monkeypatch):
    w = World()
    import psycopg
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: FakeConn(w))
    return w


def _in_threads(n, fn):
    """Run fn once on each of n short-lived threads, one after another."""
    for _ in range(n):
        t = threading.Thread(target=fn)
        t.start()
        t.join()


def _use(memory):
    def run():
        with memory._pool.cursor():
            pass
        with memory._pool.connection():
            pass
        sink = getattr(memory._tracer, "_connection", None)
        if sink:  # the default tracer sink shares the pool
            with sink():
                pass
    return run


def _memories(plugin_module, **kw):
    from prospecta._tracer import NoOpTracer
    build = plugin_module._build_memory
    main = build(database_url=URL, embed=stub_embed, bank_id="main", **kw)
    shadow = build(database_url=URL, embed=stub_embed, bank_id="shadow", tracer=NoOpTracer(), **kw)
    dual = build(database_url=URL, embed=stub_embed, bank_id="dual", **kw)
    return main, shadow, dual


def test_short_lived_threads_do_not_pile_up_connections(plugin_module, world):
    mems = _memories(plugin_module)
    for m in mems:
        _in_threads(100, _use(m))
    assert 0 < len(world.open) <= 3 * 4
    assert len(world.all) <= 3 * 4  # and none were opened just to be thrown away


def test_concurrent_threads_stay_within_the_limit(plugin_module, world):
    (m, *_), n = _memories(plugin_module, max_connections=2), 40
    threads = [threading.Thread(target=_use(m)) for _ in range(n)]
    [t.start() for t in threads]
    [t.join(timeout=10) for t in threads]
    assert not any(t.is_alive() for t in threads)
    assert len(world.all) <= 2


def test_nested_use_on_one_thread_shares_the_connection(plugin_module, world):
    (m, *_) = _memories(plugin_module, max_connections=1)
    with m._pool.connection() as outer:
        with m._pool.cursor():
            pass
        with m._pool.connection() as inner:
            assert inner is outer
    assert len(world.all) == 1


def test_lost_connection_is_replaced_on_next_use(plugin_module, world):
    (m, *_) = _memories(plugin_module, max_connections=1)
    with m._pool.connection():
        pass
    first = world.all[0]
    first.broken = True  # the server went away while it sat idle and psycopg noticed
    with m._pool.cursor():
        pass
    assert len(world.all) == 2 and first.closed and len(world.open) == 1


def test_connection_that_dies_mid_use_is_not_fatal(plugin_module, world):
    (m, *_) = _memories(plugin_module, max_connections=1)
    with m._pool.connection():
        pass
    world.all[0].dead = True  # not noticed yet: this use fails
    with pytest.raises(RuntimeError):
        with m._pool.cursor():
            pass
    world.all[0].closed = True  # psycopg marks it closed after the failure
    with m._pool.cursor():  # the next use gets a fresh connection
        pass
    assert len(world.open) == 1 and world.open[0] is not world.all[0]


def test_dedicated_connection_is_private_and_closed(plugin_module, world):
    (m, *_) = _memories(plugin_module)
    with m._pool.dedicated() as conn:
        assert conn.world is world
    assert conn.closed and not world.open


def test_unfinished_transaction_is_rolled_back_before_reuse(plugin_module, world):
    (m, *_) = _memories(plugin_module, max_connections=1)
    FakeConn.info = type("I", (), {"transaction_status": 2})()  # INTRANS
    try:
        with m._pool.connection():
            pass
    finally:
        FakeConn.info = FakeConn._Info()
    assert world.all[0].rollbacks == 1


def test_shutdown_closes_all_three_memories(plugin_module, world, tmp_path):
    main, shadow, dual = _memories(plugin_module)
    for m in (main, shadow, dual):
        _use(m)()
    p = plugin_module.ProspectaProvider()
    p._memory = main
    p._shadow = plugin_module._ShadowReader(main, shadow, bank_id="main", shadow_bank_id="shadow")
    p._dual = plugin_module._DualWriter(dual, bank_id="dual", retry_path=tmp_path / "retry.jsonl")
    assert world.open
    p.shutdown()
    assert not world.open


def test_max_connections_setting(plugin_module):
    p = plugin_module.ProspectaProvider()
    assert p._resolve_max_connections() == 4
    for raw, want in (("7", 7), (3, 3), ("0", 4), ("junk", 4), (-2, 4)):
        p._config = {"max_connections": raw}
        assert p._resolve_max_connections() == want


# ---- idle timeout --------------------------------------------------------

class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _idle_pool(plugin_module, max_connections=2, idle=300):
    clock = Clock()
    return plugin_module._BoundedPool(URL, max_connections, idle_seconds=idle, clock=clock), clock


def test_idle_connection_is_closed_and_count_returns_to_zero(plugin_module, world):
    pool, clock = _idle_pool(plugin_module)
    with pool.cursor():
        pass
    clock.now += 299
    pool.reap()
    assert len(world.open) == 1  # not idle long enough yet
    clock.now += 2
    pool.reap()
    assert not world.open and world.all[0].closed


def test_busy_connection_is_not_closed_by_the_idle_timeout(plugin_module, world):
    pool, clock = _idle_pool(plugin_module)
    with pool.connection() as conn:
        clock.now += 10_000  # held the whole time: busy, not idle
        pool.reap()
        assert not conn.closed
    pool.reap()  # just returned: idle for zero seconds
    assert len(world.open) == 1


def test_next_use_after_idle_close_reconnects(plugin_module, world):
    pool, clock = _idle_pool(plugin_module, max_connections=1)
    with pool.cursor():
        pass
    clock.now += 301
    pool.reap()
    with pool.cursor():  # never fails because of the idle close
        pass
    assert len(world.all) == 2 and world.all[0].closed and len(world.open) == 1


def test_stale_connection_is_not_reused_even_if_never_reaped(plugin_module, world):
    pool, clock = _idle_pool(plugin_module, max_connections=1)
    with pool.cursor():
        pass
    clock.now += 301
    with pool.cursor():
        pass
    assert len(world.all) == 2 and world.all[0].closed and len(world.open) == 1


def test_only_the_stale_connections_are_closed(plugin_module, world):
    pool, clock = _idle_pool(plugin_module, max_connections=2)
    a, b = threading.Event(), threading.Event()

    def hold(ev_in, ev_out):
        with pool.connection():
            ev_in.set()
            ev_out.wait(5)

    done = threading.Event()
    t1 = threading.Thread(target=hold, args=(a, done))
    t2 = threading.Thread(target=hold, args=(b, done))
    t1.start(); a.wait(5); t2.start(); b.wait(5)
    done.set(); t1.join(); t2.join()
    assert len(world.open) == 2
    clock.now += 200
    with pool.connection() as fresh:  # takes the most recently returned one
        pass
    clock.now += 200  # one idle 400s, one idle 200s
    pool.reap()
    assert len(world.open) == 1 and world.open[0] is fresh


def test_idle_connection_is_closed_by_the_pool_itself(plugin_module, world):
    pool = plugin_module._BoundedPool(URL, 2, idle_seconds=0.05)
    with pool.cursor():
        pass
    deadline = time.monotonic() + 5
    while world.open and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not world.open
    with pool.cursor():
        pass
    assert len(world.all) == 2
    pool.close()
    assert not world.open


def test_pool_idle_seconds_setting(plugin_module, monkeypatch):
    monkeypatch.delenv("PROSPECTA_POOL_IDLE_SECONDS", raising=False)
    p = plugin_module.ProspectaProvider()
    assert p._resolve_pool_idle_seconds() == 300
    for raw, want in (("45", 45), (60, 60), ("0", 300), ("junk", 300), (-2, 300)):
        p._config = {"pool_idle_seconds": raw}
        assert p._resolve_pool_idle_seconds() == want
    monkeypatch.setenv("PROSPECTA_POOL_IDLE_SECONDS", "9")
    assert p._resolve_pool_idle_seconds() == 9


def test_build_memory_passes_the_idle_timeout_to_the_pool(plugin_module, world):
    m = plugin_module._build_memory(
        2, pool_idle_seconds=17, database_url=URL, embed=stub_embed, bank_id="x")
    assert m._pool._idle_seconds == 17
