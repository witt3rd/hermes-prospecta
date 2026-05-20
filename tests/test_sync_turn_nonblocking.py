"""sync_turn must be non-blocking."""
from __future__ import annotations

import time


def test_sync_turn_returns_immediately(provider_with_stubs, monkeypatch):
    p = provider_with_stubs

    # Patch the memory's retain to be slow.
    real_retain = p._memory.retain
    delay = 1.5

    def slow_retain(content, **kw):
        time.sleep(delay)
        return real_retain(content, **kw)

    monkeypatch.setattr(p._memory, "retain", slow_retain)

    t0 = time.time()
    p.sync_turn("hello", "world", session_id="s1")
    elapsed = time.time() - t0
    assert elapsed < 0.5, f"sync_turn blocked for {elapsed:.2f}s"


def test_sync_turn_writes_eventually(provider_with_stubs):
    p = provider_with_stubs
    before = p._memory.bank_stats("test").documents

    p.sync_turn("user says hello", "assistant says hi", session_id="s2")

    # Wait for the daemon thread
    if p._sync_thread is not None:
        p._sync_thread.join(timeout=10.0)

    after = p._memory.bank_stats("test").documents
    assert after >= before + 1
