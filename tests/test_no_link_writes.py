"""Retaining notes writes no rows to memory_links (synthetic notes only)."""
from __future__ import annotations

import psycopg


def _link_count(url):
    with psycopg.connect(url) as conn:
        if conn.execute("SELECT to_regclass('memory_links')").fetchone()[0] is None:
            return 0
        return conn.execute("SELECT count(*) FROM memory_links").fetchone()[0]


def test_retain_paths_do_not_change_memory_links(provider_with_stubs, pg_url):
    p = provider_with_stubs
    before = _link_count(pg_url)

    for i in range(3):
        p.sync_turn(
            f"User: synthetic question {i} about topic alpha{i}",
            f"Assistant: synthetic answer {i} about topic alpha{i} and beta",
            session_id="links-test",
        )
        if p._sync_thread is not None:
            p._sync_thread.join(timeout=10.0)
    for i in range(4):
        p.handle_tool_call(
            "prospecta_retain",
            {"content": f"Synthetic note {i}: alpha{i} relates to beta.",
             "source": f"synthetic:{i}"},
        )

    assert _link_count(pg_url) == before
