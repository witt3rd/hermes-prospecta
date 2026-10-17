"""The two live profiles load through the plugin with no error and no warning.

``AUGUR`` and ``FORGE`` are the key sets of ``prospecta.json`` as the two live profiles carry them (read, never
written, on roger: ``forge/profiles/augur/prospecta.json`` and ``forge/profiles/forge/prospecta.json``). They hold
no secret: the keys and the model ids are all there is; the database address and the API key come from each
profile's ``.env`` and are made up here. The environment is what the profiles carry beside the file:
``PROSPECTA_LLM_MODEL``, ``PROSPECTA_LINKER=off`` and ``PROSPECTA_JEV=off``.

Two of the keys are stale and must stay harmless: ``linked_writes`` (the plugin no longer reads it: the write path is
chunk-only and the library has no linker) and ``note_date`` (removed earlier). The test runs the real library on a
real Postgres, the plugin's real ``initialize`` and ``_build_llm``, and one pass of every path a live turn takes.
"""
from __future__ import annotations

import json
import logging

import psycopg
import pytest

from tests.conftest import EMBED_DIM, FakeRerank, stub_embed
from tests.test_mode1a_hybrid import FakeWriterLLM, TODAY
from tests.test_prefetch_contract import FakeJev, jev_decision

AUGUR = {
    "embedder_kind": "litellm",
    "embedder_model": "openrouter/openai/text-embedding-3-large",
    "embedding_dim": 1536,
    "prefetch_enabled": "true",
    "linked_writes": True,
    "recall": {
        "prefetch": {
            "mode": "contract",
            "deadline_seconds": 7,
            "query_writer_model": "openrouter/google/gemini-2.5-flash-lite",
        }
    },
    "note_date": True,
    "decision_model": "typesafe/jev-1.14",
    "reranker_model": "qwen/qwen3-reranker-8b",
}
FORGE = json.loads(json.dumps(AUGUR))     # the same key set today; kept as its own profile so a divergence is a one-line edit
PROFILES = {"augur": AUGUR, "forge": FORGE}

ROOF = "The Henley branch roof will be replaced in June, the board agreed."
ROOF_Q = "What did we decide about the Henley branch roof?"
LIVE_ENV = {"PROSPECTA_LLM_MODEL": "openrouter/test/answer-model", "PROSPECTA_LINKER": "off", "PROSPECTA_JEV": "off",
            "OPENROUTER_API_KEY": "test-key-not-a-secret"}
CLEAR_ENV = ("PROSPECTA_PREFETCH", "PROSPECTA_PREFETCH_MODE", "PROSPECTA_DECISION_MODEL", "PROSPECTA_RERANKER_MODEL",
             "PROSPECTA_QUERY_WRITER_MODEL", "PROSPECTA_RECALL_STAGE3", "PROSPECTA_TIMEZONE", "PROSPECTA_LINKED_WRITES",
             "PROSPECTA_DUAL_WRITE_BANK", "PROSPECTA_SHADOW_BANK", "PROSPECTA_JEV_MODEL", "PROSPECTA_SCHEMA",
             "PROSPECTA_DATE_CUE_GATE", "PROSPECTA_DATE_TIMEOUT", "PROSPECTA_NOTE_DATE")


def stub_embed_1536(texts):
    """The profiles' bank is 1536 wide: the 32-wide stub padded with zeros (no network)."""
    return [v + [0.0] * (1536 - EMBED_DIM) for v in stub_embed(texts)]


def _live(plugin_module, name, pg_url, hermes_home, monkeypatch, config=None, extra_env=None):
    for k in CLEAR_ENV:
        monkeypatch.delenv(k, raising=False)
    for k, v in {**LIVE_ENV, **(extra_env or {})}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PROSPECTA_DATABASE_URL", pg_url)
    monkeypatch.setenv("PROSPECTA_BANK_ID", f"{name}-v2")
    (hermes_home / "prospecta.json").write_text(json.dumps(config or PROFILES[name]), encoding="utf-8")
    p = plugin_module.ProspectaProvider()
    monkeypatch.setattr(p, "_build_embedder", lambda: stub_embed_1536)     # the only stand-in: no network call
    return p


def _rows(pg_url, sql, *args):
    with psycopg.connect(pg_url) as c:
        return c.execute(sql, args).fetchall()


def _warnings(caplog):
    return [f"{r.name} {r.levelname}: {r.getMessage()}" for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_the_live_profile_loads_and_every_live_path_runs_without_an_error_or_a_warning(
        plugin_module, name, pg_url, hermes_home, monkeypatch, caplog):
    p = _live(plugin_module, name, pg_url, hermes_home, monkeypatch)
    writer, jev, rerank = FakeWriterLLM(), FakeJev(decision=jev_decision("new_ask", 0.9, stands=0.95)), FakeRerank()
    with caplog.at_level(logging.DEBUG):
        p.initialize(session_id="live-s1", hermes_home=str(hermes_home))
        try:
            # what the profile says, as the plugin reads it
            cfg = p._prefetch_cfg
            assert p._prefetch_enabled() is True and cfg.active and cfg.mode == "contract"
            assert cfg.deadline_s == 7.0 and cfg.query_writer_model == "openrouter/google/gemini-2.5-flash-lite"
            assert cfg.decision_model == "typesafe/jev-1.14" and p._reranker_model() == "qwen/qwen3-reranker-8b"
            assert p._recall_stage3() is False
            # what the real library was handed: the stages the profile's models name, no Stage 3 gate, no linker
            m = p._memory
            assert (m._reranker_model, m._decision_model) == ("qwen/qwen3-reranker-8b", "typesafe/jev-1.14")
            assert callable(m._rerank_transport) and m._jev is None and m._synth_llm is None
            assert callable(p._llm)                      # the real _build_llm ran: PROSPECTA_LLM_MODEL drives ask() and the writer's call
            # the bank the library created for the profile (its defaults: Stage 1 hybrid_chunk, Stage 2 cross_encoder)
            assert _rows(pg_url, "SELECT embedding_dim FROM banks WHERE bank_id = %s", f"{name}-v2") == [(1536,)]
            m._rerank_transport = rerank
            m._llm = writer                              # the model calls themselves are the fakes
            p._jev_transport, p._today = jev, lambda: TODAY

            # write path: chunk-only (the tool and the turn hook)
            out = json.loads(p.handle_tool_call("prospecta_retain", {"content": ROOF, "source": "note:roof",
                                                                     "tags": ["decision"]}))
            assert "document_id" in out and "error" not in out
            p.sync_turn("When is the roof done?", "In June.", session_id="live-s1")
            p._sync_thread.join(timeout=10)
            kinds = {r[0] for r in _rows(pg_url, "SELECT DISTINCT kind FROM memory_items WHERE bank_id = %s", f"{name}-v2")}
            assert kinds == {"chunk"}
            assert len(_rows(pg_url, "SELECT 1 FROM documents WHERE bank_id = %s", f"{name}-v2")) == 2

            # read paths: the tools and the contract prefetch
            found = json.loads(p.handle_tool_call("prospecta_search", {"query": "Henley branch roof"}))
            assert "note:roof" in [r["source"] for r in found["results"]] and found["reranked"] is True
            m._llm = lambda messages=None, **kw: "The roof will be replaced in June [1]."
            asked = json.loads(p.handle_tool_call("prospecta_recall", {"query": "When is the Henley roof replaced?"}))
            assert asked["answer"] == "The roof will be replaced in June [1]." and asked["sources"]
            m._llm = writer
            block = p.prefetch(ROOF_Q)
            assert ROOF in block and len(jev.decisions()) == 1 and writer.prompts
        finally:
            p.shutdown()
    assert _warnings(caplog) == []


def test_the_stale_keys_and_variables_load_silently(plugin_module, pg_url, hermes_home, monkeypatch, caplog):
    """A profile may still carry the settings of code that is gone: they are read and ignored, with no warning."""
    stale = {**AUGUR, "shadow_mode": "search", "recall.date_cue_gate": "true", "recall.date_timeout_seconds": 6,
             "recall": {**AUGUR["recall"], "date_cue_gate": False}}
    p = _live(plugin_module, "augur", pg_url, hermes_home, monkeypatch, config=stale, extra_env={
        "PROSPECTA_LINKED_WRITES": "1", "PROSPECTA_DATE_CUE_GATE": "on", "PROSPECTA_DATE_TIMEOUT": "3",
        "PROSPECTA_NOTE_DATE": "1"})
    with caplog.at_level(logging.DEBUG):
        p.initialize(session_id="s", hermes_home=str(hermes_home))
        try:
            assert p._memory._synth_llm is None and p._memory._jev is None
            assert "document_id" in json.loads(p.handle_tool_call("prospecta_retain", {"content": "a", "source": "n"}))
        finally:
            p.shutdown()
    assert _warnings(caplog) == []


def test_a_profile_with_the_old_prefetch_mode_off_gets_no_prefetch_and_no_warning(
        plugin_module, pg_url, hermes_home, monkeypatch, caplog):
    """``recall.prefetch.mode: off`` was the old synthesised answer; it is no prefetch now (and says nothing)."""
    cfg = {**AUGUR, "recall": {"prefetch": {**AUGUR["recall"]["prefetch"], "mode": "off"}}}
    p = _live(plugin_module, "augur", pg_url, hermes_home, monkeypatch, config=cfg)
    with caplog.at_level(logging.DEBUG):
        p.initialize(session_id="s", hermes_home=str(hermes_home))
        try:
            jev = FakeJev(decision=jev_decision("new_ask", 0.9))
            p._jev_transport = jev
            assert p._prefetch_cfg.mode == "off" and not p._prefetch_cfg.active
            assert p.prefetch(ROOF_Q) == "" and not jev.requests
        finally:
            p.shutdown()
    assert _warnings(caplog) == []
