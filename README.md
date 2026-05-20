# hermes-prospecta

Prospecta memory provider plugin for Hermes Agent. Bilateral prospective
synthesis memory — questions-on-questions retrieval with three-channel RRF
fusion. See https://github.com/witt3rd/prospecta.

Substrate: Postgres + pgvector. Two deployment modes:

- BYO Postgres — set `DATABASE_URL`.
- Embedded — docker-compose Postgres spun up on first init (requires docker).

LLM half of the spine uses `ctx.llm` (host-provided). Embedder configured per
plugin via `prospecta.defaults` (LiteLLM), `prospecta.embed.sentence_transformers`,
or `prospecta.embed.openai`.

Tools surfaced to the agent: `prospecta_retain`, `prospecta_recall`,
`prospecta_search`.

Prefetch is OFF by default (two LLM calls per recall). Enable via
`prefetch_enabled: true` in `$HERMES_HOME/prospecta.json` or env
`PROSPECTA_PREFETCH=1`.

License: MIT.
