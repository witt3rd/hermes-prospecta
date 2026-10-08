# hermes-prospecta

Prospecta memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
Wraps [prospecta](https://github.com/witt3rd/prospecta) — bilateral
prospective synthesis memory — as a Hermes `MemoryProvider`.

Three-channel hybrid index (semantic + content-stem + body-stem) fused via
RRF. Write-side LLM-anticipated `index_text`. Read-side LLM-generated
multi-query expansion. LLM half of the spine reads `PROSPECTA_LLM_MODEL`
(env) or `llm_model` (config) and calls `prospecta.defaults.make_default_llm`
— same pattern as Hindsight. No `ctx.llm` dependency.

## Install

```bash
git clone https://github.com/witt3rd/hermes-prospecta /path/to/hermes-prospecta
ln -s /path/to/hermes-prospecta $HERMES_HOME/plugins/memory/prospecta

# Provider deps (pick one)
pip install 'prospecta[defaults]'                       # LiteLLM (multi-provider)
pip install 'prospecta[embed-sentence-transformers]'    # local embeddings
pip install 'prospecta[embed-openai]'                   # OpenAI direct

# Activate
hermes memory setup    # pick "prospecta"
```

### Installable / durable

`pip install .` (or `uv sync`) pulls everything the provider needs, declared in
`pyproject.toml`: `prospecta[defaults,embed-sentence-transformers]` and
`psycopg`. The sentence-transformers embedder is a hard dependency, so rebuilding
a venv cannot silently drop it. `uv.lock` is committed.

### Health check

```bash
hermes-prospecta-health
```

Exits 0 only if the database answers and the configured embedder loads and
returns `embedding_dim` vectors. Otherwise it exits 1 and prints
`!!! PROSPECTA DOWN !!! unavailable: DATABASE, EMBEDDER` (plus per-component
reasons) on stderr. Wire it into cron/systemd/monitoring. It reads the same
config/env as the plugin (`PROSPECTA_DATABASE_URL`, `$HERMES_HOME/prospecta.json`).

### Backup / restore

```bash
hermes-prospecta-backup /path/to/prospecta.dump          # pg_dump, custom format
hermes-prospecta-restore /path/to/prospecta.dump         # into an empty database
hermes-prospecta-restore --clean /path/to/prospecta.dump # overwrite existing data
```

Both use the same database URL resolution as the plugin and need `pg_dump` /
`pg_restore` on PATH (client version >= server). Backup writes to a temp file
and renames, so a failed dump never replaces a good one. Restore is a single
transaction: on any error nothing changes. Failures print
`!!! PROSPECTA BACKUP FAILED !!!` / `RESTORE FAILED` on stderr and exit 1.
Restore into a database that has the `vector` extension available (pgvector).

## Substrate (γ — capable defaults)

Two modes, resolved at `initialize()`:

- **BYO Postgres** (recommended for host/prod): set `PROSPECTA_DATABASE_URL`
  (preferred) or `DATABASE_URL`. Works with a host-wide Roger Postgres,
  Neon, Azure Database for PostgreSQL Flexible Server, RDS, or self-hosted
  pgvector.
- **Embedded** (dev-only fallback): set `PROSPECTA_ALLOW_EMBEDDED=1`; plugin
  spins up a docker-compose Postgres bundled at `docker-compose.yml`. Requires
  docker. v0.1 supports one embedded substrate at a time by default; set
  `PROSPECTA_EMBEDDED_PORT` if 5432 is already in use. For multi-profile or
  production setups, use BYO mode.

Both paths run migrations and create the default bank idempotently.

## Tools

| Tool | Purpose |
|---|---|
| `prospecta_retain` | Store with optional `index_text` override |
| `prospecta_recall` | Synthesized answer + source attribution |
| `prospecta_search` | Raw chunks, no synthesis |

## Prefetch

**OFF by default.** Spine cost: two LLM calls per recall (formulate +
synthesize). Enable via `prefetch_enabled: true` in
`$HERMES_HOME/prospecta.json` or env `PROSPECTA_PREFETCH=1`.

## Shadow reads

**OFF by default.** For an embedding migration: with `shadow_bank_id` set
(config or `PROSPECTA_SHADOW_BANK`), each `prospecta_search`, `prospecta_recall`
and prefetch is also run against that bank, and the old/new pair is stored in
`recall_events` (two rows sharing `trace.shadow.id`, `role` old/new,
`jaccard` over original document ids, `old_ms`/`new_ms`, and the `tool`),
the same shape as the library's own shadow of `Memory.recall()`.

- The agent only ever gets the old bank's answer. The shadow runs after the old
  result is ready, on one background daemon thread behind a bounded queue
  (16 pending; beyond that a pair is dropped with an ERROR log), so the old
  path pays nothing. Shadow failures never reach the caller; they are logged
  with the full exception and recorded in the new row's `trace.shadow.error`.
- The shadow `Memory` is built once at `initialize` and closed in `shutdown`
  (pending pairs are drained first, up to 30 s).
- `shadow_mode: search` (default) only searches the shadow bank (embedding cost,
  no LLM) with the same queries the old side used. `full` re-runs `recall_synth`
  on the shadow bank for `prospecta_recall` and prefetch (roughly 0.2-0.4 USD
  each); `prospecta_search` is always retrieval-only.
- `recall_synth` already stores its own old-bank event, so for `prospecta_recall`
  and prefetch the pair's old row is a second row; filter on `trace -> 'shadow'`.
- The shadow bank must already exist, with `shadow_embedding_dim` matching it.

## Configuration

`get_config_schema()` exposes eleven fields (`hermes memory setup` walks them):

| Key | Default | Notes |
|---|---|---|
| `database_url` | empty | prefer `PROSPECTA_DATABASE_URL` env for host/prod; config value is fallback |
| `bank_id` | `default` | multi-tenant key; `PROSPECTA_BANK_ID` env overrides |
| `embedder_kind` | `litellm` | one of `litellm`, `sentence_transformers`, `openai` |
| `embedder_model` | empty | provider-specific default if empty |
| `llm_model` | empty | LiteLLM model id for recall/formulation (e.g. `anthropic/claude-haiku-4-5`); `PROSPECTA_LLM_MODEL` env overrides |
| `embedding_dim` | `1536` | must match the embedder |
| `prefetch_enabled` | `false` | opt-in spine cost |
| `shadow_bank_id` | empty (OFF) | shadow-read bank, see [Shadow reads](#shadow-reads); `PROSPECTA_SHADOW_BANK` env overrides (set it empty to force OFF) |
| `shadow_embed_model` | `openrouter/openai/text-embedding-3-large` | embedder for the shadow bank |
| `shadow_embedding_dim` | `1536` | dimensionality of the shadow bank |
| `shadow_mode` | `search` | `search` = retrieval only; `full` = also shadow `recall_synth` (LLM cost) |

Persisted to `$HERMES_HOME/prospecta.json`.

## CLI

```bash
hermes prospecta status         # connection + bank info
hermes prospecta stats          # document and event counters
hermes prospecta sweep-once     # run sweeper synchronously
hermes prospecta config         # show loaded config (redacted)
```

## Importing a Hindsight bank dump

```bash
hermes prospecta import dump.json --dry-run   # plan only
hermes prospecta import dump.json             # import into the configured bank
hermes prospecta import dump.json --json      # machine-readable report
```

Dump = one JSON object with `memory_units` (facts, observations; `text`
required, plus `fact_type`, `context`, `event_date`, `created_at`,
`document_id`, `entities`, `tags`, ...), `entities` (`canonical_name`,
`aliases`, ...) and `links` (`from_id`, `to_id`, `link_type`, `weight`, ...).
See the docstring of `importer.py` for the exact mapping. Prospecta has no
entity/link tables, so entities become documents and links, provenance, event
timestamps, and unrecognised fields ride in `documents.document_metadata.hindsight`
(links on both endpoints). `created_at` of each document/item is the original
timestamp. No LLM is needed: the fact text is its own `index_text`; only the
embedder is called.

- **Idempotent**: re-running leaves unchanged documents alone (no re-embed);
  changed ones are replaced under the same source (`hindsight://<bank>/<kind>/<id>`).
  Identical text under another id is merged as `import_aliases`, not dropped.
- **No-loss report**: after writing, every input unit/entity/link is read back
  from the database. Anything missing or unimportable (empty text, link with
  no endpoint in the dump, embed failure) is listed under `LOST/UNIMPORTED`
  and the command exits `1` (`2` for fatal errors such as an unreadable dump
  or no database). `--dry-run` cannot see duplicates inside the dump.

`stats` and `sweep-once` shell out to the `prospecta` CLI shipped by the
library; ensure it's on `PATH` (or invokable via `python -m prospecta.cli`).

## Constraints (P3 / P12)

- No direct provider imports (no `openai`, no `anthropic`, no `litellm` at
  plugin import time) — embedder resolution goes through `prospecta.embed.*`
  or `prospecta.defaults`, and LLM config goes through `PROSPECTA_LLM_MODEL`
  env var or `llm_model` in `prospecta.json` (not `ctx.llm`).
- `is_available()` is non-network.
- `sync_turn` is non-blocking (daemon thread, bounded join on shutdown).
- All storage paths use `hermes_home` kwarg, not hardcoded `~/.hermes`.

## License

MIT. See LICENSE.
