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

## Linked and dual writes

**Both OFF by default**; with both off nothing changes (no extra `Memory`, no
extra thread). They prepare a switch to a newer bank.

- `linked_writes` builds the main `Memory` with the stages the library CLI
  builds (`prospecta.cli._common._stage_kwargs()`: linker, Jev, rerank and
  synth LLMs, built when `OPENROUTER_API_KEY` is set; `PROSPECTA_LINKER=off` /
  `PROSPECTA_JEV=off` opt out), and retains notes with `child_chunks=True`. New
  notes get entities and links like the migrated bank. Note that the same
  stages also apply to recall on that `Memory`. The library links on its own
  worker thread; a linking (or stage build) failure is logged with the full
  exception and never fails the turn.
- `dual_write_bank_id` queues every note the main bank accepted (`sync_turn`
  and `prospecta_retain`) for one background daemon thread (bounded queue, 16
  pending, same worker as the shadow reads) that retains it into that bank with
  its own embedder, the linker on and chunk children. The agent's turn never
  waits for it and it can never fail it.
- The second write is idempotent: a note whose content hash is already in the
  second bank is skipped, so a catch-up copy and the dual write cannot
  duplicate.
- A failed second write (or a note that does not fit the queue, or is still
  pending after the 30 s shutdown drain) is logged in full and appended to the
  retry file (JSONL: content, source, tags, index_text, error, failed_at). The
  worker replays that file right after each `initialize`; notes that fail
  again stay in a fresh file.
- The second `Memory` is built once at `initialize` and closed in `shutdown`
  after the queue drains. The bank must already exist; otherwise dual write
  stays OFF with an ERROR log (the catch-up copy covers the gap).
- Timestamps: the library's `retain` takes no creation time, so the second
  bank stamps `documents.created_at` when the second write runs (seconds after
  the main one, or at replay time for a retried note). `created_on` is derived
  from the note's own content/metadata in both banks, identically. If the second
  bank asks the LLM for `index_text` it gets its own questions.

## Note dates

The library can read a date from a question ("in July", an ISO date, a year)
and filter on the stored note date (`documents.created_on`). The plugin saves
that date with every new note: `sync_turn` and `prospecta_retain` pass
`metadata={"created_on": "YYYY-MM-DD"}` to `retain` (the key the library reads;
`prospecta/_filters.py`). The same metadata goes to the dual-write bank and into
its retry file, so both banks get the same date.

- **One helper**, `local_date_iso(tz_name, now)`: the calendar day *now* in the
  `timezone` config key (an IANA name; default `America/Los_Angeles`). For a
  turn the day is fixed when the turn ends. It never raises: an unknown zone
  falls back to the default (logged), no zone data at all falls back to UTC, and
  any failure saves the note without a date. A note whose metadata already has a
  `created_on`, `created` or `date` keeps it; other metadata is never touched.
- **Default**: on, unless `linked_writes` is on. The library also uses the date
  to order the linker's temporal links, so with linked writes the change is an
  explicit choice (`note_date: true`). Otherwise the only effect is two extra
  keys in `documents.document_metadata` (`created`, `created_on`), one in the
  note's item metadata, and the date shown in recall headers. Content, tags,
  source, index text and embeddings are unchanged. `note_date: false` (or
  `PROSPECTA_NOTE_DATE=off`) turns it off.

### Filling the date on existing notes

Notes saved before this have `created_on` empty. `hermes-prospecta-fill-note-dates`
(`python fill_note_dates.py` from a checkout) fills it from each note's own
recorded time (`documents.created_at`) as the calendar day in the profile's
timezone (`--timezone`, else `prospecta.json`, else `America/Los_Angeles`).
Notes that already have a date, and notes with no text, are skipped. It is
additive: nothing but `created_on` is written and nothing is deleted. Banks
whose `created_at` is a copy or import time (a dual-write copy, an imported
dump) get that time's day.

```bash
# 1. Dry run (default): a read-only transaction, prints counts per bank only.
hermes-prospecta-fill-note-dates --profile-dir ~/.hermes/profiles/NAME --bank BANK [--bank OTHER]
#    bank=BANK notes=N already_dated=A would_fill=W no_text=T

# 2. A fresh, verified dump (a file under 24 hours old with its .sha256 beside it):
hermes-prospecta-backup /backups/prospecta-$(date +%F).dump
( cd /backups && sha256sum prospecta-$(date +%F).dump > prospecta-$(date +%F).dump.sha256 )

# 3. Write (refuses without the dump): batched, resumable, safe to re-run.
hermes-prospecta-fill-note-dates --profile-dir ~/.hermes/profiles/NAME --bank BANK \
    --apply --dump-dir /backups [--run-id MYRUN] [--batch-size 500]

# 4. Undo: dry run first, then --apply (also needs the fresh dump).
hermes-prospecta-fill-note-dates --undo --run-id MYRUN [--bank BANK]
hermes-prospecta-fill-note-dates --undo --run-id MYRUN --apply --dump-dir /backups
hermes-prospecta-fill-note-dates --undo-all --apply --dump-dir /backups
```

The database address comes from `PROSPECTA_DATABASE_URL` in the environment,
else the profile `.env` (`--profile-dir`, default `$HERMES_HOME`), else
`prospecta.json`, else `DATABASE_URL`; it is never printed. Every filled id is
recorded in the additive table `note_date_fill_log(bank_id, document_id,
filled_on, filled_at, run_id)` in the same transaction as the update (one
transaction per batch). Undo sets `created_on` back to empty for the ids of the
run (or all runs) and removes their log rows; a date that was changed by
something else since is left alone.

## Configuration

`get_config_schema()` exposes twenty fields (`hermes memory setup` walks them):

| Key | Default | Notes |
|---|---|---|
| `database_url` | empty | prefer `PROSPECTA_DATABASE_URL` env for host/prod; config value is fallback |
| `bank_id` | `default` | multi-tenant key; `PROSPECTA_BANK_ID` env overrides |
| `embedder_kind` | `litellm` | one of `litellm`, `sentence_transformers`, `openai` |
| `embedder_model` | empty | provider-specific default if empty |
| `llm_model` | empty | LiteLLM model id for recall/formulation (e.g. `anthropic/claude-haiku-4-5`); `PROSPECTA_LLM_MODEL` env overrides |
| `embedding_dim` | `1536` | must match the embedder |
| `max_connections` | `4` | most database connections each Memory (main, shadow, dual write) keeps open; threads take turns on them. Raise it only if tool calls wait for a free connection |
| `prefetch_enabled` | `false` | opt-in spine cost |
| `shadow_bank_id` | empty (OFF) | shadow-read bank, see [Shadow reads](#shadow-reads); `PROSPECTA_SHADOW_BANK` env overrides (set it empty to force OFF) |
| `shadow_embed_model` | `openrouter/openai/text-embedding-3-large` | embedder for the shadow bank |
| `shadow_embedding_dim` | `1536` | dimensionality of the shadow bank |
| `shadow_mode` | `search` | `search` = retrieval only; `full` = also shadow `recall_synth` (LLM cost) |
| `linked_writes` | `false` | linked notes, see [Linked and dual writes](#linked-and-dual-writes); `PROSPECTA_LINKED_WRITES` env overrides |
| `dual_write_bank_id` | empty (OFF) | second bank every note is also written to; `PROSPECTA_DUAL_WRITE_BANK` env overrides (set it empty to force OFF) |
| `dual_write_embed_model` | `openrouter/openai/text-embedding-3-large` | embedder for the dual-write bank |
| `dual_write_embedding_dim` | `1536` | dimensionality of the dual-write bank |
| `timezone` | `America/Los_Angeles` | IANA zone whose calendar day is saved as each new note's date, see [Note dates](#note-dates) |
| `note_date` | empty (on, except with `linked_writes`) | save each new note's date; `PROSPECTA_NOTE_DATE` env overrides |
| `hash_precheck` | empty (on) | before saving, look for the same text already in the bank and skip the write (no LLM call); `PROSPECTA_HASH_PRECHECK` env overrides. Same text under another source is logged and skipped |
| `tool_guide` | empty (on) | add a short guide to the memory tools (search vs recall, depth, dates, retain) to the system prompt block; `false` or `PROSPECTA_TOOL_GUIDE=off` turns it off |
| `dual_write_retry_path` | `prospecta-dual-write-retry.jsonl` beside `prospecta.json` | notes whose second write failed |

Persisted to `$HERMES_HOME/prospecta.json`.

Connections: each Memory the plugin builds holds at most `max_connections`
shared connections, however many threads use it, and closes them on shutdown.
With shadow reads and dual write both on, the plugin uses up to three times
that. A connection lost to a database restart is replaced on next use.

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
