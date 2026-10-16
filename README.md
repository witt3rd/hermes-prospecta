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
| `prospecta_search` | Raw note text, no synthesis: one token-budget block, keyword matches first (see [Recall](#recall-keyword-leg-union-and-block)) |

## Recall: keyword leg, union and block

`prospecta_search` (the tool that returns notes, no LLM) changed after the
memory bake-off, which found that on real histories a plain keyword search
found the evidence far more often than the store's fused recall, that rank
fusion can demote the best channel's top hit, and that ten short notes lose to
a block of raw text. Three parts, each with a config option (all under
`recall`, in `prospecta.json` as `{"recall": {...}}` or as the flat keys
`recall.fusion`, `recall.block_tokens`, `recall.keyword`, and the keyword leg's
`recall.keyword_unit`, `recall.keyword_index_warn_mb`,
`recall.keyword_batch_pause_ms`):

1. **Keyword leg** (`recall.keyword`, default `true`). A plain raw-text keyword
   search over the bank's **notes**: the raw note text in the `documents` table
   (`original_text`, with the note's id, source and date). Every bank has them,
   whether or not it has chunk items (the banks written before chunking have
   question items only, so an index over chunk items would be empty there).
   Okapi BM25 (k1 1.5, b 0.75), stemming **off**, OR semantics, lowercase
   tokens. Identifiers and file names are kept whole and also split:
   `deploy-config_v2.yaml` is searchable as that, and as `deploy`, `config`,
   `v2`, `yaml`. A hit is a note, so a note found by both the leg and the
   store (or by several of its items) appears once, with the note's own text.
   `recall.keyword_unit` selects the unit: `documents` (default) or `chunks`
   (the old behaviour: chunk items, `kind='chunk'`, text in `original_chunk`).
   The index lives in memory only (nothing on disk).

   *How it is built.* On first use, in a background thread; a search never
   waits for it: while it is building (or if it failed, which is logged with
   the full exception) the store's own recall is used alone and the answer says
   `"keyword": "warming"` / `"unavailable"`. Rows are read through the plugin's
   own bounded pool, one connection at a time, in keyset pages of 2,000 notes
   (no connection of its own, so `max_connections` and `pool_idle_seconds`
   apply as before; a connection is back in the pool before the page is
   indexed). The build runs at a lower thread priority, indexes in small
   slices and pauses after every page (`recall.keyword_batch_pause_ms`,
   default 50), so it cannot starve the gateway. There is **no size cap**: a
   bank of any size is indexed. Instead it logs the note count, the estimated
   index size, the time and the process memory
   (`prospecta keyword leg: indexed N documents ... index about X MB ...`), and
   with `recall.keyword_index_warn_mb` set (default `0`, off) it logs a WARNING
   when the estimate exceeds that many MB (the leg stays on). Memory is about
   five times the note text (measured: 73,000 notes of 378 characters on
   average, 27.6 MB of text, took about 160 MB and 5-7 s to build).

   *How it stays fresh.* Checked every 30 s with one cheap aggregate (note
   count, latest created / updated time). When only new notes were added (the
   notes already indexed are unchanged: same count and latest update time up to
   the last one read) just the new ones are read and appended, so a growing
   bank is not re-read in full on every write. If an indexed note was edited or
   deleted, or a note appears behind the last one read, the index is rebuilt in
   full (at most every 5 minutes), the old index serving meanwhile (memory
   briefly holds both). It is only built when something uses it: the union, or
   the shadow log. Only `mode: hybrid` searches use it.
2. **Keyword-first union** (`recall.fusion`, default `keyword_first_union`;
   `store` is the old behaviour). The top 3 of the keyword leg first, then the
   top 3 of the store, de-duplicated (by note when the leg indexes notes; the
   same chunk of the same note counts once when it indexes chunks), then the remaining slots in the store's own order. The best keyword
   hit is never demoted. If the library's `search` gets a floor-preserving
   option (a `floor` parameter), the plugin passes it on (feature-detected,
   never required); the pinned library has none there today (its
   `rerank.blend.floor` is a per-bank rerank setting that does not touch
   search), and the union gives the same guarantee either way.
3. **Token-budget block** (`recall.block_tokens`, default `1500`; `0` = the old
   list of notes). The search result is **one block** of raw note text in the
   fused order, each entry headed `[source: ... | date: YYYY-MM-DD]` (`undated`
   when the note has no date), until the budget is reached. With the default
   note unit the text of an entry is the note's own text, whichever leg found
   it (a note written since the index was built, or one too long to ever fit
   the budget when the store holds a shorter part that fits, keeps the store's
   text). An entry that does
   not fit is skipped whole and the next one that fits is taken; stored text is
   never cut. The block is built from up to 50 candidates (the store is asked
   for 50, not `limit`, when the block is on). Tokens are estimated, since the
   plugin has no tokenizer: whitespace words x 1.3, floored by characters / 4.
   The JSON answer has `block`, `entries`, `tokens`, `budget`, `keyword`,
   `sources` (source, date, `store` / `keyword`) and, when something was left
   out for size, `omitted` (source and date only).

Rollback is by config, no code change: `{"recall": {"fusion": "store",
"block_tokens": 0}}` returns the old `{"results": [...]}` list exactly (add
`"keyword": false` to also drop the index). Env overrides:
`PROSPECTA_RECALL_FUSION`, `PROSPECTA_RECALL_BLOCK_TOKENS`,
`PROSPECTA_RECALL_KEYWORD`. `prospecta_recall` and prefetch (the LLM
synthesis paths) are unchanged.

## Prefetch

**OFF by default.** Enable via `prefetch_enabled: true` in
`$HERMES_HOME/prospecta.json` or env `PROSPECTA_PREFETCH=1`. That is the master
switch: with it off nothing below runs. With it on, `recall.prefetch.mode`
picks the path:

- `off` (the default): the old prefetch, unchanged. Two LLM calls per recall
  (formulate + synthesize); the injected text is one model-written answer.
  Kept for rollback.
- `contract`: the written recall contract, below.

### Prefetch, mode `contract`

One typed decision per chat message, questions written only when needed, the
plugin's lanes asked, one rank call, a labelled block. No answer is
synthesised. The design is the recall flow design's Hermes column; the words it
fixes (the Jev question list, the question-writer prompt, the block's
sentences) are in a file, not in code (see below).

Per message, before the turn:

1. **Gate (free).** Hermes core already skips a bare "yes", "ok", "do it" and
   the like. The plugin also skips an empty message, a slash command and a
   message too large for anything to read (over 1,000,000 characters). A long
   paste (over `paste_chars`, default 50,000) is not cut and not asked: it is
   marked as material by its size and goes whole to the question writer.
2. **Decide.** One call to the decision model (`typesafe/jev-1.14` through
   OpenRouter's System One door, typed answers, about $0.0001) with the state
   (the agent's last answer, kept per session from `sync_turn`; the new line;
   how many turns the session has had) and the contract's eight questions:
   `line_kind`, `stands_alone`, `needs_memory`, `need_decisions`,
   `need_library`, `need_conversations`, `need_work`, `need_person`.
   `social` is always zero questions; `needs_memory` below 0.5 is zero
   questions (nothing is searched, nothing is written, nothing is injected, and
   the turn pays one decision call). The `need_*` at 0.5 or above are the
   sources to ask; none at 0.5 means all. Prospecta banks tell conversations
   (the turns the plugin retained) from everything else, so `need_conversations`
   alone keeps only the turns and any other `need_*` alone leaves the turns out.
   A go-ahead or an answer that still needs something outside the thread
   (`needs_memory` at 0.5 or above) is asked, from the agent's last answer.
3. **Questions.** A single standalone question is asked as written. Otherwise
   Prospecta's own `formulate_queries` runs with the contract's question-writer
   prompt and the agent's last answer as context, on the small text model
   (`recall.prefetch.question_model`), and returns 0 to 3 questions. The writer
   is told the kinds of knowledge only when some `need_*` reached 0.5 (it gets
   those); otherwise it gets "any" (the search still uses all sources). The
   agent's last answer is the only earlier turn it sees; at a session's first
   turn with none kept (after a restart or with a new session id) it is read
   once, whole, from the latest assistant message in `state.db` of
   `HERMES_HOME` through a separate read-only connection (silently none on any
   error).
4. **Ask.** Each question goes, in parallel, through the plugin's lanes: the
   dense chunk lane (with the library's floor when it has one), the raw keyword
   lane and the date lane (the date-cue gate and its timeout apply as for
   `prospecta_search`), fused by `recall.fusion` (`keyword_first_union`). One
   Jev rank call scores the best 16 of the union (the request limit of the
   venue's ranker). Entries scored 2 ("partly answers") or 3 ("holds the
   answer") go in the block; the rest are counted ("N more matched less
   well"). If Jev cannot rank, the first five of each question stand and the
   block says the order is the search's own.
5. **Block.** A markdown block: the contract's header paragraph, `Asked:`
   lines, then each entry as `[source: ... | date: ... | rated: ...]` and its
   whole text. It is sized by relevance: no count cap, no entry is ever cut.
   `recall.prefetch.max_tokens` (default 0 = off) is a safety valve that leaves
   out whole entries that would take the block past it. The tool path
   (`prospecta_search`) keeps its own 1,500-token block. When memory could not
   be searched the block is one line, `(memory was not consulted: <reason>)`;
   when the search found nothing that answers, one line says so.

**Never fails a turn.** The whole function is wrapped, and it has a deadline of
5 s from the call (core waits 8 s; `deadline_seconds`, at most 7). Each stage
has a fallback: no decision in 1 s (or a refusal, or no key) means a rule
(three words or fewer: nothing; anything else is asked as written, and the block
says "Questions formed without the decision model."); no questions in 1.5 s
(or bad JSON, or no model) means the sentence Jev rates most relevant from the
agent's last answer, or the line itself, whole (never a paste, never a bare
reply); a search that fails gives the one "not consulted" line. Past the
deadline the turn runs without memory and a late answer is dropped.

**Logging.** Nothing is logged per turn. The only line is one warning (no text)
when a turn goes without memory because of an error or the deadline.

**Ahead.** Core calls `queue_prefetch` after each turn with the person's text.
In contract mode the plugin ignores that text and, when the agent's own last
answer ended on a proposal ("Shall I draft it?"), prepares the block its
go-ahead will need on its own thread, keyed on that answer. The next turn uses
it, once, if the decision is `answer` or `go_ahead` for the same answer. (A bare
"yes" never reaches the plugin: core's filter skips it; "ok do it" does.)

**Switches and files** (all read when a session starts, so the next session
picks up a change with no gateway restart; env wins over config):

| Key (`{"recall": {"prefetch": {...}}}` or flat `recall.prefetch.*`) | Default | Env |
|---|---|---|
| `mode` | `off` (`off` or `contract`) | `PROSPECTA_PREFETCH_MODE` |
| `decision_model` | `typesafe/jev-1.14` | `PROSPECTA_PREFETCH_DECISION_MODEL` |
| `question_model` | `openrouter/qwen/qwen3.8-flash` (a LiteLLM id) | `PROSPECTA_PREFETCH_QUESTION_MODEL` |
| `contract_path` | bundled `contract/recall-contract-v1.md` | `PROSPECTA_PREFETCH_CONTRACT` |
| `max_tokens` | `0` (off) | `PROSPECTA_PREFETCH_MAX_TOKENS` |
| `deadline_seconds` | `5` | `PROSPECTA_PREFETCH_DEADLINE` |
| `paste_chars` | `50000` | `PROSPECTA_PREFETCH_PASTE_CHARS` |
| `decision_timeout_seconds` | `1.0` | `PROSPECTA_PREFETCH_DECISION_TIMEOUT` |
| `writer_timeout_seconds` | `1.5` | `PROSPECTA_PREFETCH_WRITER_TIMEOUT` |
| `search_timeout_seconds` | `4.5` | `PROSPECTA_PREFETCH_SEARCH_TIMEOUT` |
| `rank_timeout_seconds` | `1.0` | `PROSPECTA_PREFETCH_RANK_TIMEOUT` |

The four `*_timeout_seconds` limits are per stage and must be above 0 (a bad
value logs one warning and the default is used). They are not added up: each
stage is also capped by what is left of `deadline_seconds`, so the deadline
always wins. The search default is 4.5 s because a cold search of one question
(embedding plus the dense, keyword and graph lanes) can take 1.2 to 2.5 s; a
search that has not answered by its limit gives the "not consulted" line.

Both models are called with the profile's own keys (`OPENROUTER_API_KEY`), and
see only what the agent sees: the person's line and the agent's last answer.

**The contract file.** `contract/recall-contract-v1.md` holds, word for word
from the design: the Jev question list (`jev-questions`, JSON), the
question-writer prompt (`question-writer-prompt`), the block's first paragraph
(`block-intro`) and the fixed sentences (`lines`: the `Asked:` line, "Nothing in
it answers this", "N more matched less well", the search-order line, "Questions
formed without the decision model", the "not consulted" line and the two
ratings). Swapping the file (`contract_path`) is enough to change any of them;
the file must keep the section names and the eight question ids. Everything
else is code: the gate, the 0.5 and 2-of-3 thresholds, the source mapping,
the parallel asking, the rank pool of 16, the deadline and every fallback, the
block's `[source | date | rated]` entry label, and the timings.

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
- **Abstention is always read with the false-abstain rate.** A side *abstains*
  on a question when it returns nothing; a question is *answerable* when the old
  side, the new side or the keyword leg found something; a *false abstain* is an
  abstention on an answerable question. Every pair's two rows carry counts only
  in `trace.shadow` (`old_n`, `new_n`, `kw_n`, and `abstain`: running totals
  `pairs`, `answerable`, `old_abstain`, `old_false_abstain`, `new_pairs`,
  `new_abstain`, `new_false_abstain`), and the worker logs one INFO line per
  pair, e.g. `prospecta shadow pair tool=prospecta_search old_n=10 new_n=10
  kw_n=7 jaccard=0.43 | abstain old 0/40 (0.0%) false-abstain old 0/38 (0.0%);
  abstain new 2/40 (5.0%) false-abstain new 2/38 (5.3%)`. No question or note
  text is in it. A shadow side that errored is left out of its own counts. The
  pair itself is unchanged: it still compares the store's lists (the old bank's
  `recall` list against the new bank's), not the fused block.

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

`get_config_schema()` exposes thirty-seven fields (`hermes memory setup` walks them):

| Key | Default | Notes |
|---|---|---|
| `database_url` | empty | prefer `PROSPECTA_DATABASE_URL` env for host/prod; config value is fallback |
| `bank_id` | `default` | multi-tenant key; `PROSPECTA_BANK_ID` env overrides |
| `embedder_kind` | `litellm` | one of `litellm`, `sentence_transformers`, `openai` |
| `embedder_model` | empty | provider-specific default if empty |
| `llm_model` | empty | LiteLLM model id for recall/formulation (e.g. `anthropic/claude-haiku-4-5`); `PROSPECTA_LLM_MODEL` env overrides |
| `embedding_dim` | `1536` | must match the embedder |
| `max_connections` | `4` | most database connections each Memory (main, shadow, dual write) keeps open; threads take turns on them. Raise it only if tool calls wait for a free connection |
| `pool_idle_seconds` | `300` | a pooled database connection unused for this long is closed; the next use opens a new one. `PROSPECTA_POOL_IDLE_SECONDS` env overrides |
| `connection_watch` | `true` | warn loudly when the database nears its connection limit, see [Connections](#configuration); `PROSPECTA_CONNECTION_WATCH` env overrides |
| `connection_watch_percent` | `70` | share of the database's `max_connections` in use (1-100) at which the watch warns |
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
| `recall.fusion` | `keyword_first_union` | how `prospecta_search` joins the keyword leg with the store's recall; `store` = the old behaviour; `PROSPECTA_RECALL_FUSION` env overrides. See [Recall](#recall-keyword-leg-union-and-block) |
| `recall.block_tokens` | `1500` | `prospecta_search` returns one block of at most this many estimated tokens (whole entries only); `0` = the old list of notes; `PROSPECTA_RECALL_BLOCK_TOKENS` env overrides |
| `recall.keyword` | `true` | build the in-memory keyword leg (BM25, stemming off) over the bank's notes; `PROSPECTA_RECALL_KEYWORD` env overrides |
| `recall.date_cue_gate` | `true` | call the filter-extraction model only when the question has a date cue (month, weekday, `ago`, `last week`, a year, an ISO date, ...); otherwise search with no date filter and no model call (people matching still runs). `false` restores the library behaviour; `PROSPECTA_DATE_CUE_GATE` env overrides |
| `recall.date_timeout_seconds` | `6` | hard wall-clock cap on the date-extraction model call; on timeout or error the search continues with the library's regex date filter and one warning is logged. Stalled calls are abandoned (at most 8 at once; beyond that the model is skipped). `0` disables; `PROSPECTA_DATE_TIMEOUT` env overrides |
| `recall.keyword_unit` | `documents` | what the leg indexes: `documents` (the bank's raw notes, present in every bank) or `chunks` (the old behaviour: chunk items); `PROSPECTA_RECALL_KEYWORD_UNIT` env overrides |
| `recall.keyword_index_warn_mb` | `0` (off) | log a WARNING when the keyword index's estimated size exceeds this many MB (soft threshold, nothing is turned off); `PROSPECTA_RECALL_KEYWORD_INDEX_WARN_MB` env overrides |
| `recall.keyword_batch_pause_ms` | `50` | pause after each database page while the index is built or extended; `PROSPECTA_RECALL_KEYWORD_PAUSE_MS` env overrides |
| `recall.prefetch.mode` | `off` | `off` (the old prefetch) or `contract` (the written recall contract), see [Prefetch](#prefetch); `PROSPECTA_PREFETCH_MODE` env overrides |
| `recall.prefetch.decision_model` | `typesafe/jev-1.14` | the decision model of the contract prefetch; `PROSPECTA_PREFETCH_DECISION_MODEL` env overrides |
| `recall.prefetch.question_model` | `openrouter/qwen/qwen3.8-flash` | the small text model that writes the questions; `PROSPECTA_PREFETCH_QUESTION_MODEL` env overrides |
| `recall.prefetch.contract_path` | empty (the bundled file) | the recall contract file; `PROSPECTA_PREFETCH_CONTRACT` env overrides |
| `recall.prefetch.max_tokens` | `0` (off) | safety valve on the contract block; `PROSPECTA_PREFETCH_MAX_TOKENS` env overrides |
| `dual_write_retry_path` | `prospecta-dual-write-retry.jsonl` beside `prospecta.json` | notes whose second write failed |

Persisted to `$HERMES_HOME/prospecta.json`.

Connections: each Memory the plugin builds holds at most `max_connections`
shared connections, however many threads use it, and closes them on shutdown.
With shadow reads and dual write both on, the plugin uses up to three times
that. A connection lost to a database restart is replaced on next use.
A pooled connection that sits unused for `pool_idle_seconds` is closed, so an
idle provider holds no connections; the next use opens a new one.

Connection watch: at start and then every 2 minutes, a background thread asks
the database (on its own short-lived connection) how many connections are in
use and what `max_connections` is. At `connection_watch_percent` or more, or if
that query itself fails (a full server refuses it), it logs an ERROR with the
numbers and writes `$HERMES_HOME/memory-provider-health.json`, so the next turn
shows a notice. The notice is cleared once usage is back under the line. Set
`connection_watch` to `false` to turn it off.

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
