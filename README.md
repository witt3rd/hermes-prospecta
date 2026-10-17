# hermes-prospecta

Prospecta memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).
Wraps [prospecta](https://github.com/witt3rd/prospecta) — bilateral
prospective synthesis memory — as a Hermes `MemoryProvider`.

Hybrid search and RAG over Postgres, the architecture of the library's
[`docs/postgres-hybrid.md`](https://github.com/witt3rd/prospecta/blob/main/docs/postgres-hybrid.md):
a note is written as chunks with **no model call**; a search is **Stage 1** (one
SQL query: semantic, stemmed and exact-phrase retrievers with reserved slots,
filtered in SQL by date and by the records to leave out) and **Stage 2** (a
cross-encoder rerank), and every hit is **cited** (source, date, kind, how it
matched, the matched words in context, the whole chunk). The plugin wires the
library's search into the Hermes memory hooks and the agent's tools. The LLM
half of the spine (recall synthesis, question expansion) reads
`PROSPECTA_LLM_MODEL` (env) or `llm_model` (config) and calls
`prospecta.defaults.make_default_llm` — same pattern as Hindsight. No `ctx.llm`
dependency.

The library is pinned in `pyproject.toml` (`prospecta` at `c52a126`). Stage 1 and
Stage 2 run where the bank says so; see [Recall](#recall-stage-1--stage-2-cited).

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
| `prospecta_retain` | Store a note: chunked and searchable at once, no model call; optional `index_text` (anticipated questions) kept as extra keys next to the chunks |
| `prospecta_recall` | Synthesized answer + cited sources (with `recall.stage3`: the library's `ask()` with the Stage 3 gate), see [Recall](#recall-stage-1--stage-2-cited) |
| `prospecta_search` | Cited hits, no synthesis: source, date, kind, match, relevance, `exact_snippet`, the whole chunk; optional `date_from` / `date_to`, see [Recall](#recall-stage-1--stage-2-cited) |
| `prospecta_recall_set` | The complete cited list for a set question about one entity (map-reduce) |

## Recall: Stage 1 + Stage 2, cited

Every recall path (`prospecta_search`, `prospecta_recall`, the contract prefetch)
reads through the library, which does the work the plugin used to do itself.

**Stage 1** (the library's `hybrid_chunk` channel): one SQL query over the bank's
`kind='chunk'` notes with three retrievers that each keep their own slots (semantic
25, stemmed 20, exact or phrase 20), fused into at most 50 candidates. The question is
read by `websearch_to_tsquery`: `"an exact phrase"` matches those words side by side,
unstemmed; `OR` and `-exclude` work; identifiers such as `v1.14` match whole. The date
window and the records to leave out are SQL filters, applied before each retriever cuts
its slots. A hit whose words matched carries them in context (`exact_snippet`, the
matched words marked `<<like this>>`).

**Stage 2** (the library's `cross_encoder` stage): a cross-encoder (default
`qwen/qwen3-reranker-8b` through OpenRouter's rerank endpoint, the profile's
`OPENROUTER_API_KEY`) scores each candidate against the question; notes under a floor
are cut, near copies collapse, the best few stay. If it fails the library keeps Stage 1's
order, and the plugin says so (`"reranked": false`; in the contract block a note and the
first five hits of each question).

**Where it is switched on: the bank.** Stage 1 and Stage 2 run when the bank's
`channel_config` lists `hybrid_chunk` and its `recall_config.rerank` is
`{"enabled": true, "stage": "cross_encoder"}`. The plugin does not write a bank's
configuration; it is the operator's step (the plan's slice 5), for example:

```python
mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder",
                                  "top_n": 10, "min_score": 0.15, "timeout_s": 2.5}})
```

A bank without it keeps working for `prospecta_search` and `prospecta_recall` (the
library's older three-channel recall: the same cited fields, with `match_kind` and
`exact_snippet` empty and no rerank), but it cannot filter by date or by the records to
leave out, so the contract prefetch needs it: with the contract prefetch on, `initialize`
logs one warning for each of the two things the bank lacks. Banks that were written
with questions only (no `chunk` notes) have nothing for Stage 1 to read; every note this
plugin writes is chunked.

**The tools.** `prospecta_search` returns

```json
{"results": [{"n": 1, "source": "...", "document_id": "...", "created_on": "2026-10-12",
              "kind": "conversation", "match_kind": "exact+semantic", "relevance_score": 0.91,
              "exact_snippet": "watched <<Electric>> <<Dreams>> at the party", "text": "the whole chunk"}],
 "count": 1, "reranked": true}
```

`match_kind` is `exact+semantic`, `exact`, `lexical` or `semantic`; `relevance_score` is the
cross-encoder's 0..1 score when Stage 2 graded the hit, else Stage 1's fused score.
`date_from` / `date_to` (`YYYY-MM-DD`, inclusive) limit it to notes dated in that window
(a note's date is the day it was saved, or the one it carries); they need the
`hybrid_chunk` channel. `prospecta_recall` answers with the library's synthesis and lists
its sources with the same fields (without the text).

**`recall.stage3`** (default `false`; env `PROSPECTA_RECALL_STAGE3`): with it on,
`prospecta_recall` answers through the library's `ask()` instead: Stage 1 + Stage 2, a
grounded answer that cites `[n]` for each statement (`synthesis` is the answer followed by
its sources, `citations` ties each `[n]` to its source, date and the verbatim sentence
that supports it), and the **Stage 3** sufficiency gate: the decision model scores whether
the evidence suffices, and under 0.70 one broadened search runs (a wider pool, the date
window dropped). A gate that cannot run never blocks the answer (`sufficient` is `null`
and `gate` says why).

**The three models** are each configurable, in this order: the profile's
`prospecta.json`, then the environment, then the default (the plan's order, and the one
`llm_model` / `PROSPECTA_LLM_MODEL` already use; it differs from the `recall.*` options
below, where the environment wins):

| `prospecta.json` key | Env | Default | Used for |
|---|---|---|---|
| `decision_model` | `PROSPECTA_DECISION_MODEL` | `typesafe/jev-1.14` | Stage 0 of the contract prefetch (needs memory? which period?); the Stage 3 gate |
| `query_writer_model` | `PROSPECTA_QUERY_WRITER_MODEL` | `openrouter/google/gemini-2.5-flash-lite` (a LiteLLM id) | the contract prefetch's question writer |
| `reranker_model` | `PROSPECTA_RERANKER_MODEL` | `qwen/qwen3-reranker-8b` | Stage 2, on every search |

**What was removed.** The plugin's own lanes are gone, because the library owns them now:
the in-memory keyword leg (an Okapi BM25 index of the bank's notes, built in a background
thread, with its fingerprint, rebuild and memory-warning machinery), the keyword-first
union, and the token-budget block. So are their options: `recall.fusion`,
`recall.block_tokens`, `recall.keyword`, `recall.keyword_unit`,
`recall.keyword_index_warn_mb`, `recall.keyword_batch_pause_ms` and their env overrides.
`prospecta_search` no longer returns `block` / `entries` / `tokens` / `keyword`: it returns
the cited `results` above (`limit` bounds their number). The shadow log's false-abstain
evidence, which used the keyword leg, now asks the library's own lexical search of the
old bank, on the shadow worker thread (see [Shadow reads](#shadow-reads)).

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

Mode 1A of the hybrid architecture: one typed decision per chat message (is memory
needed, and for what period of time), questions written only when needed, each question
asked of the library's Stage 1 + Stage 2 with the thread's last turns left out in SQL, a
labelled block of cited evidence. No answer is synthesised. The design is the recall flow
design's Hermes column; the words it fixes (the Jev question list, the question-writer
prompt, the block's sentences) are in a file, not in code (see below). The bank needs the
`hybrid_chunk` channel and the `cross_encoder` rerank ([Recall](#recall-stage-1--stage-2-cited)).

Per message, before the turn:

1. **Gate (free).** Hermes core already skips a bare "yes", "ok", "do it" and
   the like. The plugin also skips an empty message, a slash command and a
   message too large for anything to read (over 1,000,000 characters). A long
   paste (over `paste_chars`, default 50,000) is not cut and not asked: it is
   marked as material by its size and goes whole to the question writer.
2. **Decide (Stage 0).** One call to the decision model (`decision_model`, default
   `typesafe/jev-1.14` through OpenRouter's System One door, typed answers, about $0.0001)
   with the state (the agent's last answer, kept per session from `sync_turn`; the new
   line; how many turns the session has had; today's date in the profile's `timezone`) and
   the contract's nine questions: `line_kind`, `stands_alone`, `needs_memory`,
   `need_decisions`, `need_library`, `need_conversations`, `need_work`, `need_person` and
   `time_scope`. `social` is always zero questions; `needs_memory` below 0.5 is the **gate**:
   zero questions (nothing is searched, nothing is written, nothing is injected, and the
   turn pays one decision call). The `need_*` at 0.5 or above are the sources to ask; none
   at 0.5 means all. Prospecta banks tell conversations (the turns the plugin retained)
   from everything else, so `need_conversations` alone keeps only the turns and any other
   `need_*` alone leaves the turns out. A go-ahead or an answer that still needs something
   outside the thread (`needs_memory` at 0.5 or above) is asked, from the agent's last
   answer.

   **`time_scope`** gives the search a window of dates (the notes' dates, inclusive,
   counted back from today): `recent` (today and yesterday), `past_week` (7 days),
   `past_month` (31), `past_year` (366), `none` (no window), or `named`: the line names a
   period ("in July", "March 2025", an ISO date), which the decision model cannot turn
   into dates, so the question writer does (below). The window is a Stage 1 filter and the
   block says when one applied ("Only notes dated 2026-10-10 to 2026-10-17 were
   searched."). A missing, unknown or invalid answer is no window: a search with no date
   filter loses nothing, a wrong window could.
3. **Questions.** A single standalone question is asked as written (not when the time
   scope is `named`, which needs the writer). Otherwise Prospecta's own `formulate_queries`
   runs with the contract's question-writer prompt and the agent's last answer as context,
   on the small text model (`query_writer_model`), and returns 0 to 3 questions: a
   follow-up is resolved against the agent's last answer only, never the whole thread. The
   writer is told the kinds of knowledge only when some `need_*` reached 0.5 (it gets
   those); otherwise it gets "any" (the search still uses all sources). It is also told
   today's date and the time scope. Two rules are appended to its prompt whatever contract
   file is loaded: keep every `"quoted phrase"` of the person's lines verbatim, and, for a
   `named` period, return `date_from` / `date_to` (checked: two real dates, in order, not
   before 1990 nor more than a year ahead, else no window). **A quoted phrase is also
   guaranteed in code:** one the model dropped is put back (quoted where its words stayed,
   else appended to the first question), and each phrase (at most two) is also asked **on
   its own**, in quotes: the library reads a question through `websearch_to_tsquery`, where
   every word must be in the chunk, so a phrase inside a full question reaches the exact
   retriever only for a chunk that also holds all the other words. The agent's last
   answer is the only earlier turn the writer sees; at a session's first turn with none
   kept (after a restart or with a new session id) it is read once, whole, from the latest
   assistant message in `state.db` of `HERMES_HOME` through a separate read-only
   connection (silently none on any error).
4. **Ask (Stage 1 + Stage 2).** Each question goes, in parallel, to the library's
   search for up to 50 candidates, with `excluded_sources` (the thread's last two turns,
   below) and `date_from` / `date_to` filtered **in SQL** before the candidates are cut,
   then reranked by the cross-encoder (`reranker_model`). A note that two questions both
   found appears once, with its matched words if either found them. If the library's
   Stage 2 fails, its own fallback is Stage 1's order: the first five hits of each question
   stand and the block says "The reranker could not rank these; they are in the search's
   own order." If the library call itself fails, the block is the one "not consulted" line.
   There is no second rank call: Jev no longer scores the results.
5. **Block.** A markdown block: the contract's header paragraph, `Asked:` lines, then the
   library's own cited entries, each `[n] source=... | date=... | kind=... | match=...`,
   an `Exact match snippet: "..."` line when a lexical retriever matched, and the whole
   chunk (`kind` is left out for a note with none). It is sized by Stage 2's relevance
   floor: no count cap, no entry is ever cut. `recall.prefetch.max_tokens` (default 0 =
   off) is a safety valve that leaves out whole entries that would take the block past it.
   When memory could not be searched the block is one line, `(memory was not consulted:
   <reason>)`; when the search found nothing that answers, one line says so.

**Never fails a turn.** The whole function is wrapped, and it has a deadline of
5 s from the call (core waits 8 s; `deadline_seconds`, at most 7). Each stage
has a fallback: no decision in 1 s (or a refusal, or no key) means a rule
(three words or fewer: nothing; anything else is asked as written, with no
window, and the block says "Questions formed without the decision model."); no
questions in 1.5 s (or bad JSON, or no model) means the sentence Jev rates most
relevant from the agent's last answer, or the line itself, whole (never a paste,
never a bare reply); a search that fails (or a bank that cannot run it) gives the one
"not consulted" line. Past the
deadline the turn runs without memory and a late answer is dropped.

**Logging.** Nothing is logged per turn. The only line is one warning (no text)
when a turn goes without memory because of an error or the deadline.

**Ahead.** Core calls `queue_prefetch` after each turn with the person's text.
In contract mode the plugin ignores that text and, when the agent's own last
answer ended on a proposal ("Shall I draft it?"), prepares the block its
go-ahead will need on its own thread, keyed on that answer. The next turn uses
it, once, if the decision is `answer` or `go_ahead` for the same answer. (A bare
"yes" never reaches the plugin: core's filter skips it; "ok do it" does.)

**No echoes.** The brief never hands the agent back what it already has in view: the current session's last two retained turns (`turn:<session>:<ms>`) are passed to the library as `excluded_sources` and left out **in SQL**, so Stage 1 never returns them and the freed slots go to the next candidates (older turns of the same session, other sessions and notes stay). `sync_turn` names each turn's record when the turn ends; after a restart the last two are read once from the bank. `recall.prefetch.drop_echoes` (default `true`) turns it off. This replaces the old after-the-fact text comparison, including its rule that the agent's last answer counts only from 60 characters: a record is now excluded because it *is* one of the last two turns (every chunk of it, however long the turn), not because its text holds the agent's words, so a short last answer ("Done.") can no longer match an older turn by accident and a note that merely quotes the agent is never dropped.

**Switches and files** (all read when a session starts, so the next session
picks up a change with no gateway restart; env wins over config):

| Key (`{"recall": {"prefetch": {...}}}` or flat `recall.prefetch.*`) | Default | Env |
|---|---|---|
| `mode` | `off` (`off` or `contract`) | `PROSPECTA_PREFETCH_MODE` |
| `contract_path` | bundled `contract/recall-contract-v1.md` | `PROSPECTA_PREFETCH_CONTRACT` |
| `max_tokens` | `0` (off) | `PROSPECTA_PREFETCH_MAX_TOKENS` |
| `deadline_seconds` | `5` | `PROSPECTA_PREFETCH_DEADLINE` |
| `paste_chars` | `50000` | `PROSPECTA_PREFETCH_PASTE_CHARS` |
| `drop_echoes` | `true` | `PROSPECTA_PREFETCH_DROP_ECHOES` |
| `decision_timeout_seconds` | `1.0` | `PROSPECTA_PREFETCH_DECISION_TIMEOUT` |
| `writer_timeout_seconds` | `1.5` | `PROSPECTA_PREFETCH_WRITER_TIMEOUT` |
| `search_timeout_seconds` | `4.5` | `PROSPECTA_PREFETCH_SEARCH_TIMEOUT` |

The models are `decision_model`, `query_writer_model` and `reranker_model`, set in
`prospecta.json` (top level) or the environment, see [Recall](#recall-stage-1--stage-2-cited).
The old `recall.prefetch.decision_model` / `question_model` keys and their
`PROSPECTA_PREFETCH_DECISION_MODEL` / `PROSPECTA_PREFETCH_QUESTION_MODEL` env vars no
longer apply (renamed, with a new default for the writer); a session that still has one
logs a warning naming the new key. `PROSPECTA_PREFETCH_RANK_TIMEOUT` and
`rank_timeout_seconds` are gone with the rank call.

The three `*_timeout_seconds` limits are per stage and must be above 0 (a bad
value logs one warning and the default is used). They are not added up: each
stage is also capped by what is left of `deadline_seconds`, so the deadline
always wins. The search default is 4.5 s: it covers the question's embedding, the one SQL
query and the reranker's own 2.5 s limit; a search that has not answered by its limit
gives the "not consulted" line.

The models are called with the profile's own keys (`OPENROUTER_API_KEY`), and
see only what the agent sees: the person's line and the agent's last answer.

**The contract file.** `contract/recall-contract-v1.md` holds, word for word
from the design: the Jev question list (`jev-questions`, JSON), the
question-writer prompt (`question-writer-prompt`), the block's first paragraph
(`block-intro`) and the fixed sentences (`lines`: the `Asked:` line, "Nothing in
it answers this", "N more matched less well", the search-order line, the window
line, "Questions formed without the decision model" and the "not consulted"
line). Swapping the file (`contract_path`) is enough to change any of them; the
file must keep the section names and the eight original question ids (`time_scope`
is optional: without it no window is ever asked for). What changed since the design,
and why: the ninth question `time_scope` (Stage 0 now also finds the period); the
entries are the library's cited evidence instead of `[source | date | rated]` labels,
and Jev no longer rates them (Stage 2 does), so the two `rated-*` lines are gone;
"N more matched less well" now counts only entries left out by `max_tokens` (Stage 2's
own cuts are final); the search-order line names the reranker, not Jev; the window line
is new. The opening paragraph and the "Memory was asked: ... Nothing in it answers this."
line are unchanged. Everything else is code: the gate, the 0.5 threshold, the source
mapping, the parallel asking, the deadline and every fallback, the quoted-phrase and
named-period rules of the writer, and the timings.

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
  side, the new side or a plain lexical search of the old bank (the library's own
  `mode="lexical"`, Postgres full text, asked on the shadow worker thread) found
  something; a *false abstain* is an abstention on an answerable question. Every pair's two rows carry counts only
  in `trace.shadow` (`old_n`, `new_n`, `kw_n`, and `abstain`: running totals
  `pairs`, `answerable`, `old_abstain`, `old_false_abstain`, `new_pairs`,
  `new_abstain`, `new_false_abstain`), and the worker logs one INFO line per
  pair, e.g. `prospecta shadow pair tool=prospecta_search old_n=10 new_n=10
  kw_n=7 jaccard=0.43 | abstain old 0/40 (0.0%) false-abstain old 0/38 (0.0%);
  abstain new 2/40 (5.0%) false-abstain new 2/38 (5.3%)`. No question or note
  text is in it. A shadow side that errored is left out of its own counts. The
  pair itself is unchanged: it still compares the lists the agent got (the old bank's
  search against the new bank's). `kw_n` is the lexical count; it is `n/a` when that
  search fails.

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
  from the note's own content/metadata in both banks, identically. A note
  retained with `index_text` is written to the second bank with it, as in the main one.

## Note dates

The library can filter on the stored note date (`documents.created_on`) and cites it on
every hit. **The library stamps it**: `retain()` always sets `created_on`, to the note's own
date (frontmatter or `metadata`: `created_on`, `created` or `date`) or else today's date
in `PROSPECTA_TIMEZONE` (UTC when unset). The plugin used to pass `metadata={"created_on":
...}` itself; it no longer does, and the `note_date` setting is **removed** (the library
cannot be told not to stamp, so there is nothing left to switch; `PROSPECTA_NOTE_DATE` is
ignored). A note whose metadata carries a date keeps it.

- **`timezone`** (config key; an IANA name; default `America/Los_Angeles`) is kept: at
  `initialize` the plugin sets `PROSPECTA_TIMEZONE` to it (the profile's value, else an
  already-set `PROSPECTA_TIMEZONE`, else the default; an unknown name falls back to the
  default, logged), so the library stamps the profile's calendar day instead of UTC's. It is
  also the zone of "today" for the contract prefetch's time scopes. It is process-wide for
  the gateway (one profile per gateway process).
- **One helper**, `local_date_iso(tz_name, now)`: the calendar day *now* in a zone. It never
  raises: an unknown zone falls back to the default (logged), no zone data at all falls back
  to UTC.
- One edge changes: the day is now taken when the library writes the note, not when the turn
  ends, so a turn that ends within moments of local midnight may be stamped with the next day.
- Dual write: the second bank is stamped by the library when the second write runs
  (seconds later, or at replay time for a retried note), in the same zone; the retry file no
  longer carries a `created_on`.

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

`get_config_schema()` exposes thirty-two fields (`hermes memory setup` walks them):

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
| `timezone` | `America/Los_Angeles` | IANA zone whose calendar day the library saves as each new note's date (handed to it as `PROSPECTA_TIMEZONE`), see [Note dates](#note-dates) |
| `hash_precheck` | empty (on) | before saving, look for the same text already in the bank and skip the write (no embedding calls); `PROSPECTA_HASH_PRECHECK` env overrides. Same text under another source is logged and skipped |
| `tool_guide` | empty (on) | add a short guide to the memory tools (search vs recall, depth, dates, retain) to the system prompt block; `false` or `PROSPECTA_TOOL_GUIDE=off` turns it off |
| `decision_model` | `typesafe/jev-1.14` | the Stage 0 decision model (contract prefetch) and the Stage 3 gate model; the profile value wins over `PROSPECTA_DECISION_MODEL`. See [Recall](#recall-stage-1--stage-2-cited) |
| `query_writer_model` | `openrouter/google/gemini-2.5-flash-lite` | the small text model that writes the contract prefetch's questions (a LiteLLM id); the profile value wins over `PROSPECTA_QUERY_WRITER_MODEL` |
| `reranker_model` | `qwen/qwen3-reranker-8b` | the Stage 2 cross-encoder (OpenRouter rerank); the profile value wins over `PROSPECTA_RERANKER_MODEL` |
| `recall.stage3` | `false` | `prospecta_recall` answers through the library's `ask()` with the Stage 3 sufficiency gate; `PROSPECTA_RECALL_STAGE3` env overrides |
| `recall.date_cue_gate` | `true` | call the filter-extraction model only when the question has a date cue (month, weekday, `ago`, `last week`, a year, an ISO date, ...); otherwise search with no date filter and no model call (people matching still runs). `false` restores the library behaviour; `PROSPECTA_DATE_CUE_GATE` env overrides |
| `recall.date_timeout_seconds` | `6` | hard wall-clock cap on the date-extraction model call; on timeout or error the search continues with the library's regex date filter and one warning is logged. Stalled calls are abandoned (at most 8 at once; beyond that the model is skipped). `0` disables; `PROSPECTA_DATE_TIMEOUT` env overrides |
| `recall.prefetch.mode` | `off` | `off` (the old prefetch) or `contract` (the written recall contract), see [Prefetch](#prefetch); `PROSPECTA_PREFETCH_MODE` env overrides |
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
