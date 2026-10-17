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
library's search into the Hermes memory hooks and the agent's tools. The text
model (the answer `prospecta_recall` writes, and the callable the contract prefetch's
query writer uses) reads `PROSPECTA_LLM_MODEL` (env) or `llm_model` (config) and calls
`prospecta.defaults.make_default_llm` — same pattern as Hindsight. The plugin refuses to
start with no model set. No `ctx.llm` dependency.

The library is pinned in `pyproject.toml` (`prospecta` at `fe68efd`, the library after
its removal of all legacy: one read path, one write path; it has `Memory.ask` and
`Memory.expand_and_recall`). Stage 1 and Stage 2 run where the bank says so; see
[Recall](#recall-stage-1--stage-2-cited).

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
| `prospecta_retain` | Store a note (`content`, `source`, optional `tags`): chunked and searchable at once, no model call |
| `prospecta_recall` | A grounded answer with `[n]` citations and the cited sources (`source`, date, kind, match, relevance, `exact_snippet`): the library's `ask()`, Stage 1 + Stage 2 then one answer from the best notes alone; optional `limit` and `depth`; with `recall.stage3` also the Stage 3 gate. See [Recall](#recall-stage-1--stage-2-cited) |
| `prospecta_search` | Cited hits, no answer-writing model: source, date, kind, match, relevance, `exact_snippet`, the whole chunk; optional `limit`, `depth`, `date_from` / `date_to`. See [Recall](#recall-stage-1--stage-2-cited) |

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
order, and the plugin says so (`"reranked": false`; in the contract block a note).

**Where it is switched on: the bank.** Stage 1 and Stage 2 run when the bank's
`channel_config` lists `hybrid_chunk` and its `recall_config.rerank` is
`{"enabled": true, "stage": "cross_encoder"}`. The plugin does not write a bank's
configuration; it is the operator's step (the plan's slice 5), for example:

```python
mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder",
                                  "top_n": 10, "min_score": 0.15, "timeout_s": 2.5}})
```

An **empty or unset `channel_config` is the `hybrid_chunk` default** (an old bank such as
`augur` or `forge` stores `[]`, and a stored config that names only channels the library no
longer has runs the default too). A bank with no `recall_config.rerank` keeps Stage 1's own
order. The one switch the library respects is `hybrid_chunk` with `"enabled": false`: nothing is
retrieved then, and the contract prefetch needs Stage 1. With the contract prefetch on,
`initialize` logs one warning for each thing the bank lacks: a switched-off `hybrid_chunk`, no
`cross_encoder` rerank (and one if the installed library has no `Memory.expand_and_recall`).
Banks that were written with questions only (no `chunk` notes) have nothing for Stage 1 to read;
every note this plugin writes is chunked, and nothing reads the question items an old bank holds.

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
`hybrid_chunk` channel.

**`prospecta_recall`** is the library's `ask()` (Mode 2B): Stage 1 + Stage 2 over the bank, then
one model call (`PROSPECTA_LLM_MODEL`) that answers from the best `limit` notes alone and cites
`[n]` for each statement. With nothing in memory the answer is `not in memory` and no model is
called. It returns

```json
{"answer": "The roof will be replaced in June [1].",
 "synthesis": "The roof will be replaced in June [1].\n\nSources:\n[1] [note:roof | 2026-10-12] \"...\"",
 "citations": [{"n": 1, "known": true, "source": "note:roof", "created_on": "2026-10-12",
                "snippet": "the verbatim sentence that supports it", "exact_snippet": null}],
 "sources": [{"n": 1, "source": "note:roof", "document_id": "...", "created_on": "2026-10-12", "kind": "note",
              "match_kind": "exact+semantic", "relevance_score": 0.91, "exact_snippet": "the <<roof>> ..."}]}
```

`answer` has the `[n]` markers; `synthesis` is the answer followed by its numbered sources;
`citations` ties each `[n]` to its source, date and the verbatim sentence of the chunk that
supports it; `sources` lists every note the answer may cite, with the same fields as a search hit
(without the text). `limit` (default 10) is the number of notes the answer may cite. `depth`
(`standard`, the default, or `deep`) is the library's: how wide a pool of candidates the
cross-encoder reads, so `deep` suits "everything about X" questions; it is passed to the library
unchanged (the plugin's own `depth` used to mean something else, `recall_synth`'s: it is gone).

**`recall.stage3`** (default `false`; env `PROSPECTA_RECALL_STAGE3`): with it on,
`prospecta_recall` also runs the **Stage 3** sufficiency gate: the decision model scores whether
the evidence suffices, and under 0.70 one broadened search runs (a wider pool, the date
window dropped) before the answer is written. The result then also carries `sufficient`,
`retried` and `gate`. A gate that cannot run never blocks the answer (`sufficient` is `null`
and `gate` says why). It is passed explicitly either way: off means no gate, even when the bank's
own `recall_config` enables Stage 3.

**The models** are each configurable. The plugin holds only the **profile's** value
(`prospecta.json`); when the profile sets none, the library resolves the model, in its own
order (the bank's `recall_config`, then the environment variable, then its default), so the
order and the defaults live in one place:

| `prospecta.json` key | Env (read by the library) | Library default | Used for |
|---|---|---|---|
| `llm_model` (the plugin's own; no library default) | `PROSPECTA_LLM_MODEL` | none: the plugin refuses to start without one | the answer `prospecta_recall` writes (the library's `ask()` uses the Memory's text model); the callable the query writer is called through |
| `decision_model` | `PROSPECTA_DECISION_MODEL` | `typesafe/jev-1.14` | Stage 0 of the contract prefetch (needs memory? which period?); the Stage 3 gate |
| `query_writer_model` (also `recall.prefetch.query_writer_model`, which wins) | `PROSPECTA_QUERY_WRITER_MODEL` | `openrouter/google/gemini-2.5-flash-lite` (a LiteLLM id) | the library's query writer: the contract prefetch's searches |
| `reranker_model` | `PROSPECTA_RERANKER_MODEL` | `qwen/qwen3-reranker-8b` | Stage 2, on every search |

A profile value wins over the environment. The one difference to the library's order: the
Stage 0 gate (the plugin's own Jev call) takes the profile's `decision_model`, else the
library's environment-and-default choice, and does not read the bank's
`recall_config.models.decision_model` (the Stage 3 gate and the search tools do).

**What was removed.** The plugin's own lanes are gone, because the library owns them now:
the in-memory keyword leg (an Okapi BM25 index of the bank's notes, built in a background
thread, with its fingerprint, rebuild and memory-warning machinery), the keyword-first
union, and the token-budget block. So are their options: `recall.fusion`,
`recall.block_tokens`, `recall.keyword`, `recall.keyword_unit`,
`recall.keyword_index_warn_mb`, `recall.keyword_batch_pause_ms` and their env overrides.
`prospecta_search` no longer returns `block` / `entries` / `tokens` / `keyword`: it returns
the cited `results` above (`limit` bounds their number).

**Removed with the library's legacy** (witt3rd/prospecta PR 90): `recall_synth` and its
query-formulation step (`prospecta_recall` is `ask()` now, above); the `mode` of a search
(`hybrid` / `semantic` / `lexical`: the library has one retrieval path, so `prospecta_search`
has no `mode`: a model that still sends one gets the same search); `prospecta_retain`'s
`index_text` argument (the write path is chunk-only; a stale `index_text` argument is ignored);
the date-cue gate (`recall.date_cue_gate`, `recall.date_timeout_seconds`: it wrapped the library's
date-extraction model call, which no longer exists); `linked_writes` (below); and the old
synthesised-answer prefetch (below). A profile that still carries one of these keys, or
`PROSPECTA_LINKED_WRITES` / `PROSPECTA_LINKER` / `PROSPECTA_JEV` / `PROSPECTA_DATE_CUE_GATE` in its
environment, loads without a warning: they are read by nothing.

## Prefetch

**OFF by default.** Enable via `prefetch_enabled: true` in
`$HERMES_HOME/prospecta.json` or env `PROSPECTA_PREFETCH=1`. That is the master
switch: with it off nothing below runs. With it on, `recall.prefetch.mode` is:

- `contract` (the default): the written recall contract, below. It is the only prefetch there is.
- `off`: no prefetch at all: no decision call, no search, nothing injected. (`off` used to mean
  the old synthesised-answer prefetch, which went with the library's `recall_synth`; a profile that
  still says `off` now gets no prefetch.)

Any other value logs one warning and uses `contract`.

### Prefetch, mode `contract`

Mode 1 of the hybrid architecture: **automatic recall is query expansion**, and the library
does it. Per chat message the plugin gates and decides (one typed decision: is memory needed,
and for what period of time), then makes **one call** to the library's
`Memory.expand_and_recall`: its query-writer model expands the message into structured
searches (a `query`, verbatim `terms`, a period), every search runs the library's Stage 1 +
Stage 2 in parallel with the same filters, and the records merge into one cited block. No
answer is synthesised. The design is the recall flow design's Hermes column; the words it
fixes (the Jev question list, the block's sentences) are in a file, not in code (see below).
The bank needs the `hybrid_chunk` channel (an empty `channel_config` is its default) and the
`cross_encoder` rerank ([Recall](#recall-stage-1--stage-2-cited)). The library is pinned at
`main@fe68efd`; bump the pin only with a test run against the new commit.

Per message, before the turn:

1. **Gate (free).** Hermes core already skips a bare "yes", "ok", "do it" and
   the like. The plugin also skips an empty message, a slash command and a
   message too large for anything to read (over 1,000,000 characters). A long
   paste (over `paste_chars`, default 50,000) is not cut and not asked: it is
   marked as material by its size and goes whole to the library's writer.
2. **Decide (Stage 0, the plugin's own gate).** One call to the decision model (the profile's
   `decision_model`, else the library's: `typesafe/jev-1.14` through OpenRouter's System One
   door, typed answers, about $0.0001) with the state (the agent's last answer, kept per
   session from `sync_turn`; the new line; how many turns the session has had; today's date
   in the profile's `timezone`) and the contract's nine questions. The plugin reads three of
   them: `line_kind`, `needs_memory` and `time_scope`. `social` is always zero searches;
   `needs_memory` below 0.5 is the **gate**: **no expansion call is made** (nothing is
   searched, nothing is written, nothing is injected, and the turn pays one decision call).
   The gate is the plugin's, run before the call because the window below comes from the same
   answer, so the library's own gate is switched off (`gate=False`) and never asks twice. The
   other six questions (`stands_alone` and the five `need_*`) are still asked, because they
   are the contract's words, but nothing reads them any more: the library's writer takes no
   kinds of knowledge (the contract file may drop them).

   **`time_scope`** gives every search a window of dates (the notes' dates, inclusive,
   counted back from today): `recent` (today and yesterday), `past_week` (7 days),
   `past_month` (31), `past_year` (366) or `none`. It is passed as `date_from` / `date_to`, so
   every search that names no period of its own carries it. The window is the period of the
   **memory** the line asks about (what happened, was said or was done then, so older notes would be
   worthless); a relative period that only says when the **request** happens ("this week's theme",
   tonight, next week) is `none`, and so is doubt. The wording is the contract's own; to measure how the
   model answers it, run `tools/time_scope_panel.py` (see "Measuring the time scope" below).
   A period the line **names** ("in
   July", "March 2025") is `named`: the decision model cannot turn it into dates, so no window
   is passed and the library's writer dates the search that names it (its own `date_from` /
   `date_to`; the other searches stay open). The block says when a window applied ("Only notes
   dated 2026-10-10 to 2026-10-17 were searched."). A missing, unknown or invalid answer is no
   window: a search with no date filter loses nothing, a wrong window could.
3. **Expand and search (the library).** `expand_and_recall(line, "chat", agent_last=...,
   date_from=..., date_to=..., excluded_sources=..., deadline=..., max_tokens=...,
   query_writer_model=..., gate=False, record=False)`. The query-writer model writes about 3
   searches for a short message and up to 10 to 20 for an elaborate one (at most 20), each
   standing alone: a **follow-up** is resolved against `agent_last`, the agent's last answer
   (kept from `sync_turn`; at a session's first turn with none kept, after a restart or with a
   new session id, it is read once, whole, from the latest assistant message in `state.db` of
   `HERMES_HOME` through a separate read-only connection, silently none on any error). Quoted
   phrases, identifiers and names are guaranteed in a search's `terms`. Every search runs Stage
   1 (up to 50 candidates; `excluded_sources` and the dates are filtered **in SQL** before the
   candidates are cut) and Stage 2 (the cross-encoder, against its own query) at the same time,
   inside one deadline. A record found by several searches appears once. If a search's Stage 2
   fails, its own fallback is Stage 1's order and the block says "The reranker could not rank
   these; they are in the search's own order." If the writer fails the library searches the
   message itself (one search); a search still running at the deadline is dropped.
4. **Block.** The contract's opening paragraph (it replaces the library's heading), then the
   library's block: **the searches that ran** ("Searches run": each query with its terms and
   period), the line telling the agent to **cite `[n]`** when it uses an item, then the cited
   entries, each `[n] source=... | date=... | kind=... | match=... | found_by=...` (the searches
   that found it), an `Exact match snippet: "..."` line when a lexical retriever matched, and
   the whole chunk (`kind` is left out for a note with none). It is sized by Stage 2's relevance
   floor: no count cap, no entry is ever cut. `recall.prefetch.max_tokens` (default 0 = not set)
   is the library's safety valve: it leaves out whole entries that would take the block past it
   and says how many (not set: the library's own, the bank's `recall_config.expansion.max_tokens`,
   which is off unless an operator set it). When memory could not be searched the block is one line, `(memory was not
   consulted: <reason>)`; when the searches found nothing that answers, one line lists them:
   `Memory was asked: <the searches' queries>. Nothing in it answers this.`

**Never fails a turn.** The whole function is wrapped, and the whole turn has a deadline of
5 s from the call (core waits 8 s; `deadline_seconds`, at most 7). The decision has its own
limit (`decision_timeout_seconds`, 1 s); the library is given **what is left of the deadline** and
shares it between its writer and its searches (the writer at most half, and a quarter always
left for the searches), so nothing runs past it: a late search is dropped by the library, and
a library call that has still not returned a moment after the deadline is ended by the plugin.
Each step has a fallback: no decision in 1 s (or a refusal, or no key) means a rule (three
words or fewer: nothing; anything else is expanded, with no window, and the block says
"Searches were made without the decision model."); a writer that fails gives one search with the
message itself (the library's); a library that fails, cannot run (a bank with no `hybrid_chunk`
channel), has no search that answered, or is late gives the one "not consulted" line.

**Logging.** Nothing is logged per turn, and the library is called with `record=False`, so it
writes no recall event and no model-call event for a chat message. The only line is one warning
(no text) when a turn goes without memory because the plugin itself failed or the deadline passed.

**Skill and cron turns get no recall.** A turn that is a skill invocation (a slash skill, a skill bundle, stacked skills, a preloaded or auto-loaded skill) or a cron wake gets no automatic recall at all: no decision call, no search, no writer call, nothing injected (not even the "not consulted" line), nothing prepared ahead. The skill looks up what it needs. The plugin knows from the turn's first line: `on_turn_start` receives the whole message and flags a turn that begins with one of Hermes' `[IMPORTANT: ...]` skill/cron lines; and because core's `prefetch_all` strips a skill's scaffolding, a cron wake reaches `prefetch` as the cron hint plus the job's prompt, so that hint (which starts every cron prompt, with or without a skill) is recognised too. A message that only mentions a skill, or quotes such a line mid-text, is an ordinary turn and expands as usual.

**Ahead.** Core calls `queue_prefetch` after each turn with the person's text.
In contract mode the plugin ignores that text and, when the agent's own last
answer ended on a proposal ("Shall I draft it?"), expands it on its own thread
(the message is "the person has not replied yet; prepare for the reply", the agent's
answer is `agent_last`), keyed on that answer. The next turn uses
it, once, if the decision is `answer` or `go_ahead` for the same answer. (A bare
"yes" never reaches the plugin: core's filter skips it; "ok do it" does.)

**No echoes.** The brief never hands the agent back what it already has in view: the current session's last two retained turns (`turn:<session>:<ms>`) are passed to the library as `excluded_sources` and left out **in SQL** for every search, so Stage 1 never returns them and the freed slots go to the next candidates (older turns of the same session, other sessions and notes stay). `sync_turn` names each turn's record when the turn ends; after a restart the last two are read once from the bank. `recall.prefetch.drop_echoes` (default `true`) turns it off. A record is excluded because it *is* one of the last two turns (every chunk of it, however long the turn), not because its text holds the agent's words, so a short last answer ("Done.") can never match an older turn by accident and a note that merely quotes the agent is never dropped.

**Switches and files** (all read when a session starts, so the next session
picks up a change with no gateway restart; env wins over config):

| Key (`{"recall": {"prefetch": {...}}}` or flat `recall.prefetch.*`) | Default | Env |
|---|---|---|
| `mode` | `contract` (`contract` or `off`) | `PROSPECTA_PREFETCH_MODE` |
| `contract_path` | bundled `contract/recall-contract-v1.md` | `PROSPECTA_PREFETCH_CONTRACT` |
| `max_tokens` | `0` (off) | `PROSPECTA_PREFETCH_MAX_TOKENS` |
| `deadline_seconds` | `5` (at most `7`) | `PROSPECTA_PREFETCH_DEADLINE` |
| `paste_chars` | `50000` | `PROSPECTA_PREFETCH_PASTE_CHARS` |
| `drop_echoes` | `true` | `PROSPECTA_PREFETCH_DROP_ECHOES` |
| `decision_timeout_seconds` | `1.0` | `PROSPECTA_PREFETCH_DECISION_TIMEOUT` |
| `query_writer_model` | unset: the library resolves it | `PROSPECTA_QUERY_WRITER_MODEL` (the library's, below the profile) |

The models are `decision_model`, `query_writer_model` and `reranker_model`, set in
`prospecta.json` (top level; the writer's also under `recall.prefetch`) or the environment, see
[Recall](#recall-stage-1--stage-2-cited).

**Renamed and removed settings.** `recall.prefetch.question_model` is renamed
`recall.prefetch.query_writer_model` (the profile-wide `query_writer_model` also works). The old
key is read as a **deprecated alias** for this release: it still sets the writer model, and
`initialize` logs one warning line naming the new key. The other old settings no longer apply, and
a session that still has one logs one warning line: `recall.prefetch.decision_model` and the
`PROSPECTA_PREFETCH_DECISION_MODEL` / `PROSPECTA_PREFETCH_QUESTION_MODEL` env vars (use
`decision_model` / `query_writer_model` and their environment variables), and
`recall.prefetch.writer_timeout_seconds` / `search_timeout_seconds` with
`PROSPECTA_PREFETCH_WRITER_TIMEOUT` / `PROSPECTA_PREFETCH_SEARCH_TIMEOUT`: the library owns the
writer and the searches now and runs them inside the deadline, so the plugin keeps only the limit
of the one stage it owns, the decision (it must be above 0; a bad value logs one warning and the
default is used; it is also capped by what is left of `deadline_seconds`).

The models are called with the profile's own keys (`OPENROUTER_API_KEY`), and
see only what the agent sees: the person's line and the agent's last answer.

**The contract file.** `contract/recall-contract-v1.md` holds, word for word
from the design: the Jev question list (`jev-questions`, JSON), the block's first paragraph
(`block-intro`) and the fixed sentences (`lines`: "Nothing in it answers this", the
search-order line, the window line, "Searches were made without the decision model" and the
"not consulted" line). Swapping the file (`contract_path`) is enough to change any of them; the
file must keep the section names and the question ids `line_kind` and `needs_memory`
(`time_scope` is optional: without it no window is ever asked for). What changed since the
design, and why: the ninth question `time_scope` (Stage 0 also finds the period); the entries are
the library's cited evidence instead of `[source | date | rated]` labels, and Jev no longer rates
them (Stage 2 does), so the two `rated-*` lines are gone; the search-order line names the
reranker, not Jev; the window line is new; and the recall is now query expansion by the library,
so the **question-writer prompt is no longer in the file** (the writer's prompt is the library's),
the block lists the **searches** that ran (and the library's cite-`[n]` line) instead of
`Asked:` questions, `Memory was asked:` now lists the searches, and the `asked` and `more` lines
are gone (the library says how many records its token valve left out). Everything else is code:
the gate, the 0.5 threshold, the parallel searches (the library's), the deadline and every
fallback.

**Measuring the time scope.** `tests/data/time_scope_panel.json` is a panel of invented lines: half
where a relative period only modifies the request (expected `none`), half where the line limits the
memory to a past period (expected `recent`, `past_week`, `past_month`, `past_year` or `named`).
`python tools/time_scope_panel.py CONTRACT [CONTRACT_AFTER] [--runs 3]` sends each line (3 times) through
the real decision transport with `OPENROUTER_API_KEY` from the environment (never printed) and prints
the answers per line and the agreement per class; with two contract files it prints before and after
(`git show origin/main:contract/recall-contract-v1.md > /tmp/before.md`).

## Shadow reads

**OFF by default.** For an embedding migration: with `shadow_bank_id` set
(config or `PROSPECTA_SHADOW_BANK`), each `prospecta_search` and `prospecta_recall`
is also run against that bank, and the old/new pair is stored in
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
- The shadow side is retrieval only (embedding cost, no LLM): it searches the shadow
  bank with the same question the old side used, and compares the notes. For
  `prospecta_recall` the old side's notes are the ones `ask()` cited and its answer is
  stored on the old row. (There is no `shadow_mode`: `full` re-ran `recall_synth` on the
  shadow bank, which the library no longer has; a profile that still says `full` logs one
  warning and gets the retrieval shadow. The contract prefetch is not shadowed.)
- The old bank's `ask()` stores its own events too, so for `prospecta_recall` the pair's
  old row is a second row; filter on `trace -> 'shadow'`.
- The shadow bank must already exist, with `shadow_embedding_dim` matching it.
- **Abstention is always read with the false-abstain rate.** A side *abstains*
  on a question when it returns nothing; a question is *answerable* when the old
  side or the new side found something; a *false abstain* is an abstention on an
  answerable question. (The library's lexical-only search, which used to be a third
  leg of this evidence, is gone with its `mode`: `kw_n` is no longer recorded.)
  Every pair's two rows carry counts only
  in `trace.shadow` (`old_n`, `new_n`, and `abstain`: running totals
  `pairs`, `answerable`, `old_abstain`, `old_false_abstain`, `new_pairs`,
  `new_abstain`, `new_false_abstain`), and the worker logs one INFO line per
  pair, e.g. `prospecta shadow pair tool=prospecta_search old_n=10 new_n=10
  jaccard=0.43 | abstain old 0/40 (0.0%) false-abstain old 0/38 (0.0%);
  abstain new 2/40 (5.0%) false-abstain new 2/38 (5.3%)`. No question or note
  text is in it. A shadow side that errored is left out of its own counts. The
  pair itself compares the lists the agent got (the old bank's search, or the
  notes its `ask()` cited, against the new bank's search).

## Dual writes

**OFF by default**; with it off nothing changes (no extra `Memory`, no extra thread). It
prepares a switch to a newer bank. The write path is **chunk-only**: `sync_turn` and
`prospecta_retain` call the library's `retain(content, source, tags, metadata)` and nothing
else, no model call, and the library has no linker, so no entities or links are written.

**`linked_writes` is removed.** It built the main `Memory` with the library CLI's stages and wrote
`child_chunks`; with the write path chunk-only it had no effect left (the stages it built are
read-side stages: the plugin builds the ones it needs itself, from the profile's models). A profile
that still carries `linked_writes: true`, `PROSPECTA_LINKED_WRITES`, `PROSPECTA_LINKER=off` or
`PROSPECTA_JEV=off` loads with no warning (a test loads the live profiles' exact key sets against the
real library). One consequence: the main `Memory` no longer gets the CLI's `synth_llm` (a hard-coded
answer model, built when `OPENROUTER_API_KEY` is set), so the library's `ask()` answers with the
Memory's own text model, `PROSPECTA_LLM_MODEL`. (That call is recorded without token and cost
figures, as the library does for a callable that returns plain text.)

- `dual_write_bank_id` queues every note the main bank accepted (`sync_turn`
  and `prospecta_retain`) for one background daemon thread (bounded queue, 16
  pending, same worker as the shadow reads) that retains it into that bank with
  its own embedder, as chunks. The agent's turn never
  waits for it and it can never fail it.
- The second write is idempotent: a note whose content hash is already in the
  second bank is skipped, so a catch-up copy and the dual write cannot
  duplicate.
- A failed second write (or a note that does not fit the queue, or is still
  pending after the 30 s shutdown drain) is logged in full and appended to the
  retry file (JSONL: content, source, tags, error, failed_at). The
  worker replays that file right after each `initialize`; notes that fail
  again stay in a fresh file. (A retry line an older version saved with an
  `index_text` replays without it.)
- The second `Memory` is built once at `initialize` and closed in `shutdown`
  after the queue drains. The bank must already exist; otherwise dual write
  stays OFF with an ERROR log (the catch-up copy covers the gap).
- Timestamps: the library's `retain` takes no creation time, so the second
  bank stamps `documents.created_at` when the second write runs (seconds after
  the main one, or at replay time for a retried note). `created_on` is derived
  from the note's own content/metadata in both banks, identically.

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

`get_config_schema()` exposes twenty-eight fields (`hermes memory setup` walks them):

| Key | Default | Notes |
|---|---|---|
| `database_url` | empty | prefer `PROSPECTA_DATABASE_URL` env for host/prod; config value is fallback |
| `bank_id` | `default` | multi-tenant key; `PROSPECTA_BANK_ID` env overrides |
| `embedder_kind` | `litellm` | one of `litellm`, `sentence_transformers`, `openai` |
| `embedder_model` | empty | provider-specific default if empty |
| `llm_model` | empty | LiteLLM model id of the text model: the answer `prospecta_recall` writes, and the callable the library's query writer is called through (e.g. `anthropic/claude-haiku-4-5`); `PROSPECTA_LLM_MODEL` env overrides; the plugin refuses to start with neither |
| `embedding_dim` | `1536` | must match the embedder |
| `max_connections` | `4` | most database connections each Memory (main, shadow, dual write) keeps open; threads take turns on them. Raise it only if tool calls wait for a free connection |
| `pool_idle_seconds` | `300` | a pooled database connection unused for this long is closed; the next use opens a new one. `PROSPECTA_POOL_IDLE_SECONDS` env overrides |
| `connection_watch` | `true` | warn loudly when the database nears its connection limit, see [Connections](#configuration); `PROSPECTA_CONNECTION_WATCH` env overrides |
| `connection_watch_percent` | `70` | share of the database's `max_connections` in use (1-100) at which the watch warns |
| `prefetch_enabled` | `false` | the master switch of the pre-turn recall (see `recall.prefetch.mode`); opt-in spine cost |
| `shadow_bank_id` | empty (OFF) | shadow-read bank, see [Shadow reads](#shadow-reads); `PROSPECTA_SHADOW_BANK` env overrides (set it empty to force OFF) |
| `shadow_embed_model` | `openrouter/openai/text-embedding-3-large` | embedder for the shadow bank |
| `shadow_embedding_dim` | `1536` | dimensionality of the shadow bank |
| `dual_write_bank_id` | empty (OFF) | second bank every note is also written to (chunks), see [Dual writes](#dual-writes); `PROSPECTA_DUAL_WRITE_BANK` env overrides (set it empty to force OFF) |
| `dual_write_embed_model` | `openrouter/openai/text-embedding-3-large` | embedder for the dual-write bank |
| `dual_write_embedding_dim` | `1536` | dimensionality of the dual-write bank |
| `timezone` | `America/Los_Angeles` | IANA zone whose calendar day the library saves as each new note's date (handed to it as `PROSPECTA_TIMEZONE`), see [Note dates](#note-dates) |
| `hash_precheck` | empty (on) | before saving, look for the same text already in the bank and skip the write (no embedding calls); `PROSPECTA_HASH_PRECHECK` env overrides. Same text under another source is logged and skipped |
| `tool_guide` | empty (on) | add a short guide to the memory tools (search vs recall, depth, dates, retain) to the system prompt block; `false` or `PROSPECTA_TOOL_GUIDE=off` turns it off |
| `decision_model` | unset (the library: `typesafe/jev-1.14`) | the Stage 0 decision model (contract prefetch) and the Stage 3 gate model; the profile value wins over `PROSPECTA_DECISION_MODEL`. See [Recall](#recall-stage-1--stage-2-cited) |
| `query_writer_model` | unset (the library: `openrouter/google/gemini-2.5-flash-lite`) | the small text model that expands the contract prefetch's message into searches (a LiteLLM id); also read as `recall.prefetch.query_writer_model` (which wins); the profile value wins over `PROSPECTA_QUERY_WRITER_MODEL` |
| `reranker_model` | unset (the library: `qwen/qwen3-reranker-8b`) | the Stage 2 cross-encoder (OpenRouter rerank); the profile value wins over `PROSPECTA_RERANKER_MODEL` |
| `recall.stage3` | `false` | add the Stage 3 sufficiency gate to `prospecta_recall` (the library's `ask()`); `PROSPECTA_RECALL_STAGE3` env overrides |
| `recall.prefetch.mode` | `contract` | `contract` (the written recall contract: one decision, then the library's query expansion) or `off` (no prefetch), see [Prefetch](#prefetch); `PROSPECTA_PREFETCH_MODE` env overrides |
| `recall.prefetch.contract_path` | empty (the bundled file) | the recall contract file; `PROSPECTA_PREFETCH_CONTRACT` env overrides |
| `recall.prefetch.max_tokens` | `0` (off) | safety valve on the contract block (the library's); `PROSPECTA_PREFETCH_MAX_TOKENS` env overrides |
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
timestamp. No LLM is needed: each fact is written as `chunk` items, as every note is (the
library's write path is chunk-only), which is what hybrid recall (Stage 1) reads, so an
imported note can be found by the contract prefetch and `prospecta_search`; only the embedder
is called. A bank imported by an older version holds question items beside the chunks (or
question items only): nothing reads the question items; a bank with no chunks is re-imported
after deleting those documents (a re-run leaves an unchanged document alone, so it does not
add the chunks).

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
