# Tencent agent memory, Prospecta and Spire memory: can they share one memory contract?

Scout report, 2026-10-13. Knowledge only: I read code, docs, configs and schemas; I changed no repo
or profile, read no memory bank, session, diary or chat history, and touched no secret.
Every example below is invented. Nothing here quotes a person. Numbers about the real Prospecta
banks come only from the firstmate data notes named in section 0 (counts and mechanisms, never content).

The house-style PDF of this report is `pdf/agent-memory-unification-v1.pdf` (source and build recipe
in `pdf/`). This file is the full evidence version: every claim carries its file reference.

## 0. What I did

Read (all read-only):

| Area | What I read | Revision |
|---|---|---|
| Tencent system | `/home/dt/src/ext/TencentDB-Agent-Memory`: `README.md`, `ROADMAP.md`, `CHANGELOG.md`, `LICENSE`, `MemoryCore/README.md`, `MemoryCore/v3-api-memorycore-doc.md`, `MemoryCore/src/core/**` (prompts, record, hooks, store/isolation, persona, scene), `MemoryCore/src/config.ts`, `MemoryCore/src/utils/sanitize.ts`, `MemoryCore/hermes-plugin/memory/memory_tencentdb/{__init__.py,README.md,plugin.yaml}`, `MemoryCore/openclaw-plugin/docs/architecture.md`, `MemoryCore/pi-plugin/README.md`, `MemoryProxy/README.md` and `src/injection/injectors/*`, `MemoryKnowledge/README.md`, `MemoryPanel/README.md` and `src/panel/domain/chat-memory-governance.ts`, `agents/README.md`, `agents/hermes/README.md`, `agents/adapter-agent-development.md`, `sdk/memory-core/python/README.md`, `deploy/global-images/README.md`, `.github/workflows/pr-ci.yml` | `b1f79fc` |
| Prospecta plugin | `/home/dt/.treehouse/hermes-prospecta-3afe6c/3/hermes-prospecta`: `README.md`, `__init__.py`, tests list | `a729cea` |
| Prospecta library | `/home/dt/src/witt3rd/prospecta`: `README.md`, `PRINCIPLES.md`, `docs/{recall-stages,graph-links,limits}.md`, `docs/design/prospecta/decision-record-1.md`, migration names | `15ac86d` |
| Prospecta data notes | `.../firstmate/data/window1/{augur-switch-status,four-query-judge,augur-next-steps}.md`, `data/prospecta-gap/report.md` | 2026-10-06 and 2026-10-12 |
| Spire memory | `/home/dt/src/li/spire-venue/docs/venue.md` (sections Memory, Access contracts, Person continuity, Records and provenance, Memory proposals, The recall door, The memory tools, Memory dreaming, Projection and replay, the Library passages and search paragraphs), `server/src/memory/{zettel-judge,memory-rule,librarian-offered-memory}`; `/home/dt/src/li/spire-project/docs/notes/memory-architecture-v1/{slides.py,sources.md}` (the 2026-10-04 memory architecture note); `/home/dt/src/li/agent-binding-host/{docs/reference.md,bin/host-memory.py}`; `/home/dt/src/witt3rd/hermes-spire` (README, `skills/spire/SKILL.md`) | venue `2cc28b8a6`, project `4a1e9f74`, host `be8c002`, hermes-spire `71936fd` |
| rung | `/home/dt/src/witt3rd/rung/docs/rung-memory.md`, `docs/rung-memory-walkthrough.md`, `docs/reports/rung-memory-provider-contract-v1/build.py` (the settled contract card), `docs/reports/rung-jevmem-memory-v2/build.py` (the Jev-Mem decision) | `605b678` |
| Hermes | `/home/dt/src/li/hermes-agent/agent/memory_provider.py`, `agent/memory_manager.py`, `agent/agent_init.py:1800-1847`, `run_agent.py:4215-4240` | `415c8e0acd` (a checkout, not necessarily the live gateway) |
| Zettel provider (for the class) | `/home/dt/src/witt3rd/{hermes-zk-memory,zk-memory}` READMEs | `2fb72ea`, `1bb39fc` |

Commands run were read-only (`ls`, `sed`, `grep`, `find`, `git log`, `publishing build` into a scratch dir). One named input
is absent: the brief lists `docs/design/prospecta/brief*.md`; the Prospecta library has none (I listed every remote branch
with `git ls-tree` and found no file named `brief*`). The design record that exists is `docs/design/prospecta/{decision-record-1,plan-v2,port-decision-sheet,schema}.md`;
I used `decision-record-1.md`, `PRINCIPLES.md` and the three feature docs above.

Not done: nothing was run against a model or a database; no head-to-head recall test of the three systems was made.
Every "better" below is a design reading, not a measurement, unless a number is cited.

## 1. The general class: the agent memory contract

The same problem appears five times in what I read. Each time it is: **recall before the turn, retain after the turn,
tools the agent can call, a scope key (whose memory), provenance (where it came from), and cost (what it spent)**.

| System | Recall before a turn | Retain after a turn | Agent tools | Scope key | Provenance | Cost |
|---|---|---|---|---|---|---|
| Hermes `MemoryProvider` (the host of Prospecta and Tencent plugins) | `prefetch(query)` | `sync_turn(user, assistant)` | `get_tool_schemas` / `handle_tool_call` | `session_id` and kwargs (`user_id`, `agent_identity`, `hermes_home`) | none required | none required |
| Prospecta plugin | `prefetch` (off by default) | `sync_turn` (whole turn) | `prospecta_retain/recall/search/recall_set` | `bank_id` only | `source` string, `recall_events` | `llm_calls` table, no token columns |
| Tencent plugin | `prefetch` (L1 top 5 + L3 + L2 names) | `sync_turn` (raw turn to L0) | `memory_search`, `conversation_search`, `read_scene` | team / agent / user (+ task, session) | `source_message_ids` on L1 | quota/credit modules, not surfaced to the agent |
| rung `rung-memory/1` | `rung_memory_recall` | `rung_memory_retain` (completed turn only) | every other tool the provider lists | opaque `scope` | `id`, `observed_at`, `attrs`, `injected` ids | `cost_usd`, `calls`, declared `max_cost_usd` |
| Spire memory | recall door `POST /api/recall`, host provider for rung | venue's own sealed-turn retention; proposals door for foreign offers | `recall`, `remember`, `forget`, `read_passage` | subject (`person:N`, project, chat, agent) plus audience | citation (`note:`, `passage:`, `line:`, `proposal:`), `sources`, `standing`, `by` | `cost` and `binding.usage` on every model call |

References for the rows: Hermes `agent/memory_provider.py:1-34, 140-260`; Prospecta `hermes-prospecta/__init__.py:1130-1484`; Tencent plugin `memory_tencentdb/__init__.py:607-760, 823-905`; rung `docs/rung-memory.md`; Spire `spire-venue/docs/venue.md:6253-6476`.

Tencent's own README says it too: "An adapter generally has three responsibilities: 1. Write completed turns or sessions to L0. 2. Recall L1/L2/L3 before constructing the next prompt. 3. Inject recalled results into the Agent as bounded, clearly labeled context."
(`MemoryCore/README.md:167-174`). The OpenClaw client plugin (`MemoryCore/openclaw-plugin/docs/architecture.md`, section 3)
and the Pi plugin (`MemoryCore/pi-plugin/README.md`) are two more instances.

Consequence: the sensible question is not "which store?" but "which contract, and can each store answer to it?".

## 2. The Tencent system

### 2.1 What it is

Four services plus SDKs, MIT-licensed ("Copyright (C) 2026 Tencent", `LICENSE:1-26`), Node 22.16+ (`README.md:12`, `MemoryCore/README.md:49-57`):

- **MemoryCore** (port 8420): the store and API. L0 to L3 chat memory, Skills, metadata (users, teams, agents, tasks, assets, ACL). SQLite plus local files by default; Tencent VectorDB and MongoDB are optional/experimental (`MemoryCore/README.md:30-50`, `CHANGELOG.md:12-26`).
- **MemoryKnowledge** (8421): Wiki (LLM ingests documents into linked pages, FTS5 search and a link graph) and CodeGraph (symbols, calls, impact) (`MemoryKnowledge/README.md:1-30`).
- **MemoryProxy** (8096): a transparent LLM request proxy; point an agent's base URL at it and it does session init, injection, write-back, rate limits and billing reports (`MemoryProxy/README.md:1-50`).
- **MemoryPanel** (8125): the web console for teams, agents, assets and ACLs (`MemoryPanel/README.md:1-20`).

Maturity: first public release 2.0.0-beta.1 on 2026-07-21; 2.0.2 on 2026-10-09 (`CHANGELOG.md:12, 338`). The docs disagree about the current release
(`README.md:304` says v2.0.0; `ROADMAP.md:6` says v2.0.1-beta.1; the changelog says 2.0.2). The proxy README describes L2 as "Agent Profile" and L3 as "Team / Global memory"
(`MemoryProxy/README.md:64-65`) while MemoryCore means L2 = scenes, L3 = persona (`MemoryCore/README.md:7`). The public repo has no unit-test files (`find -name '*.test.*' -o -name '*.spec.*' -o -name __tests__` returns none; what exists is a few end-to-end scripts such as `MemoryPanel/scripts/e2e-skill-authz.sh` and `__contract__` helper modules without a runner); a `vitest.config.ts` is present but no tests. CI only packs the plugin, validates its manifest, guards size and checks skill-queue isolation (`.github/workflows/pr-ci.yml:18-160`).

### 2.2 Layers: what is stored, how it is extracted, how it is recalled

| Layer | Stored | Extracted how | Recalled how |
|---|---|---|---|
| L0 | every user/assistant message with role and timestamp, 1 to 8192 characters per message (`v3-api-memorycore-doc.md:109-143`) | written directly; text is cleaned of injected tags, gateway metadata, line-leading timestamps, media markers, base64 images (`src/utils/sanitize.ts:11-73`); assistant code fences are stripped before capture (`:75-90`) | `conversation_search` tool (BM25 and optional vectors, time filters) |
| L1 atom | one fact: type `persona`, `episodic`, `instruction` (plus `work_fact/task/method/artifact` in work mode), priority 0 to 100 (-1 = strict standing order), scene name, source message ids, activity start/end time, timestamp trail, version (`src/core/record/l1-writer.ts:31-100`) | an LLM pass splits scenes and extracts atoms from "new messages" with background messages; rule "better to omit than to keep trivia"; episodic times must be computed as absolute ISO dates from the message timestamp (`src/core/prompts/l1-extraction.ts:15-60`); a second LLM pass compares each new atom with the candidate pool and chooses store / skip / update / merge, cross-type and many-to-many (`src/core/prompts/l1-dedup.ts:16-70`) | top 5 above score 0.3 by BM25 plus vector, rank-fused (RRF, k=60) when an embedder exists, else BM25 only (`src/core/hooks/auto-recall.ts:459-532, 644-651`; `MemoryCore/README.md:47`) |
| L2 scenario | markdown files per scene, rewritten by a sandboxed LLM agent that can read and write only `scene_blocks/` (`src/core/scene/scene-extractor.ts:1-20`) | after L1 | only the list of scene names is injected; the model reads one by tool (`read_scene`) (`hermes-plugin/.../__init__.py:284-300, 700-715`) |
| L3 persona | one `persona.md`, regenerated on triggers: agent request, cold start, first scene, every 50 new memories (`src/core/persona/persona-trigger.ts:36-95`, `src/config.ts:578`) | after L2 | injected whole every turn |

Pipeline cadence (code defaults): L1 after every 5 conversations with warm-up 1, 2, 4, 5; or after 600 s idle; L2 10 s after L1, no more often than every 900 s per session, and forced at 3600 s
(`src/config.ts:584-592`; the comments beside the interface at `:66-80` quote other numbers, 30 s idle and 90 s delay, so the file disagrees with itself). Recall caps: 5 results, threshold 0.3, 5 s timeout (`src/config.ts:89-95, 596-599`; `auto-recall.ts:107`).
No LLM call at recall; no rerank; no recency; no graph; no date reasoning on the question (activity times are printed beside a recalled atom, `auto-recall.ts:803-840`, not used to rank). The API does have explicit `time_start` / `time_end` filters (`v3-api-memorycore-doc.md:144-160, 196-240`).
The recalled block is followed by a tool guide that limits the model to 3 searches per turn (`auto-recall.ts:41-52`).

Skills (a second memory type) are versioned procedure documents with resource files, extracted from tasks (`v3-api-memorycore-doc.md:330-545`; `README.md:112-120`).

### 2.3 Scopes, teams, and the shared-server model

- "All agents share the same memory server": one proxy URL, zero code in the agent (`README.md:55-80`). The shared thing is the server, not one pool of memories.
- v3 data plane: every call needs `team_id`, `agent_id`, `user_id` (headers `x-tdai-*` or body), optional `task_id`, `session_id`; absent values fall into a "default" bucket (`MemoryCore/README.md:187`; `src/core/store/isolation.ts:14-90`). The store re-checks every retrieved row against the filter (`rowMatchesIsolation`, `isolation.ts:140-171`).
- Sharing between agents is explicit: an agent's chat memory is shared with the whole team by default (`memory_shared_with_team: true`) and an agent can "borrow" at most two other agents' memories, same team only (`MemoryPanel/src/panel/domain/chat-memory-governance.ts:16-26, 36-60`). The proxy merges self plus borrowed recall by score and tags each line with the agent it came from (`MemoryProxy/src/injection/injectors/tdai-l1-recall-injector.ts:1-30`).
- Assets (Skill, Wiki, CodeGraph) have visibility `private` (owner only, not even admins), `team`, `restricted` (user/role/agent ACL), `agent` (equipped to named agents) (`README.md:229-238`). Roles: system admin, team admin, member.
- Per-team custom extraction prompts exist (`MemoryCore/src/core/memory-prompt/types.ts:1-60`; API `v3-api-memorycore-doc.md:656-740`).
- Sessions bind through headers or an interactive picker; Hermes uses headers, all four required, or the proxy passes through without injecting (`agents/hermes/README.md:1-100`).

### 2.4 The Hermes integration (two ways)

1. **Provider plugin** (`MemoryCore/hermes-plugin/memory/memory_tencentdb/`): a thin Python `MemoryProvider` that starts or finds a Node sidecar on 127.0.0.1:8420 and calls its HTTP API. `prefetch` runs three calls in parallel (L1 search top 5, L3 core read, L2 scene list) and joins them as `<relevant-memories>`, `<user-core>`, `<scene-navigation>` (`__init__.py:607-720`). `sync_turn` posts the two messages to L0 on a background thread with a four-thread cap (`:720-790`). Three tools (`:231-300, 823-905`). A circuit breaker (5 failures, 60 s) and a watchdog restart the sidecar (`:48-60, 330-470`).
   Thin spots found:
   - `on_memory_write` and `on_session_end` are empty (`:911-935`); Hermes' built-in memory writes are not mirrored.
   - No write tool and no forget tool for the agent; the only agent tools search.
   - The plugin reads `agent_id`, `team_id`, `user_id` from the init kwargs (`:541-543`) but Hermes passes `agent_identity` (the profile name), and `user_id` only for gateway chats (`hermes-agent/agent/agent_init.py:1800-1830`). `agent_identity` is never read in the plugin (grep, no hit). Result: every Hermes profile that points at one gateway lands in the same `default` team and `default` agent bucket unless extra wiring is added. For Augur, Forge and Lee this would pool their memories by accident.
2. **Proxy route**: set the Hermes `base_url` to `http://<proxy>:8096/hermes/<spaceId>` and send four headers (`agents/hermes/README.md:9-40`). Zero plugin code, but all model traffic passes through the proxy and a missing header silently turns memory off (`:70-90`).

### 2.5 Strengths and thin spots

Strengths (worth knowing even if we adopt nothing):
- A clean, enforced scope key on every row plus a post-retrieve re-check.
- Write-time absolute dating of events from message timestamps; original timestamps survive import (`CHANGELOG.md` 2.0.1, fixes: imported sessions keep their original time).
- A conflict decision at write (store/skip/update/merge) with a merged timestamp trail and an integer version.
- Persona and scene index give a compact "who is this person" start, without any search.
- Team sharing with explicit borrow limits and a source label per line; an admin console with ACLs and an analytics page.
- Operations: health endpoint reporting FTS5 availability (`MemoryCore/README.md:55`), circuit breaker, supervised sidecar, one-command Docker stack (`deploy/global-images/README.md`).
- Whole knowledge side (Wiki, CodeGraph) and skills in one system.

Thin spots:
- Recall is shallow: top 5, one threshold, no rerank, no date or entity understanding, BM25-only by default (`MemoryCore/README.md:47`). Prospecta's stack is several stages deeper (section 3).
- Extraction is lossy and silent: a dropped fact leaves no trace; Spire records every decline with a reason (section 4).
- One benchmark claim (PersonaMem 48% to 76%, `README.md:271-277`) with no method, harness or data in the repo (grep: only README lines). It measures understanding of stated user traits, not dated facts, set questions or work memory.
- No unit-test files in the open repo (see 2.1); docs that disagree about version and layer meaning; Chinese-first prompts with an instruction to answer in the user's language (`l1-extraction.ts:15-20`).
- Editable memories are only on the roadmap; "the panel currently only allows viewing and deleting" (`ROADMAP.md:46-51`).
- No secret redaction (the sanitizer strips injected tags and metadata only; grep for redact/secret finds only API-trace and metadata code, `src/utils/sanitize.ts`).
- Hermes scoping gap above; three Hermes extras (explicit write, forget, session-end flush) missing.
- Heavy: a Node sidecar plus pnpm/tsx at runtime, two LLM key sets, optional Redis/ClickHouse/COS in the full stack.

### 2.6 Borrow versus adopt

- **Adopt as the shared server?** No. License is not the problem (MIT lets us run it). The problems are fit and weight: recall is weaker than Prospecta's, it has no provenance richer than a message id, it knows nothing of Spire's audience and crossing rules, and Spire's design principle is that nothing keeps a second copy of what the FactStore holds (`spire-project/docs/notes/memory-architecture-v1/slides.py`, page 10 invariant 1). A second server means a second place where private chat lines could be kept.
- **Borrow ideas and prompts:** yes, selectively (section 6). MIT requires the notice only if we copy code or prompt text.

## 3. Prospecta: what it is, and against Tencent

Hermes plugin plus library (`hermes-prospecta/README.md`, `prospecta/README.md`, `PRINCIPLES.md`):

- **Write.** `sync_turn` retains the whole turn (`"<user>: …\n\n<assistant>: …"`, source `turn:<session>:<ms>`, tag `conversation`) on a daemon thread (`__init__.py:1417-1442`). One LLM call writes "anticipated questions" (`index_text`) used for the vector; the body is kept verbatim and indexed as a second text channel. About $0.003 per turn at the configured model (`data/prospecta-gap/report.md` section 1). Tools: `prospecta_retain` (replace by `source`), `recall`, `search`, `recall_set` (map-reduce for set questions) (`__init__.py:1212-1327`).
- **Read.** `recall_synth`: an LLM turns the message into several questions; channels (dense chunk, question, optional real BM25, graph, a meta channel that reads people and date ranges from the question) are fused by weighted RRF; optional listwise rerank (Sonnet) or Jev scoring with a gate; optional reader and one hop; a synthesized answer with sources and cost in `recall_events` (`docs/recall-stages.md`, `README.md` "Configurable channels"). Two depths (`standard`, `deep`). Per-turn prefetch exists but is off by default because of the two LLM calls (`__init__.py:1444-1470`).
- **Structure.** Entities and aliases, links (`NEXT`, `PRECEDES`, `SUCCEEDS`, `TEMPORALLY_CLOSE`, `ENTITY`, `SEMANTIC`, `CAUSAL`, `RELATED_TO`) judged by Jev, all-pairs by default; Augur's memory holds about 3.8 million links at about 300 bytes each (`docs/graph-links.md`; `data/window1/augur-next-steps.md` item 4).
- **Dates.** Library reads a date or range from the question and filters on `documents.created_on`; the plugin now saves the turn's local date (default zone America/Los_Angeles) and a fill script dated 4,346 of 4,346 notes in each of the two banks on 2026-10-12 (0 before) (`README.md` "Note dates"; `data/window1/four-query-judge.md` last section). `retain` cannot set `created_at` (`data/prospecta-gap/report.md` section 1, "Ordering and timestamps").
- **Evidence it learns from itself.** Durable event tables (`retain_events`, `recall_events`, `formulate_events`, `llm_calls`, candidates per channel), a terminal UI, shadow reads and dual write for safe bank migration, a bounded connection pool (`PRINCIPLES.md` P19; `__init__.py:116-280, 370-560`).
- **Scale.** Augur old bank 4,333 notes (new 3,933 copied 8 October); Forge 72,706 notes, 97 percent imported Hindsight material (`data/window1/augur-switch-status.md`).
- **Scope.** `bank_id` only. No user, agent, team, audience (`get_config_schema`, `__init__.py:726-830`). One bank per Janus profile. The plugin ignores the init kwargs Hermes passes (grep for `agent_context`, `agent_identity`: no hit).

### 3.1 What Prospecta lacks (Tencent has it)

| Gap | Tencent | Value to us |
|---|---|---|
| A scope key beyond the bank (team, user, agent, task, session) with a re-check on read | `isolation.ts`, v3 API | **High**: needed for any shared contract |
| Edit or delete one memory through a tool or UI | `atomic/update` (version++), `atomic/delete`, panel delete (edit on roadmap) | **High**: no forget tool in the plugin (`get_tool_schemas` lists four tools, none deletes) |
| Absolute dates written into the memory at write time | extractor rule `l1-extraction.ts:49` | Medium-high: cheap (a prompt change) |
| A conflict decision at write (skip / update / merge) | `l1-dedup.ts` | Medium. A repeated turn is rejected by the database only after the LLM call has been paid (`data/prospecta-gap/report.md` section 1, "Idempotency"), and 4 to 15 percent of the turns rebuilt from the session log were row-level repeats (same report, summary); the same fact said in two phrasings is not merged |
| A short persona or profile block always present | L3 `persona.md` | Low-medium: Hermes' own `MEMORY.md`/`USER.md` already cover the always-in-context need (`prospecta/README.md` landscape section) |
| A scene index the model can navigate | L2 `scene-navigation` | Low-medium |
| Team memory types for work (task, owner, deadline, SOP) | `work_*` types | Low for Janus, relevant to Forge |
| Recall-tool usage guide and a search budget in the prompt | `MEMORY_TOOLS_GUIDE` | Medium: free to copy; Prospecta's `system_prompt_block` is five sentences (`__init__.py:1203-1210`) |
| Skill extraction | Skill module | Low now (Spire has skills through its catalog) |
| Knowledge side (Wiki, CodeGraph) | MemoryKnowledge | Out of scope for Janus memory; Spire's Library covers documents |
| Web panel, roles, SSO | MemoryPanel | Not needed; Spire is the product UI |
| Secret/metadata sanitizing at capture | `sanitize.ts` strips injected tags, metadata, timestamps | Medium: Prospecta retains raw turn text with no cleaning (`__init__.py:1424-1432`); see privacy line, section 7 |

### 3.2 What Prospecta does better than Tencent

| Prospecta | Tencent |
|---|---|
| Verbatim turns kept; recall can rescue any note through the body channel; nothing is lost to extraction | Atoms are LLM paraphrases; the original sits in L0 but is found by a weaker search |
| Question-space matching, multi-query expansion, 5 channels, rerank, hop, depth control | One query, top 5, one threshold |
| Date and person scope from the question, with a regex baseline | none |
| Entity and alias resolution; set questions by map-reduce with a citation per fact | none |
| Cost and every LLM prompt/response stored per event; shadow reads to test a change before switching | Optional usage and trace services (Langfuse, ClickHouse, OTLP, `src/core/report/`; proxy per-turn token detail), off unless configured; no per-recall audit trail in the store |
| Caller-wins overrides, no provider imports in the library, plain Postgres | Node sidecar, SQLite/VectorDB/Mongo variants |
| Measured against its own bank with a judge step (old vs new, counts only) | One unreproducible benchmark |

Honest caveat: Prospecta's own comparison (old bank vs new bank, four fixed queries) showed the new bank returned more distinct notes but not always better answers; the old bank won two of four on exact wording, and the date field was empty in both banks until 2026-10-12 (`data/window1/four-query-judge.md`). So "deeper" is design, and its payoff is under test, not proven.

## 4. Spire memory: as built and as designed

Normative owner: `spire-project/docs/spire.md` (Memory); venue mechanism: `spire-venue/docs/venue.md:5200-6540`; architecture note of 2026-10-04: `spire-project/docs/notes/memory-architecture-v1` (slides.py, 20 pages, "design only: nothing here is built" at the time). Since then most of its pieces are described as built in `venue.md` (recall door, one tool set, proposals decided by the retention judge, episodes, crossing rule, dreaming). I read `venue.md` as the current state and the note as the design.

### 4.1 Five sources, one record, two doors

(`memory-architecture-v1/slides.py` page 5; `venue.md:5200-5260`)

1. **Notes (zettels).** One thought per note with subject, subkind (`concept`, `entity_update`, `decision`; skill reserved), standing (observation / interpretation / assertion), sources, and a revision history; forget is a tombstone (`venue.md:5875-5895, 6026-6100, 6414-6476`).
2. **The Library.** Documents (repository or SharePoint) with revisions, cut with no model into passages carrying their place (page, heading, lines), at most 2,400 characters each; study by the Librarian reads prose and writes notes that point at whole documents (`venue.md:2053-2160`; `server/src/memory/library-passages.ts:1-60`).
3. **Conversation history.** Every chat line is a fact; "episodes" make lines searchable inside their audience, never copied (`venue.md:6381-6396`).
4. **Automatic retain and recall through rung.** The host's provider (`agent-binding-host/bin/host-memory.py`) speaks `rung-memory/1`, calls the venue's recall door, declines every turn (the venue retains sealed turns itself) and turns a provider `note` into a proposal (`agent-binding-host/docs/reference.md:343-375`).
5. **Deliberate work.** `recall`, `remember`, `forget`, `read_passage`, the same four for every kind of agent (`venue.md:6414-6476`).

### 4.2 Write path

One automatic owner: the venue's retention judge decides each sealed turn (and each proposal) against up to 8 current notes that share a term, under the subject's written rule; the answer is `{retain, subkind, title, text, merge_uuid, links, reason}`; a decline carries a reason too (`server/src/memory/zettel-judge.ts:115-139`; rule files `memory-rule.ts`, `librarian-offered-memory.md`). Outcome, disposition and note commit in one transaction (`venue.md:6026-6035`). A person's rule: keep what they say about themselves and how they work, never what is about someone else (`memory-rule.ts:14-17`). The judge prompt has no instruction to write absolute dates (no hit for date/absolute/today in `zettel-judge.ts`).

### 4.3 Read path

`POST /api/recall`: subjects the asker may read, narrowed by the **audience** of the conversation being spoken into; lexical match in each source (notes, passages, own proposals, episodes); deterministic fusion by turns; a re-check of every record against current access and revision; optional Jev 1.13 reordering of the best 16 (0 to 3 score), with the fused order as fallback; whole records, no default cap; each record has a citation that opens under the reader's own access (`venue.md:6253-6360`). **No embedding and no model call on the default path**; cost zero (`venue.md:6290-6296`).

### 4.4 Privacy and scope

One access function, the same for people and agents: `person:N` readable by that person and their Janus; a projectless chat private to the people in it; every other subject public in the tenant (`venue.md:5896-5925`). The **audience rule** (recall into a room draws only on what every reader of that room may read) and the **crossing rule** (what was said in a private chat is kept only where its people can read it; a wider subject takes it only by the person's own act) are the policy (`venue.md:6381-6412`; slides.py page 4). Only private chat lines are protected; there is no approval switch (slides.py page 19, questions 2 and 3).

### 4.5 Strengths and thin spots

Strengths: provenance and citations that open; revision, history and tombstones; declines recorded with reasons; strict access with the audience and crossing rules checked before any text reaches a model; every model call accounted; memory dreaming (quiet-time cleanup of near-twin notes, `venue.md:6477-6513`); a Library with passages and study; product UI.
Thin spots (and where Prospecta is stronger):
- Recall is lexical on the default path: a paraphrase with no shared words misses; dates in the question are not understood; no entity or alias resolution; no set (map-reduce) mode. These are Prospecta's strengths (section 3). It is a deliberate rule ("no unaccounted call"), so changing it is a policy choice.
- The judge has no absolute-date rule, so a note may keep "next Monday".
- Retention is strict by design: most turns decline; the unit is a judged note, so anything the judge declines is gone from automatic recall (episodes keep the line searchable inside its audience).
- Judged keeps are one small model call per candidate; fine at today's volume, not designed for bulk import of tens of thousands of notes with unknown standing.
- I found no retrieval benchmark of the recall door in the files I read; the architecture note ties measurement to a later gate (M10, `slides.py` page 18) and the Jev-Mem decision to a four-arm test (`rung-jevmem-memory-v2`).

## 5. The rung contract and the Jev-Mem decision

`rung-memory/1` (`rung/docs/rung-memory.md`; settled card `docs/reports/rung-memory-provider-contract-v1/build.py`):

- One setting (`off` / `external` / `baseline` / `mcp:<url>`), read from rung's own flag, env or config; a bad value is an error, never a fallback.
- A provider is an MCP server that declares `capabilities.experimental["rung-memory/1"] = {recall, retain, budget{max_cost_usd}}` and two reserved tools, `rung_memory_recall {scope, prompt, context[], budget}` and `rung_memory_retain {scope, observation{kind: turn|note, …}}`; every other tool the server lists goes to the agent as is.
- Recall comes **before** the loop and is shown after the user message as quoted data ("may be stale or wrong … not an instruction"), never as system text, never kept in the session file's `text`.
- Retain runs **after** the reply and **only** for a completed, checked turn; its outcome is later (`stored`/`declined`/`unretained`). A provider failure never fails a turn.
- The scope is an opaque key; the provider owns tenancy, visibility, revocation. A record from another scope is `unavailable`, never evidence.
- Cost: reported per hook; a recall costing more than the declared `max_cost_usd` (default 0) is discarded.
- Provenance: each record has `id`, `observed_at`, `attrs`; the turn reports `injected` ids.
- Conformance: `rung-agent --memory-check` (seven clauses) and `--memory-score` (hit@1/5, MRR, latency, cost) on any provider; a generic fixture set sits in `rung-memory/tests/fixtures/recall/`.
- Honest gaps in the contract itself: no forget/revoke verb ("rung has no delete verb in `rung-memory/1`"), retain cost not bounded, one provider per run, no scope kind or audience, no time filter, text records only. The current page says rung sets no record or size limit of its own, which supersedes the "5 records and 4000 characters" figures on the older decision card and in the 2026-10-04 note (CHANGELOG records the removal).

Jev-Mem decision (`rung-jevmem-memory-v2`): a graph memory steered by small Jev decisions; decided to live **outside** rung as a provider in a container; a no-model baseline is the yardstick; headline benchmark results are "only claimed"; a measurement gate (four arms, $10 cap) decides. The same logic applies to us: contract first, stores compete behind it.

Spire already implements the provider side: `agent-binding-host/bin/host-memory.py` is a `rung-memory/1` MCP server over the recall door, passing rung's own `--memory-check` (`docs/reference.md:343-375`: "`rung-memory/1` conformance … `memory-conformance` in CI").

## 6. Where Hermes stands today (why this is more than theory)

- Hermes allows **one** external memory provider per profile (`agent/memory_manager.py:1-14, 365-425`). So Prospecta, a Tencent plugin and a Spire provider cannot all run together without a composite provider.
- Augur and Forge can already reach Spire's memory through MCP tools: a prepared kit installs `mcp_spire_recall`, `mcp_spire_remember`, `mcp_spire_forget`, `mcp_spire_library_search`, `mcp_spire_read_passage` (`hermes-spire/skills/spire/SKILL.md` Prerequisites). I did not check whether it is applied to a live profile. If it is, each Janus has two memories today: Prospecta (automatic, every turn, one bank) and Spire (deliberate, judged notes). They do not know about each other.
- Hermes passes the original user message to `sync_turn`, so a recalled block is not re-retained (`run_agent.py:4215-4240`); errors there are swallowed (`except Exception: pass`).

## 7. Unification options, honestly

**Option A: one shared memory server both call.**
Two shapes: (A1) Spire's FactStore becomes the store and Prospecta data is imported as notes; (A2) Prospecta's Postgres becomes the store and Spire reads and writes through it.
- A1: loses what Prospecta does best (question-space recall, rerank, graph, set queries, verbatim kept) unless the venue grows them; 77,000-plus imported notes would enter with unknown standing; Spire's judge keeps one note per judged thought and declines the rest, the opposite of "keep every turn". Rebuild-from-facts and "replay calls no model" (`venue.md:6514-6534`) would have to survive bulk model-written indexes. Large effort; breaks invariants 1, 4, 6, 7 of the architecture note (`slides.py` page 10).
- A2: Prospecta has no access model; the audience and crossing rules, citations that open under current access, tombstones and revision history would all have to be rebuilt there. Spire's principle "tenancy before models" means a recall must be narrowed before any text is read; Prospecta narrows by bank only.
- Either way a private-chat store would sit behind a shared door. Estimate: months, high risk. Not recommended.

**Option B: one contract, two providers (stores stay separate).**
Pick the contract, make both answer to it:
- Contract = `rung-memory/1` (already written, already conformance-tested, already served by Spire) plus the four Spire tools and three small additions (section 8).
- Spire side: done for rung (`host-memory.py`); missing: a Hermes-side adapter that speaks the same contract so a Janus can use the Spire door automatically instead of only by tool.
- Prospecta side: a small MCP façade over `Memory` (or a Hermes adapter for any `rung-memory/1` provider), so rung agents can use Prospecta and a Janus profile can choose either provider with one setting.
- Hermes needs one external provider; a **composite** provider (calls both, merges with labels) is possible but doubles latency and needs a merge rule; start with a per-profile choice.
- Estimate (mine, not measured): contract page 1 day; conformance run 1 to 2 days; Prospecta façade 3 to 5 days; Hermes adapter 3 to 5 days; scope/sanitize additions 2 days; total roughly 2 to 3 weeks of small slices.
- Risk: the contract is lowest-common-denominator (text records, opaque scope); good enough for recall/retain/tools, weaker for provenance (citation only as an id) and for audience (venue-side). The contract must carry the audience and forget, or it hides the privacy line.

**Option C: stay separate and borrow ideas.**
- Cheapest, no new risk; each side improves alone. But: three divergent stores per Janus over time, two different privacy stories, no common test set, and the same borrowing questions will return. Fine as the default if B is not approved.

Recommendation: **C now, with B's first step**, and never A. Details in section 9.

### Privacy line (what the contract must carry)

- Spire's rule: only private chat lines are private (a person's own conversation with an agent, or a chat between people); `person:N` is that person's and their Janus's; everything else is the tenant's commons; there is no approval switch (`venue.md:5896-5925`; `slides.py` page 19).
- Prospecta's banks are mostly that private material: the replay counts show most turns come from the telegram channel, a chat with the person and usually private (Augur 1,147 of 1,345 turns in the gap; Forge 1,202 of 1,472; `data/prospecta-gap/report.md` section 3 totals) and Forge's bank is 97 percent imported Hindsight text. Under Spire's rule such lines may be kept **where that conversation's people can read them**, which is the person and their Janus. So a Prospecta bank must never feed a shared room automatically (audience rule) and must never be written into a project's notes except by the person's own act (crossing rule).
- Sanitize at the boundary: the contract should say what is stripped before anything is kept or sent: secrets (rung redacts them before a provider sees content, `rung-memory.md` "Limits rung enforces"), injected memory blocks (Tencent's sanitizer does this, `sanitize.ts:11-30`), framework metadata and line-leading timestamps (same file). Prospecta today keeps the raw text and sends it to the index LLM (`__init__.py:1424-1432`).

## 8. A universal memory approach: what it would be

A provider contract (working name `agent-memory/1`; it is `rung-memory/1` plus additions, not a new design):

1. **Recall before the turn.** Input: the ask (not the context blocks), the audience of the room being spoken into, a budget. Output: whole records, each with `id`, `text`, `observed_at`, `score`, `source`, a citation that opens, and `standing`. Failure is typed (`found`/`empty`/`unavailable` + reason), never an empty success.
2. **Retain after the turn.** Input: a completed turn or a note, with scope and `operation_id`. Output: `stored` / `declined` / `proposed` with a reason. A provider may decline turns (Spire does) or keep them (Prospecta does).
3. **Agent tools.** The same four everywhere: `recall`, `remember`, `forget`, `read_passage`; `remember` may become a proposal.
4. **Scope key.** One opaque key plus, separately, the audience; the key partitions, never grants (Spire's wording, `venue.md:6207-6208`). Hermes' `agent_identity` and `user_id` map to it; the Tencent plugin shows what happens when they do not (section 2.4).
5. **Provenance.** Every record says whose words it rests on and where it came from; citations open under the reader's access.
6. **Cost.** Every provider call reports calls and dollars; a budget can refuse a recall; retain cost is bounded or at least reported.
Additions rung-memory/1 lacks: a `forget` verb, an `audience` field, a `sanitize` clause (what is stripped before keeping), and optional time filters on recall.

A universal *store* is not implied: the contract hides the store.

## 9. Ranked borrowable ideas

(From Tencent unless marked.)

For Prospecta:
1. Scope key on every row and in every call, with a post-read re-check (`isolation.ts`) — needed for B. Effort: schema column plus filter; medium.
2. A `forget` and an `edit` verb with a version number (`atomic/update`, `atomic/delete`) — contract part 3. Effort: small to medium.
3. Absolute dates at write time (`l1-extraction.ts:49`) — prompt change in the index-text prompt or in a pre-pass; small. Complements the new note-date field.
4. Conflict decision at write (`l1-dedup.ts`) combined with the gap report's own fix (check the content hash **before** the LLM call): small for the hash check, medium for a merge pass.
5. A tool guide in `system_prompt_block` (what each tool is for, a per-turn search budget, `MEMORY_TOOLS_GUIDE`): free.
6. Sanitize on capture (strip injected blocks and secrets), reusing the idea of `sanitize.ts`: small.
7. A scene or topic index from the entity table (L2 idea), and a short profile block: low priority.

For Spire (from Tencent and from Prospecta):
1. Write-time absolute dates in the retention judge prompt (Tencent `l1-extraction.ts:49`): a prompt change.
2. Persona-style consolidation after N new notes or on cold start (`persona-trigger.ts`) as the next slice of memory dreaming, which today is cleanup only: medium.
3. From Prospecta: a date-and-person scope and a dense-chunk channel at the recall door for paraphrases, behind the accounted-call rule (policy decision, medium to large); and entity aliases for people names.
4. From Prospecta: set questions by map-reduce with a citation per fact; from Tencent: nothing for sets.
5. From Tencent: an explicit "borrow" limit and per-line source label when one agent reads another's notes (Spire already labels `by`).

Do not borrow: the transparent proxy (all model traffic through one process, silent off when a header is missing), LLM-extracted atoms as the only durable form, the second server.

## 10. Risks, effort, smallest first step

Risks:
- Privacy: the single biggest. A shared door over Prospecta's banks risks private lines reaching a shared room. Mitigation: audience on every recall; banks marked private-person by default; no automatic crossing.
- Scope accident: Hermes passes `agent_identity`; any new provider that reads `agent_id` instead pools profiles. Tested by one conformance clause (leak across scope).
- Contract drift: three copies of the rules (Hermes, rung, Spire) already exist. Mitigation: one page, one conformance test.
- Cost: composite recall doubles latency and cost; Spire's default path costs zero, Prospecta's recall costs about two LLM calls.
- Over-claiming: no head-to-head measure yet; do not pick a winner before the first score run.

Effort: A months; B two to three weeks in small slices; C none (borrowing items 3 to 6 above are each small).

**Smallest first step (no code):** one page, `agent-memory/1`, that maps Hermes, rung and Spire in the table of section 1, adds `forget`, `audience`, `sanitize`, and lists every gap per system. It is a document the caretaker can review in ten minutes. If the captain wants evidence next: a scout builds a ~150-line `rung-memory/1` façade over Prospecta's `Memory` in a worktree and runs rung's `--memory-check` and `--memory-score` on rung's generic fixture set (no private data) for baseline, Prospecta and (if a venue stack is available) Spire's provider. That gives the first like-for-like numbers.

## 11. Captain questions (at most six; default for each is "build nothing")

1. **Direction: one contract, not one store?** Recommend yes: pursue Option B and never A; keep C as the default. Changes: yes = the smallest-step page and later slices are planned; no = Option C only (borrow items, no shared contract).
2. **Approve the no-code first step (the contract page and gap table)?** Recommend yes. Changes: yes = one document in `spire-project` or `rung` docs (the caretaker decides the home), about a day, no code; no = nothing.
3. **Privacy line: confirm that the contract carries the audience rule, and that Prospecta banks stay private to the person and their Janus (never feed shared rooms; reach a project only by the person's own `remember`)?** Recommend yes. Changes: yes = the audience field and a per-bank "private" label become contract requirements; no = a shared recall could draw on private lines, which Spire's policy forbids.
4. **Sanitize at the boundary: require stripping secrets and injected memory blocks before any turn is kept (Prospecta first)?** Recommend yes. Changes: yes = a small change to `sync_turn` later; no = pasted keys stay in banks and go to the index model.
5. **Evidence step: may a scout (separate task) front Prospecta with the contract and score it against rung's generic test set?** Recommend yes, after question 2. Changes: yes = one scout, worktree only, no live profile touched; no = decisions stay on design reading.
6. **Cheap borrows now: copy three small ideas (absolute dates in Spire's judge prompt; a tool guide in Prospecta's prompt block; check the content hash before the LLM call)?** Recommend yes, as three separate small tasks after question 1. Changes: yes = three small changes in three repos, each its own task; no = nothing.

## 12. What I could not settle (and why)

- Whether the hermes-spire kit is applied on a live profile (kit is "prepared only"); not checked.
- The live Hermes gateway version (read a checkout dated 2026-09-02).
- No recall quality numbers for Tencent or for Spire's door; none run (read-only, no stack).
- Cost per fact stored for Tencent's pipeline (not given in the repo).
- `docs/design/prospecta/brief*.md` named in the brief does not exist.

## 13. Evidence index (commands and key refs)

```
ls / find / sed on the repos listed in section 0
find TencentDB-Agent-Memory \( -name '*.test.*' -o -name '*.spec.*' -o -name __tests__ \) -not -path '*/node_modules/*'   -> no output
grep -rn "PersonaMem" TencentDB-Agent-Memory --include=*.md                                                                 -> README.md:275,277 and README_CN.md only
grep -n "agent_identity\|agent_context" memory_tencentdb/*.py hermes-prospecta/__init__.py                                -> no output
grep -n -i "date\|absolute\|relative\|today" spire-venue/server/src/memory/zettel-judge.ts                                   -> no date rule in the prompts
git ls-tree -r --name-only <each remote branch> | grep -i brief  (prospecta)                                                -> no output
publishing build (scratch, then pdf/)                                                                                        -> see pdf/BUILD.md
```

Key line references are given inline. The three firstmate data notes were read for counts only.
