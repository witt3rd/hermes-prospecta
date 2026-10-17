"""The pre-turn recall contract for the Hermes plugin (``recall.prefetch.mode: contract``).

Automatic recall is query expansion (Prospecta's ``docs/postgres-hybrid.md``, Mode 1). This module owns what
the plugin owns: the gate and the typed decision per chat message (is memory needed, and for what period of
time), the thread's own last turns to leave out, and the contract's wording of the block. The rest is the
library's single entry point, ``Memory.expand_and_recall``: its query-writer model expands the message into
structured searches (a ``query``, verbatim ``terms``, a period), every search runs Stage 1 (hybrid candidates)
and Stage 2 (cross-encoder rerank) in parallel with the same filters, and the records merge into one cited
block. No answer is synthesised. The design is the written recall contract (``contract/recall-contract-v1.md``
carries its words). Nothing here touches the database or the network on its own: the provider hands in the
library's entry point and the Jev transport, so every path runs offline in tests.

A turn is never failed or held by this module: the whole run has a deadline that the library is given the
remainder of, a late search is dropped by the library, and a hung call is ended here (``run_bounded``).
Nothing per turn is logged (the captain's ruling): the only log line is the provider's one warning when a turn
goes without memory because of an error in the plugin or the deadline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---- defaults (each one is a setting or the contract's own number) ----

MODES = ("off", "contract")
DEFAULT_MODE = "off"
# No model id has a default here: the profile's value, when it sets one, is handed to the library; the library
# resolves the rest (its bank's recall_config, then PROSPECTA_QUERY_WRITER_MODEL / PROSPECTA_DECISION_MODEL /
# PROSPECTA_RERANKER_MODEL, then its own default).
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().parent / "contract" / "recall-contract-v1.md"

DEADLINE_S = 5.0            # from the call; core's own wait is 8 s
MAX_DEADLINE_S = 7.0        # the cap on recall.prefetch.deadline_seconds (the library's own default deadline)
DECISION_TIMEOUT_S = 1.0    # no answer in 1 s -> the rule decides (the one stage limit the plugin still owns)
EXPAND_GRACE_S = 0.15       # the library keeps its own deadline; a call that has not returned this long after it is ended here
MIN_EXPAND_S = 0.1          # less than this left of the deadline: the library is not called
NEEDS_THRESHOLD = 0.5       # a noul at or above this is "yes"
PASTE_CHARS = 50_000        # over this the line is material (Jev's 60 KB request window)
MAX_LINE_CHARS = 1_000_000  # beyond any model's window: nothing reads it
SHORT_REPLY_WORDS = 3
AHEAD_TTL_S = 600.0
# What the library's writer reads as the message when the person has not replied yet (queue_prefetch): it
# expands the context the agent's own proposal (agent_last) will need.
AHEAD_PROMPT = "(the person has not replied yet; prepare for the reply)"

_TRIVIAL_FALLBACK = re.compile(r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|hi|hey|hello|"
                               r"yo|sup|continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)"
                               r"[\W_]*$", re.IGNORECASE)

LINE_KINDS = ("answer", "go_ahead", "follow_up", "new_ask", "material", "social")
TURN_SOURCE_PREFIX = "turn:"

# Stage 0's time scope: what the decision model may answer, and the window each relative scope means
# (days back from today, inclusive). "named" is a period the line itself names (a month, a year, a
# date): the library's writer dates the search that names it; the decision model cannot, so no window
# is passed for it.
TIME_SCOPES = ("none", "recent", "past_week", "past_month", "past_year", "named")
RELATIVE_DAYS = {"recent": 1, "past_week": 7, "past_month": 31, "past_year": 366}


class ContractError(Exception):
    """The contract file is missing or malformed."""


# =====================================================================  the contract file

@dataclass(frozen=True)
class Contract:
    questions: Dict[str, Any]          # the Jev request's `questions`, as written
    intro: str                         # the block's first paragraph (it replaces the library's own heading)
    lines: Dict[str, str]              # fixed sentences, with {x} marks

    def line(self, key: str, **values: Any) -> str:
        text = self.lines.get(key, "")
        for k, v in values.items():
            text = text.replace("{" + k + "}", str(v))
        return text


_SECTION = re.compile(r"^## (\S+)[ \t]*$", re.MULTILINE)
_FENCE = re.compile(r"^```[\w-]*[ \t]*\n(.*?)\n```[ \t]*$", re.MULTILINE | re.DOTALL)


def parse_contract(text: str) -> Contract:
    """The contract's three sections: ``jev-questions`` (JSON), ``block-intro`` and ``lines``. Any other
    section (an older file's ``question-writer-prompt``, which the library's writer now owns) is not read."""
    marks = list(_SECTION.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections[m.group(1)] = text[m.end():end]

    def fenced(name: str) -> str:
        body = sections.get(name)
        if body is None:
            raise ContractError(f"contract has no '## {name}' section")
        m = _FENCE.search(body)
        if not m:
            raise ContractError(f"contract section '{name}' has no fenced block")
        return m.group(1)

    try:
        questions = json.loads(fenced("jev-questions"))
    except ValueError as exc:
        raise ContractError(f"contract jev-questions is not JSON: {exc}") from exc
    if not isinstance(questions, dict):
        raise ContractError("contract jev-questions is not an object")
    for need in ("line_kind", "needs_memory"):      # the two the gate reads (time_scope is optional)
        if need not in questions:
            raise ContractError(f"contract jev-questions lacks '{need}'")
    kinds = (questions["line_kind"].get("criteria") or {})
    if not set(LINE_KINDS) <= set(kinds):
        raise ContractError("contract line_kind lacks one of " + ", ".join(LINE_KINDS))
    lines: Dict[str, str] = {}
    for row in fenced("lines").splitlines():
        key, sep, value = row.partition(":")
        if sep and key.strip():
            lines[key.strip()] = value.strip()
    return Contract(questions=questions, intro=fenced("block-intro").strip(), lines=lines)


_contract_cache: Dict[Tuple[str, float], Contract] = {}


def load_contract(path: Any = None) -> Contract:
    """The contract file at ``path`` (default: the bundled copy), cached until it changes."""
    p = Path(path) if path else DEFAULT_CONTRACT_PATH
    try:
        key = (str(p), p.stat().st_mtime)
        if key in _contract_cache:
            return _contract_cache[key]
        contract = parse_contract(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContractError(f"cannot read the recall contract {p}: {exc}") from exc
    _contract_cache.clear()
    _contract_cache[key] = contract
    return contract


# =====================================================================  settings

@dataclass(frozen=True)
class Settings:
    mode: str = DEFAULT_MODE
    decision_model: Optional[str] = None        # None: the library's (prospecta.stages.JEV_MODEL)
    query_writer_model: Optional[str] = None    # None: the library resolves it (bank config, env, default)
    contract_path: str = ""
    max_tokens: int = 0                 # safety valve on the block (the library's); 0 = off
    deadline_s: float = DEADLINE_S
    decision_timeout_s: float = DECISION_TIMEOUT_S
    paste_chars: int = PASTE_CHARS
    drop_echoes: bool = True            # exclude the thread's own last two turns from every search (excluded_sources)

    @property
    def active(self) -> bool:
        return self.mode == "contract"


# =====================================================================  transports

Transport = Callable[[dict, float], dict]


def default_transport(api_key: Optional[str] = None) -> Transport:
    """OpenRouter's System One door (typed decisions, not chat/completions) through the library's own
    transport. The key is the agent's own ``OPENROUTER_API_KEY``, read per call and never stored."""
    from prospecta.stages import openrouter_jev_transport
    return openrouter_jev_transport(api_key)


def library_decision_model() -> str:
    """The decision model when the profile sets none: the library's own resolution
    (PROSPECTA_DECISION_MODEL, else its default)."""
    from prospecta.stages import JEV_MODEL
    return JEV_MODEL


def run_bounded(fn: Callable[[], Any], timeout: float) -> Any:
    """``fn()``'s result, or ``TimeoutError`` if it has not finished in ``timeout`` seconds (the
    call keeps running on its daemon thread; its late answer is dropped)."""
    box: Dict[str, Any] = {}
    done = threading.Event()

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name="prospecta-prefetch-call").start()
    if not done.wait(max(0.0, timeout)):
        raise TimeoutError(f"no answer in {timeout:g} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _cost(resp: Any) -> float:
    try:
        v = float(((resp or {}).get("usage") or {}).get("cost") or 0.0)
        return v if math.isfinite(v) and v > 0 else 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# =====================================================================  gate

_SCAFFOLD = re.compile(r"</?memory-context>", re.IGNORECASE)

# A skill or cron turn is not a question to look up: the skill fetches what it needs itself. Every shape Hermes
# gives such a turn opens with a bracketed ``[IMPORTANT: ...]`` line, so a turn is one of them only when it BEGINS
# with the line (a message that quotes one mid-text, or only mentions a skill, is ordinary). Where Hermes builds
# each one: agent/skill_commands.py (invoke line of a slash skill, the stacked-bundle header, the preloaded-skill
# line), agent/skill_bundles.py (the bundle header), cron/scheduler.py (the cron hint that starts EVERY cron prompt,
# the skill's invoke line, the skipped-skill notice), gateway/run.py (the auto-loaded-skill line). Core's
# prefetch_all keeps only the user's instruction of a skill turn, so for a cron wake the plugin sees the cron hint
# followed by the job's prompt: the hint is the signal that survives. A slash skill typed with an instruction
# reaches prefetch as that bare instruction; only ``on_turn_start``, which gets the whole message, tells it apart.
_SKILL_TURN = re.compile(
    r'\A\s*\[IMPORTANT: (?:'
    r'(?P<skill>The user has invoked the ")'                          # skill / skill bundle / stacked skills
    r'|(?P<cron>You are running as a scheduled cron job\.)'           # the cron hint (job with or without a skill)
    r'|(?P<skipped>The following skill\(s\) were listed for this job)'  # cron: skills not found
    r'|(?P<preloaded>The user launched this CLI session with the ")'
    r'|(?P<autoloaded>The "[^"\n]*" skill is auto-loaded\.)'          # gateway topic/channel skill binding
    r')')


def skill_turn_shape(text: Any) -> Optional[str]:
    """The shape name when ``text`` is a skill or cron turn (it begins with one of Hermes' skill/cron lines),
    else None. Never raises."""
    if not isinstance(text, str):
        return None
    m = _SKILL_TURN.match(text)
    return m.lastgroup if m else None


def gate(text: Any) -> Tuple[str, str]:
    """('go', text) to go on, else ('skip', reason). Core's trivial-word filter has already run
    (it is applied again here, so a direct call is as safe as core's); this adds: empty after
    the scaffolding strip, slash commands, and text too large for anything to read."""
    if not isinstance(text, str):
        return "skip", "no text"
    line = _SCAFFOLD.sub("", text).strip()
    if not line:
        return "skip", "empty"
    if skill_turn_shape(line):
        return "skip", "skill turn"
    if line.startswith("/"):
        return "skip", "slash command"
    try:
        from agent.memory_provider import is_trivial_prompt
        trivial = is_trivial_prompt(line)
    except Exception:
        trivial = bool(_TRIVIAL_FALLBACK.match(line))
    if trivial:
        return "skip", "trivial"
    if len(line) > MAX_LINE_CHARS:
        return "skip", "too large to read"
    return "go", line


# =====================================================================  decision (Jev)

@dataclass
class Decision:
    """What the plugin reads of the decision model: the kind of line, whether memory is needed (the gate)
    and the period of time the line limits itself to."""
    line_kind: str
    needs_memory: float
    by: str = "jev"                 # jev | size | rule | ahead
    cost_usd: float = 0.0
    time_scope: str = "none"        # one of TIME_SCOPES: the period of time the line limits its question to


def build_state(*, line: str, agent_last: str, turns: int, thread: str = "", agent: str = "",
                today: str = "") -> dict:
    state = {
        "agent": agent or "an assistant of one person (a Hermes gateway)",
        "turn": "a message from the person in a chat with the agent",
        "thread": thread or "a conversation with the person",
        "agent_last": agent_last or "(none: this is the first message of the conversation)",
        "new_lines": [f"The person (to you): {line}"],
        "in_view": (f"{turns} earlier turn(s) of this conversation are in the session; the agent's last "
                    "answer is in agent_last; no notes, open work or decisions are in view"),
    }
    if today:   # what the time_scope question counts back from
        state["today"] = today
    return state


def decision_request(contract: Contract, model: str, state: dict) -> dict:
    return {"model": model, "state": state, "questions": contract.questions}


def read_decision(contract: Contract, resp: dict) -> Decision:
    """The decision from a System One answer, or ValueError if ``line_kind`` or ``needs_memory`` is not
    answered with the right type (a partial answer is no decision). The contract's other questions
    (``stands_alone``, the ``need_*``) are asked but not read: the library's writer needs none of them."""
    answers = resp.get("answers") if isinstance(resp, dict) else None
    if not isinstance(answers, dict):
        raise ValueError("response has no answers")
    for key in ("line_kind", "needs_memory"):
        if key not in answers:
            raise ValueError(f"answers lack {key}")

    a = answers["needs_memory"]
    v = a.get("noul") if isinstance(a, dict) else None
    if not _is_number(v) or not 0 <= v <= 1:
        raise ValueError("needs_memory is not a noul in 0..1")
    a = answers["line_kind"]
    kind = a.get("choice") if isinstance(a, dict) else None
    if kind not in LINE_KINDS:
        raise ValueError("line_kind is not one of the contract's kinds")
    return Decision(line_kind=kind, needs_memory=float(v), cost_usd=_cost(resp),
                    time_scope=read_time_scope(contract, answers))


def read_time_scope(contract: Contract, answers: dict) -> str:
    """The time scope the decision model chose, or "none". The question is the contract's own (a
    contract without it, or a missing or unknown answer, means no window): a search with no date
    filter loses nothing, a wrong window could, so this never voids the decision."""
    if "time_scope" not in contract.questions:
        return "none"
    a = answers.get("time_scope")
    choice = a.get("choice") if isinstance(a, dict) else None
    return choice if choice in TIME_SCOPES else "none"


def relative_window(scope: str, today: date) -> Optional[Tuple[str, str]]:
    """The notes' date window (from, to; ISO, inclusive) a relative time scope means, counted back
    from ``today``; None for "none" and for "named" (the library's writer dates that one)."""
    days = RELATIVE_DAYS.get(scope)
    if days is None:
        return None
    return (today - timedelta(days=days)).isoformat(), today.isoformat()


def rule_decision(line: str) -> Decision:
    """The contract's rule when Jev cannot decide: three words or fewer is zero; anything else is
    expanded (``Decision.by == 'rule'``; the caller adds the contract's line)."""
    short = len(line.split()) <= SHORT_REPLY_WORDS
    return Decision(line_kind="social" if short else "new_ask", needs_memory=0.0 if short else 1.0, by="rule")


def size_decision() -> Decision:
    """A paste: too long for Jev to read whole, so it is material, and it goes to the library's writer."""
    return Decision(line_kind="material", needs_memory=1.0, by="size")


# =====================================================================  helpers

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s and s.strip()]


# =====================================================================  the run

@dataclass
class Brief:
    """What one run decided and produced. ``text`` is what Hermes injects ('' = nothing).
    ``decision`` is zero | ask | fallback. Nothing in it is logged."""
    text: str = ""
    decision: str = "zero"
    line_kind: str = ""
    needs_memory: float = 0.0
    searches: List[str] = field(default_factory=list)   # the queries the library's writer wrote
    injected: int = 0               # records in the block
    cost_usd: float = 0.0
    stages_ms: Dict[str, int] = field(default_factory=dict)   # decide, expand, and the library's own (lib_*)
    reason: str = ""
    ahead_used: bool = False
    time_scope: str = "none"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    ranked: bool = True             # Stage 2 graded what was injected (False: Stage 1's own order)
    fallbacks: List[str] = field(default_factory=list)   # the library's degraded steps, in words


@dataclass
class AheadEntry:
    key: str
    text: str
    at: float


def ahead_key(agent_last: str) -> str:
    return hashlib.sha1((agent_last or "").encode("utf-8", "replace")).hexdigest()


def looks_like_proposal(agent_last: str) -> bool:
    """The agent ended on a question or an offer ("Shall I draft it?")."""
    tail = (agent_last or "").strip()[-400:]
    return "?" in (sentences(tail) or [""])[-1]


@dataclass
class Deps:
    """Everything the run needs from outside. The provider builds it; tests pass fakes."""
    expand: Callable[..., Any]                       # Memory.expand_and_recall: (prompt, kind, **arguments) -> ExpandedRecall
    transport: Optional[Transport] = None            # Jev (System One), the gate; None = unavailable
    clock: Callable[[], float] = time.monotonic
    today: Callable[[], date] = date.today           # the calendar day the time scope counts back from


class _Run:
    def __init__(self, deps: Deps, settings: Settings, contract: Contract, started: float,
                 excluded: Sequence[str] = ()) -> None:
        self.deps, self.s, self.c = deps, settings, contract
        self.deadline = started + settings.deadline_s
        self.brief = Brief()
        self.excluded: Tuple[str, ...] = tuple(excluded) if settings.drop_echoes else ()

    def left(self) -> float:
        return self.deadline - self.deps.clock()

    def budget(self, stage_s: float) -> float:
        return max(0.0, min(stage_s, self.left()))

    def timed(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = self.deps.clock()
        try:
            return fn()
        finally:
            self.brief.stages_ms[name] = int((self.deps.clock() - t0) * 1000)

    def today(self) -> date:
        try:
            return self.deps.today()
        except Exception:
            return date.today()

    # -- stages --

    def decide(self, line: str, agent_last: str, turns: int, thread: str) -> Decision:
        if len(line) > self.s.paste_chars:
            return size_decision()
        t = self.deps.transport
        try:
            if t is None:
                raise RuntimeError("no Jev transport")
            state = build_state(line=line, agent_last=agent_last, turns=turns, thread=thread,
                                today=self.today().isoformat())
            req = decision_request(self.c, self.s.decision_model or library_decision_model(), state)
            timeout = self.budget(self.s.decision_timeout_s)
            if timeout <= 0:
                raise TimeoutError("no time left")
            d = read_decision(self.c, self.timed("decide", lambda: run_bounded(lambda: t(req, timeout), timeout)))
            self.brief.cost_usd += d.cost_usd
            return d
        except Exception as exc:
            logger.debug("prospecta prefetch: no decision from Jev (%s: %s)", type(exc).__name__, exc)
            return rule_decision(line)

    def not_consulted(self, reason: str) -> str:
        self.brief.reason = reason
        return self.c.line("not-consulted", reason=reason)

    def recall(self, prompt: str, agent_last: str, d: Decision, notes: List[str]) -> str:
        """The block for ``prompt``: ONE call to the library's expansion (writer, parallel searches, merge, one
        cited block), given everything left of the deadline and the same filters for every search: the
        thread's last turns (``excluded_sources``) and the window of the time scope. Never raises: a library
        that fails, does not answer in time or cannot be read is the contract's "not consulted" line."""
        b = self.brief
        window = relative_window(d.time_scope, self.today())
        b.date_from, b.date_to = window or (None, None)
        budget = self.left()
        if budget < MIN_EXPAND_S:
            return self.not_consulted("the search did not answer in time")
        kwargs = dict(agent_last=agent_last or None, date_from=b.date_from, date_to=b.date_to,
                      excluded_sources=list(self.excluded) or None, deadline=budget,
                      max_tokens=self.s.max_tokens or None,   # 0 = not set: the library's (the bank's valve, else off)
                      query_writer_model=self.s.query_writer_model,
                      gate=False,        # decided above (the window needs the same answer): never twice
                      record=False,      # no recall logging
                      now=self.today())
        try:
            result = self.timed("expand", lambda: run_bounded(
                lambda: self.deps.expand(prompt, "chat", **kwargs), budget + EXPAND_GRACE_S))
        except TimeoutError:
            return self.not_consulted("the search did not answer in time")
        except Exception as exc:
            logger.debug("prospecta prefetch: the library's expansion failed (%s)", type(exc).__name__)
            return self.not_consulted("the search failed")
        try:
            return self.text_of(result, window, notes)
        except Exception as exc:
            logger.debug("prospecta prefetch: could not read the library's result (%s)", type(exc).__name__)
            return self.not_consulted("the search failed")

    def text_of(self, result: Any, window: Optional[Tuple[str, str]], notes: List[str]) -> str:
        b, c = self.brief, self.c
        b.fallbacks = [str(x) for x in (result.fallbacks or [])]
        b.cost_usd += _cost({"usage": {"cost": (result.accounting or {}).get("cost_usd")}})
        b.stages_ms.update({f"lib_{k}": int(v) for k, v in (result.timings_ms or {}).items() if _is_number(v)})
        b.searches = [s.query for s in result.searches]
        ok = [r for r in result.runs if r.status == "ok"]
        if result.skipped:
            b.reason = result.skipped_reason or "skipped"
            return ""
        win = c.line("window", date_from=b.date_from, date_to=b.date_to) if window else ""
        if result.items and result.block:
            b.injected = len(result.items)
            b.ranked = not any(r.fallback_reason for r in ok)
            extra = notes + ([] if b.ranked else [c.line("search-order")]) + ([win] if win else [])
            body = result.block
            from prospecta.cited import HEADING     # the contract's intro stands where the library's heading is
            if body.startswith(HEADING):
                body = body[len(HEADING):].lstrip("\n")
            return "\n\n".join([c.intro] + (["\n".join(extra)] if extra else []) + [body])
        if ok:      # the searches ran and found nothing that answers
            nothing = c.line("nothing-answers", searches=" / ".join(r.spec.query for r in ok))
            return f"{nothing} {win}" if win else nothing
        late = result.deadline_hit or any(r.status == "late" for r in result.runs)
        return self.not_consulted("the search did not answer in time" if late else "the search failed")


def run_contract(line: str, *, agent_last: str, turns: int, deps: Deps, settings: Settings,
                 contract: Contract, thread: str = "", ahead: Optional[AheadEntry] = None,
                 started: Optional[float] = None, excluded_sources: Sequence[str] = ()) -> Brief:
    """One turn: gate, decide (needs memory? which period?), expand, block. ``excluded_sources`` are the
    records of the thread's own last turns, which every search leaves out. Never raises for a stage
    failing; the caller still wraps it (``prefetch``)."""
    run = _Run(deps, settings, contract, deps.clock() if started is None else started, excluded_sources)
    b = run.brief
    verdict, line = gate(line)
    if verdict == "skip":
        b.reason = line
        return b
    d = run.decide(line, agent_last, turns, thread)
    b.line_kind, b.needs_memory, b.time_scope = d.line_kind, d.needs_memory, d.time_scope
    b.decision = "fallback" if d.by == "rule" else "zero"
    if d.line_kind == "social" or d.needs_memory < NEEDS_THRESHOLD:
        b.reason = "no need" if d.by != "rule" else "a reply of a few words"
        return b
    b.decision = "fallback" if d.by == "rule" else "ask"
    notes = [contract.line("no-decision")] if d.by == "rule" else []

    if (ahead is not None and ahead.text and d.line_kind in ("answer", "go_ahead")
            and ahead.key == ahead_key(agent_last)):
        b.text, b.ahead_used, b.reason = ahead.text, True, "prepared ahead"
        return b

    b.text = run.recall(line, agent_last, d, notes)
    return b


def run_ahead(agent_last: str, *, deps: Deps, settings: Settings, contract: Contract,
              thread: str = "", started: Optional[float] = None,
              excluded_sources: Sequence[str] = ()) -> Optional[AheadEntry]:
    """Prepare the block the agent's own proposal will need, while the person answers. Only when
    the agent ended on a proposal; no decision (nobody has replied yet). None when there is nothing."""
    if not looks_like_proposal(agent_last) or len(agent_last) > MAX_LINE_CHARS:
        return None
    run = _Run(deps, settings, contract, deps.clock() if started is None else started, excluded_sources)
    d = Decision(line_kind="go_ahead", needs_memory=1.0, by="ahead")
    text = run.recall(AHEAD_PROMPT, agent_last, d, [])
    if run.brief.reason or not run.brief.injected:   # not consulted, or nothing answers: nothing to keep
        return None
    return AheadEntry(key=ahead_key(agent_last), text=text, at=deps.clock())
