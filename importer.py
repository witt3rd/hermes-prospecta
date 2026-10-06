"""Import foreign memories (Hindsight bank dump) into a prospecta bank.

Dump format (JSON object; every key optional except as noted) -- see README
"Importing a Hindsight bank dump":

    {
      "bank_id": "janus",
      "memory_units": [{"id", "text" (required), "fact_type", "context",
                        "event_date", "occurred_start", "occurred_end",
                        "mentioned_at", "created_at", "document_id",
                        "chunk_id", "tags", "entities", "metadata",
                        "proof_count", "source_memory_ids", ...}],
      "entities": [{"id", "canonical_name", "entity_type", "aliases",
                    "first_seen", "last_seen", "metadata", ...}],
      "links": [{"id", "from_id", "to_id", "link_type", "weight",
                 "entity_id", "created_at", ...}]
    }

Mapping (prospecta has no entity/link tables, so nothing is dropped; it is
carried in ``documents.document_metadata['hindsight']``):

* each memory unit (fact / observation / ...) -> one document, source
  ``hindsight://<bank>/unit/<id>``, tags ``hindsight``, ``fact_type:<t>``,
  ``entity:<name>`` + original tags, created_at = original timestamp.
* each entity -> one document, source ``hindsight://<bank>/entity/<id>``.
* each link -> recorded on BOTH endpoint documents (out/in).
* any unrecognised field -> ``hindsight.extra``.
* identical text under a different id -> merged into the first document as
  ``hindsight.import_aliases`` (prospecta dedups on content hash).

Idempotent: a re-run leaves unchanged documents untouched (no re-embed).
Returns a no-loss report; ``report["lost"]`` must be empty.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Callable

UNIT_KNOWN = {
    "id", "text", "fact_type", "context", "event_date", "occurred_start",
    "occurred_end", "mentioned_at", "created_at", "document_id", "chunk_id",
    "tags", "entities", "metadata", "proof_count", "source_memory_ids",
}
ENTITY_KNOWN = {
    "id", "canonical_name", "entity_type", "aliases", "first_seen",
    "last_seen", "metadata",
}
LINK_KNOWN = {"id", "from_id", "to_id", "link_type", "weight", "entity_id", "created_at"}


def _src(bank: str, kind: str, ident: str) -> str:
    return f"hindsight://{bank}/{kind}/{ident}"


def _ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _link_key(l: dict) -> str:
    if l.get("id"):
        return str(l["id"])
    raw = "|".join(str(l.get(k, "")) for k in ("from_id", "to_id", "link_type", "entity_id"))
    return "h:" + hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_dump(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        dump = json.load(f)
    if not isinstance(dump, dict):
        raise ValueError("dump must be a JSON object")
    return dump


def _plan(dump: dict, bank: str) -> tuple[list[dict], list[str]]:
    """Build the ordered list of items to import and a list of problems."""
    problems: list[str] = []
    ids = {str(u.get("id")) for u in dump.get("memory_units", []) if u.get("id") is not None}
    ids |= {str(e.get("id")) for e in dump.get("entities", []) if e.get("id") is not None}
    out_links: dict[str, list] = {}
    in_links: dict[str, list] = {}
    for l in dump.get("links", []):
        k = _link_key(l)
        rec = {k_: v for k_, v in l.items()}
        rec["key"] = k
        f, t = str(l.get("from_id")), str(l.get("to_id"))
        if f not in ids and t not in ids:
            problems.append(f"link {k}: neither endpoint ({f}, {t}) is in the dump")
            continue
        out_links.setdefault(f, []).append(dict(rec, direction="out"))
        in_links.setdefault(t, []).append(dict(rec, direction="in"))

    items: list[dict] = []
    for u in dump.get("memory_units", []):
        ident = str(u.get("id"))
        text = u.get("text")
        if u.get("id") is None or not isinstance(text, str) or not text.strip():
            problems.append(f"unit {ident}: missing id or empty text")
            continue
        ents = [e if isinstance(e, str) else (e.get("name") or e.get("canonical_name") or str(e.get("id")))
                for e in u.get("entities", [])]
        tags = ["hindsight"]
        if u.get("fact_type"):
            tags.append(f"fact_type:{u['fact_type']}")
        tags += [f"entity:{e}" for e in ents]
        tags += [str(t) for t in u.get("tags", [])]
        h = {
            "kind": "unit", "id": ident, "bank_id": bank,
            "fact_type": u.get("fact_type"), "context": u.get("context"),
            "timestamps": {k: u.get(k) for k in (
                "event_date", "occurred_start", "occurred_end", "mentioned_at", "created_at")
                if u.get(k) is not None},
            "provenance": {k: u.get(k) for k in (
                "document_id", "chunk_id", "source_memory_ids", "proof_count")
                if u.get(k) is not None},
            "entities": ents, "metadata": u.get("metadata") or {},
            "links": out_links.get(ident, []) + in_links.get(ident, []),
            "extra": {k: v for k, v in u.items() if k not in UNIT_KNOWN},
        }
        items.append({"kind": "unit", "id": ident, "text": text, "tags": tags,
                      "h": h, "created": u.get("created_at") or u.get("mentioned_at") or u.get("event_date")})
    for e in dump.get("entities", []):
        ident = str(e.get("id"))
        name = e.get("canonical_name")
        if e.get("id") is None or not name:
            problems.append(f"entity {ident}: missing id or canonical_name")
            continue
        text = f"Entity: {name}" + (f" ({e['entity_type']})" if e.get("entity_type") else "")
        if e.get("aliases"):
            text += ". Also known as: " + ", ".join(map(str, e["aliases"]))
        h = {
            "kind": "entity", "id": ident, "bank_id": bank,
            "entity_type": e.get("entity_type"), "canonical_name": name,
            "aliases": e.get("aliases") or [],
            "timestamps": {k: e.get(k) for k in ("first_seen", "last_seen") if e.get(k) is not None},
            "metadata": e.get("metadata") or {},
            "links": out_links.get(ident, []) + in_links.get(ident, []),
            "extra": {k: v for k, v in e.items() if k not in ENTITY_KNOWN},
        }
        items.append({"kind": "entity", "id": ident, "text": text,
                      "tags": ["hindsight", "hindsight:entity", f"entity:{name}"],
                      "h": h, "created": e.get("first_seen")})
    return items, problems


def _strip_aliases(h: dict) -> dict:
    return {k: v for k, v in h.items() if k != "import_aliases"}


def import_dump(memory, dump: dict, *, bank_id: str | None = None,
                dry_run: bool = False) -> dict:
    """Import ``dump`` into ``memory``'s bank. Returns the no-loss report."""
    bank = bank_id or memory.default_bank_id
    src_bank = str(dump.get("bank_id") or bank)
    items, problems = _plan(dump, src_bank)
    counts = {"units": len(dump.get("memory_units", [])),
              "entities": len(dump.get("entities", [])),
              "links": len(dump.get("links", []))}
    res = {"imported": 0, "unchanged": 0, "merged_duplicates": 0, "failed": 0}
    failures: list[str] = []

    for it in items:
        source = _src(src_bank, it["kind"], it["id"])
        it["source"] = source
        text, h = it["text"], it["h"]
        chash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            with memory._pool.connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT id, source, original_text, document_metadata, tags FROM documents "
                            "WHERE bank_id=%s AND content_hash=%s", (bank, chash))
                row = cur.fetchone()
            aliases: list = []
            if row:
                existing_h = (row[3] or {}).get("hindsight", {})
                aliases = list(existing_h.get("import_aliases", []))
                if row[1] == source:
                    if _strip_aliases(existing_h) == _strip_aliases(h) and list(row[4]) == it["tags"]:
                        res["unchanged"] += 1
                        continue
                else:
                    alias = {"source": source, "id": it["id"], "kind": it["kind"],
                             "hindsight": h}
                    if alias in aliases:
                        res["unchanged"] += 1
                        continue
                    res["merged_duplicates"] += 1
                    if not dry_run:
                        aliases.append(alias)
                        new_meta = dict(row[3] or {})
                        new_meta["hindsight"] = dict(existing_h, import_aliases=aliases)
                        with memory._pool.connection() as conn, conn.cursor() as cur:
                            cur.execute("UPDATE documents SET document_metadata=%s::jsonb WHERE id=%s",
                                        (json.dumps(new_meta), row[0]))
                            conn.commit()
                    continue
            if dry_run:
                res["imported"] += 1
                continue
            hh = dict(h, import_aliases=aliases) if aliases else h
            doc_id = memory.retain(text, source=source, index_text=[text], tags=it["tags"],
                                   metadata={"hindsight": hh})
            created = _ts(it["created"])
            if created:
                with memory._pool.connection() as conn, conn.cursor() as cur:
                    cur.execute("UPDATE documents SET created_at=%s WHERE id=%s", (created, doc_id))
                    cur.execute("UPDATE memory_items SET created_at=%s WHERE document_id=%s",
                                (created, doc_id))
                    conn.commit()
            res["imported"] += 1
        except Exception as e:  # report, never silently drop
            res["failed"] += 1
            failures.append(f"{it['kind']} {it['id']}: {type(e).__name__}: {e}")

    lost = list(problems) + failures
    verified = {"units": 0, "entities": 0, "links": 0}
    if not dry_run:
        found_links, found_src = set(), set()
        with memory._pool.connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT source, document_metadata FROM documents "
                        "WHERE bank_id=%s AND source LIKE %s", (bank, f"hindsight://{src_bank}/%"))
            for source, meta in cur.fetchall():
                h = (meta or {}).get("hindsight", {})
                found_src.add(source)
                for a in h.get("import_aliases", []):
                    found_src.add(a["source"])
                    found_links.update(l["key"] for l in a["hindsight"].get("links", []))
                found_links.update(l["key"] for l in h.get("links", []))
        for it in items:
            if it["source"] in found_src:
                verified["units" if it["kind"] == "unit" else "entities"] += 1
            elif f"{it['kind']} {it['id']}" not in " ".join(failures):
                lost.append(f"{it['kind']} {it['id']}: not found after import")
        for l in dump.get("links", []):
            if _link_key(l) in found_links:
                verified["links"] += 1
            elif not any(f"link {_link_key(l)}:" in p for p in problems):
                lost.append(f"link {_link_key(l)}: not found after import")
        # links whose problem was reported above are in lost already
    return {"bank_id": bank, "source_bank_id": src_bank, "dry_run": dry_run,
            "input": counts, **res, "verified_present": verified,
            "lost": lost, "ok": not lost}


def format_report(r: dict) -> str:
    lines = [f"prospecta import report (bank={r['bank_id']}, dry_run={r['dry_run']})",
             f"  input     : {r['input']}",
             f"  imported  : {r['imported']}   unchanged: {r['unchanged']}   "
             f"merged_duplicates: {r['merged_duplicates']}   failed: {r['failed']}",
             f"  verified  : {r['verified_present']}"]
    if r["lost"]:
        lines.append(f"  LOST/UNIMPORTED ({len(r['lost'])}):")
        lines += [f"    - {x}" for x in r["lost"]]
        lines.append("  RESULT: LOSS DETECTED")
    else:
        lines.append("  RESULT: no loss")
    return "\n".join(lines)
