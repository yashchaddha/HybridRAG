"""Citation contract + post-hoc verification.

Contract:
- SQLite evidence gets ids  S1, S2, ...   (ordered by score)
- PDF evidence gets ids     D1, D2, ...
- Every sentence in the answer must end with >=1 marker `[S#]` / `[D#]`.
- A marker is only legal if it points at an Evidence object actually retrieved
  in this run. Verification is a pure function — no model in the loop — so a
  passing answer is *mechanically* grounded, not "probably grounded".
"""
from __future__ import annotations

import re

from .models import Evidence, RoutingDecision

MARKER_RE = re.compile(r"\[([SD]\d+)\]")


def assign_ids(sql_evidence: list[Evidence], doc_evidence: list[Evidence]) -> list[Evidence]:
    sql_sorted = sorted(sql_evidence, key=lambda e: -e.score)
    doc_sorted = sorted(doc_evidence, key=lambda e: -e.score)
    for i, ev in enumerate(sql_sorted, 1):
        ev.id = f"S{i}"
    for i, ev in enumerate(doc_sorted, 1):
        ev.id = f"D{i}"
    return sql_sorted + doc_sorted


def prune_and_renumber(text: str, evidence: list[Evidence]
                       ) -> tuple[str, list[Evidence]]:
    """Keep only cited evidence and renumber ids gaplessly (S1.., D1..).

    The full retrieval set stays in the run trace for audit; the *bibliography*
    shows exactly what the answer stands on. Markers in the text are rewritten
    to match.
    """
    seen: list[str] = []
    for m in MARKER_RE.findall(text):
        if m not in seen:
            seen.append(m)
    by_id = {ev.id: ev for ev in evidence}
    mapping: dict[str, str] = {}
    counters = {"S": 0, "D": 0}
    for old in seen:
        if old in by_id:
            prefix = old[0]
            counters[prefix] += 1
            mapping[old] = f"{prefix}{counters[prefix]}"

    new_text = MARKER_RE.sub(
        lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", text)
    kept: list[Evidence] = []
    for old in seen:
        if old in by_id:
            ev = by_id[old]
            ev.id = mapping[old]
            kept.append(ev)
    kept.sort(key=lambda e: (e.id[0] != "S", int(e.id[1:])))
    return new_text, kept


def verify(answer_text: str, evidence: list[Evidence],
           routing: RoutingDecision) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    known = {ev.id for ev in evidence}
    used = set(MARKER_RE.findall(answer_text))

    unknown = used - known
    if unknown:
        warnings.append(f"answer cites unknown evidence ids: {sorted(unknown)}")

    has_sql_ev = any(ev.source_type == "sqlite" and ev.metadata.get("row_count", 0) > 0
                     for ev in evidence)
    has_doc_ev = any(ev.source_type == "pdf" for ev in evidence)
    if routing.needs_sql and has_sql_ev and not any(m.startswith("S") for m in used):
        warnings.append("routing required SQL but no [S#] citation appears")
    if routing.needs_pdf and has_doc_ev and not any(m.startswith("D") for m in used):
        warnings.append("routing required PDF but no [D#] citation appears")

    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if s]
    uncited = [s for s in sentences if not MARKER_RE.search(s)]
    if uncited:
        warnings.append(f"{len(uncited)} sentence(s) carry no citation")

    unused = sorted(known - used)
    if unused:
        warnings.append(f"retrieved but uncited (kept for audit): {unused}")

    fatal = bool(unknown) \
        or (routing.needs_sql and has_sql_ev and not any(m.startswith("S") for m in used)) \
        or (routing.needs_pdf and has_doc_ev and not any(m.startswith("D") for m in used)) \
        or bool(uncited)
    return (not fatal), warnings
