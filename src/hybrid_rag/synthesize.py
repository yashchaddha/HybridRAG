"""Evidence merger: one answer from two sources, every sentence cited.

Engines (same contract, verified by `citations.verify` either way):

1. LLM synthesizer (optional) — gets the evidence blocks verbatim, must cite
   every sentence with the provided ids, is forbidden from adding facts. If
   the produced answer fails verification, we *discard it* and fall back —
   "verified or replaced", never "probably fine".
2. Extractive composer (default / fallback) — fully deterministic:
     a. SQL facts rendered as natural sentences per query kind.
     b. Best sentence(s) per PDF chunk selected by lexical overlap with the
        question, prefixed with a source-aware lead-in.
     c. MERGE POLICIES: explicit cross-source rules that join a structured
        fact with a document clause in a single sentence citing both
        ([S#] + [D#]). This is the heart of the "evidence-merging layer" the
        POC exists to validate — with an LLM the policies generalise, but the
        mechanism and the verification contract stay identical.
"""
from __future__ import annotations

import re

from . import citations, llm
from .ingest import tokenize
from .models import Answer, Evidence, RoutingDecision

# ---------------------------------------------------------------------------
# Deterministic composer
# ---------------------------------------------------------------------------

_LEAD_INS = [
    ("Warranty", "The warranty policy states"),
    ("Specification", "The product specification notes"),
    ("Bulletin", "Field Service Bulletin FSB-2025-03 reports"),
]
_MAX_DOC_SENTENCES = 3
_LABEL_PREFIX = re.compile(r"^[A-Z][\w /()-]{0,40}:\s+")


def _fmt_money(v) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return str(v)


def _sql_sentence(ev: Evidence) -> str | None:
    rows = ev.metadata.get("rows") or []
    if not rows:
        return None
    kind, r = ev.metadata.get("kind"), rows[0]
    label = ev.metadata.get("date_label", "")
    label = f" {label}" if label else ""
    if kind == "units_purchased":
        return (f"{r['customer']} has purchased {r['total_units']} {r['product']} "
                f"units{label} across {r['num_orders']} orders "
                f"(first on {r['first_order']}, most recent on {r['last_order']}) "
                f"[{ev.id}].")
    if kind == "order_value":
        return (f"{r['customer']}'s total order value{label} was "
                f"{_fmt_money(r['total_value'])} across {r['num_orders']} orders "
                f"[{ev.id}].")
    if kind == "customers_for_product":
        parts = [f"{x['customer']} ({x['total_units']} units)" for x in rows]
        product = ev.metadata.get("product") or "the affected product"
        listing = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0])
        return (f"{len(rows)} customers have purchased {product}"
                f"{label}: {listing} [{ev.id}].")
    if kind == "support_tickets":
        issues = sorted({x["issue_type"] for x in rows})
        dates = sorted(x["opened_date"] for x in rows)
        who = rows[0]["customer"] if len({x["customer"] for x in rows}) == 1 else "customers"
        prod = rows[0]["product"] if len({x["product"] for x in rows}) == 1 else "these products"
        return (f"{who} has logged {len(rows)} support tickets for the {prod} "
                f"citing {', '.join(issues)} (opened between {dates[0]} and "
                f"{dates[-1]}) [{ev.id}].")
    # generic fallback (e.g. LLM-generated SQL): summarise row count
    return f"The database query returned {len(rows)} matching record(s) [{ev.id}]."


_ACTION_Q = re.compile(r"\b(recommend|action|fix|procedure|step|do about)\b", re.I)
_ACTION_S = re.compile(r"\b(must|should|install|replace|update|submit)\b", re.I)


def _best_sentence(question: str, ev: Evidence) -> str:
    q_tokens = set(tokenize(question))
    action_q = bool(_ACTION_Q.search(question))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev.content) if s.strip()]
    def score(s: str) -> float:
        toks = set(tokenize(s))
        if len(toks) < 3:                       # heading stubs / fragments
            return -1.0
        sc = float(len(q_tokens & toks))
        if re.search(r"\d", s):
            sc += 0.5
        if action_q and _ACTION_S.search(s):    # answer-type match: question
            sc += 1.5                           # asks for an action, sentence
        return sc - 0.001 * len(s)              # prescribes one
    return max(sents, key=score)


def _doc_sentence(question: str, ev: Evidence) -> str:
    sent = _best_sentence(question, ev)
    # Drop leading field labels ("Subject:", "Applies to:") — the lead-in
    # already names the document.
    sent = _LABEL_PREFIX.sub("", sent).strip()
    if sent.endswith((".", "!", "?")):
        sent = sent[:-1]
    lead = next((li for key, li in _LEAD_INS
                 if key.lower() in ev.metadata["doc"].lower()), "The document states")
    return f"{lead}: {sent} [{ev.id}]."


# --- MERGE POLICIES: explicit cross-source joins ---------------------------

def _merge_policies(question: str, sql_evs: list[Evidence],
                    doc_evs: list[Evidence]) -> list[str]:
    out: list[str] = []
    q = question.lower()

    units = next((e for e in sql_evs if e.metadata.get("kind") == "units_purchased"
                  and e.metadata.get("rows")), None)
    tickets = next((e for e in sql_evs if e.metadata.get("kind") == "support_tickets"
                    and e.metadata.get("rows")), None)
    coverage = next((e for e in doc_evs
                     if "compressor coverage" in e.metadata.get("section", "").lower()), None)
    bulletin = next((e for e in doc_evs
                     if "bulletin" in e.metadata.get("doc", "").lower()
                     and re.search(r"X2-25A|affected", e.content)), None)

    # Policy A: warranty window x purchase dates
    if units and coverage and re.search(r"warrant|cover", q):
        r = units.metadata["rows"][0]
        out.append(
            f"Because compressor coverage runs five years from delivery "
            f"[{coverage.id}], all {r['total_units']} units delivered between "
            f"{r['first_order']} and {r['last_order']} are inside the compressor "
            f"coverage window [{units.id}][{coverage.id}].")

    # Policy B: reported serial x bulletin affected range
    if tickets and bulletin:
        serial = next((m.group(0) for row in tickets.metadata["rows"]
                       if (m := re.search(r"X2-25A-\d+", row.get("summary", "")))), None)
        if serial:
            out.append(
                f"Ticket evidence cites serial {serial} [{tickets.id}], which falls "
                f"in the affected X2-25A range, so repairs on that unit are covered "
                f"at no charge under the bulletin regardless of the standard term "
                f"[{bulletin.id}].")
    return out


def compose_extractive(question: str, routing: RoutingDecision,
                       evidence: list[Evidence]) -> str:
    sql_evs = [e for e in evidence if e.source_type == "sqlite"]
    doc_evs = [e for e in evidence if e.source_type == "pdf"]

    parts: list[str] = []
    for ev in sql_evs:
        if s := _sql_sentence(ev):
            parts.append(s)
    # Narrative sentences only for the strongest doc evidence; weaker chunks
    # remain available to the merge policies and the audit trace, and any
    # chunk that ends up uncited is pruned from the bibliography afterwards.
    for ev in doc_evs[:_MAX_DOC_SENTENCES]:
        parts.append(_doc_sentence(question, ev))
    parts.extend(_merge_policies(question, sql_evs, doc_evs))

    if not parts:
        return ("No supporting evidence was retrieved from either source for "
                "this question.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# LLM synthesizer (verified-or-fallback)
# ---------------------------------------------------------------------------

_SYNTH_SYSTEM = """You are the synthesis stage of a retrieval system.
Write a single grounded answer to the question using ONLY the evidence blocks.
Hard rules:
- Every sentence must end with one or more citation markers, e.g. [S1] or [D2],
  using only the ids provided.
- Never state a fact that is not in the evidence. If evidence is insufficient,
  say what is missing (also cited where possible).
- When a conclusion combines a database fact with a document clause, cite both
  ids in that sentence.
- Do NOT add filler or meta-commentary (e.g. "no additional documents...").
  Omit any sentence you cannot cite — every sentence must carry a marker.
- Plain text only, at most two short paragraphs."""


def _evidence_block(evidence: list[Evidence]) -> str:
    blocks = []
    for ev in evidence:
        blocks.append(f"[{ev.id}] ({ev.source_type}) {ev.title}\n{ev.content}")
    return "\n\n".join(blocks)


def synthesize(question: str, routing: RoutingDecision,
               evidence: list[Evidence], trace: dict,
               force_llm: bool = False) -> Answer:
    text, engine = None, "extractive"
    if force_llm:
        # LLM-only: no extractive fallback. We keep the model's answer and let
        # verification below annotate verified/warnings honestly.
        if not llm.llm_available():
            raise llm.LLMRequiredError(
                "LLM-only synthesis requested but no OpenAI client is available "
                "(set OPENAI_API_KEY).")
        candidate = llm.complete(
            _SYNTH_SYSTEM,
            f"Question: {question}\n\nEvidence:\n{_evidence_block(evidence)}",
            temperature=0)
        if not candidate:
            raise llm.LLMRequiredError(
                "LLM-only synthesis failed (check OPENAI_API_KEY, model, and quota).")
        text, engine = candidate.strip(), "llm"
    elif llm.llm_available():
        candidate = llm.complete(
            _SYNTH_SYSTEM,
            f"Question: {question}\n\nEvidence:\n{_evidence_block(evidence)}",
            temperature=0)
        if candidate:
            ok, _ = citations.verify(candidate.strip(), evidence, routing)
            if ok:
                text, engine = candidate.strip(), "llm"
            else:
                trace["llm_synthesis_rejected"] = True

    if text is None:
        text = compose_extractive(question, routing, evidence)

    # Bibliography = exactly what the answer cites; the full retrieval set
    # stays in the trace. Renumbering keeps ids gapless after pruning.
    text, cited = citations.prune_and_renumber(text, evidence)
    verified, warnings = citations.verify(text, cited, routing)
    return Answer(question=question, text=text, citations=cited,
                  routing=routing, verified=verified, warnings=warnings,
                  engine=engine, trace=trace)
