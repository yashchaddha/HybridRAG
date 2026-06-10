"""Query router: decide which evidence sources a question needs.

Two engines, same output contract (`RoutingDecision`):

1. LLM classifier (when OPENAI_API_KEY is set) — strict JSON contract,
   validated; any parse failure falls through to (2).
2. Deterministic scorer — transparent, testable, zero-latency:
     - lexical signals  : aggregation/transaction vocabulary -> SQL,
                          policy/spec/procedure vocabulary   -> PDF
     - graph signals    : a question entity that exists in the DB *and* is
                          mentioned in documents pushes toward HYBRID.

The router is intentionally biased toward HYBRID on ties: retrieving one
extra source is cheap; missing one silently corrupts the answer.
"""
from __future__ import annotations

import re

from . import llm
from .config import settings
from .kg import KnowledgeGraph
from .models import RoutingDecision

SQL_HINTS = {
    "how many": 2.0, "total": 1.5, "count": 1.5, "sum": 1.5, "average": 1.5,
    "revenue": 2.0, "order value": 2.0, "spend": 1.5,
    "purchased": 1.5, "purchase": 1.0, "bought": 1.5, "order": 1.0,
    "orders": 1.5, "customers": 1.0, "which customers": 2.0, "tickets": 1.5,
    "since": 0.5, "between": 0.5, "last year": 0.5, "per month": 1.0,
}
PDF_HINTS = {
    "warranty": 2.0, "policy": 1.5, "coverage": 1.5, "covered": 1.5,
    "specification": 2.0, "spec": 1.0, "manual": 1.5, "bulletin": 2.0,
    "service bulletin": 2.5, "procedure": 1.5, "according to": 1.5,
    "say": 1.0, "says": 1.0, "state": 0.5, "recommend": 1.5,
    "recommended": 1.5, "corrective": 1.5, "exclusion": 1.5, "claim": 1.0,
    "maintenance": 1.5, "operating": 1.0, "temperature": 1.0,
    "document": 1.0, "documents": 1.0, "fix": 1.0,
}

_LLM_SYSTEM = """You route questions for a hybrid retrieval system over:
- a SQLite database: customers, products, orders, order_items, support_tickets
- PDF documents: product specifications, warranty policy, field service bulletins
Reply ONLY with JSON: {"needs_sql": bool, "needs_pdf": bool,
"confidence": float 0-1, "rationale": "<one sentence>"}"""


def _score(question: str, hints: dict[str, float]) -> tuple[float, list[str]]:
    q = question.lower()
    score, hits = 0.0, []
    for phrase, w in hints.items():
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            score += w
            hits.append(phrase)
    return score, hits


def route(question: str, graph: KnowledgeGraph,
          force_llm: bool = False) -> RoutingDecision:
    entities = sorted(graph.entities_in(question))

    sql_score, sql_hits = _score(question, SQL_HINTS)
    pdf_score, pdf_hits = _score(question, PDF_HINTS)

    bridges = []
    for ent in entities:
        in_db = graph.has_db_record(ent)
        in_docs = graph.doc_mention_count(ent) > 0
        if in_db:
            sql_score += 0.5
        if in_docs:
            pdf_score += 0.5
        if in_db and in_docs:
            bridges.append(ent)

    # Try the LLM first; fall back to thresholds (unless LLM-only is forced).
    decided = llm.complete_json(_LLM_SYSTEM, f"Question: {question}", temperature=0)
    if decided and isinstance(decided.get("needs_sql"), bool) \
            and isinstance(decided.get("needs_pdf"), bool):
        needs_sql, needs_pdf = decided["needs_sql"], decided["needs_pdf"]
        engine = "llm"
        confidence = float(decided.get("confidence", 0.8))
        rationale = str(decided.get("rationale", ""))[:300]
    elif force_llm:
        raise llm.LLMRequiredError(
            "LLM-only routing requested but the LLM call failed or returned "
            "invalid JSON (check OPENAI_API_KEY, model, and quota).")
    else:
        engine = "heuristic"
        t = settings.route_threshold
        needs_sql = sql_score >= t
        needs_pdf = pdf_score >= t
        if not (needs_sql or needs_pdf):          # nothing fired: be safe
            needs_sql = needs_pdf = True
        if bridges and (needs_sql or needs_pdf):  # bridged entity -> hybrid bias
            needs_sql = needs_pdf = True
        gap = abs(sql_score - pdf_score)
        confidence = round(min(0.95, 0.55 + 0.06 * (sql_score + pdf_score)
                               + (0.1 if bridges else 0.0) - 0.02 * gap), 2)
        rationale = (f"sql_score={sql_score:.1f} ({', '.join(sql_hits) or 'none'}); "
                     f"pdf_score={pdf_score:.1f} ({', '.join(pdf_hits) or 'none'}); "
                     f"bridged entities={bridges or 'none'}")

    route_name = "hybrid" if (needs_sql and needs_pdf) else ("sql" if needs_sql else "pdf")
    return RoutingDecision(
        route=route_name, needs_sql=needs_sql, needs_pdf=needs_pdf,
        confidence=max(0.0, min(1.0, confidence)),
        matched_entities=entities, engine=engine, rationale=rationale,
        signals={"sql_score": sql_score, "pdf_score": pdf_score,
                 "sql_hits": sql_hits, "pdf_hits": pdf_hits,
                 "bridged_entities": bridges},
    )


# ---------------------------------------------------------------------------
# Front-door triage: does this turn need the knowledge base, or is it chat?
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = """You are the front door of an assistant for a company knowledge base:
- a SQLite database: customers, products, orders, order_items, support_tickets
- PDF documents: product specifications, warranty policy, field service bulletins
Decide whether answering the user needs that knowledge base.
Reply ONLY with JSON: {"needs_retrieval": bool, "reply": "<text>"}
- Greeting / small talk / thanks / questions about you or your abilities ->
  needs_retrieval=false, and put a short, friendly reply in "reply".
- Anything about customers, orders, products, warranty, specs, bulletins, or
  company data -> needs_retrieval=true ("reply" may be empty)."""


def triage(question: str) -> dict:
    """Gate the chat: returns {'needs_retrieval': bool, 'reply': str}.

    On any LLM/parse failure, default to needs_retrieval=True so genuine
    knowledge-base questions are never dropped.
    """
    out = llm.complete_json(_TRIAGE_SYSTEM, f"User: {question}", temperature=0)
    if out and isinstance(out.get("needs_retrieval"), bool):
        return {"needs_retrieval": out["needs_retrieval"],
                "reply": str(out.get("reply", "")).strip()}
    return {"needs_retrieval": True, "reply": ""}
