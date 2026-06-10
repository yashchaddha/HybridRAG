"""End-to-end orchestration with a structured trace.

ask(question) ->
  1. ROUTE        entity extraction + signal scoring (or LLM) -> RoutingDecision
  2. RETRIEVE     docs first (lexical + graph expansion), then graph-resolve
                  doc-native entities (bulletin -> product), then SQL via the
                  semantic layer — so each side can inform the other.
  3. MERGE        assign citation ids, synthesize one answer (LLM or extractive)
  4. VERIFY       pure-function citation check; result + warnings on the Answer

Every stage appends to `trace`, which is persisted to runs/ as JSON: that file
is the artefact that lets you *validate the routing and evidence-merging
layer* without trusting console output.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import citations, doc_tool, router, sql_tool, synthesize
from .config import settings
from .doc_tool import Retriever
from .kg import KnowledgeGraph
from .models import Answer


class Pipeline:
    def __init__(self) -> None:
        self.graph = KnowledgeGraph()
        self.retriever = Retriever()

    def ask(self, question: str, save_trace: bool = True,
            force_llm: bool = False, on_step=None) -> Answer:
        def step(title: str, detail: str = "") -> None:
            if on_step:
                on_step(title, detail)

        trace: dict = {"question": question, "force_llm": force_llm,
                       "started_at": datetime.now(timezone.utc).isoformat()}
        t0 = time.perf_counter()

        # 1. route ----------------------------------------------------------
        decision = router.route(question, self.graph, force_llm=force_llm)
        trace["routing"] = decision.model_dump()
        trace["t_route_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        step("Routing",
             f"{decision.route.upper()} (confidence {decision.confidence:.2f}, "
             f"{decision.engine}) — {decision.rationale}")
        if decision.matched_entities:
            step("Entities", ", ".join(decision.matched_entities))

        entities = set(decision.matched_entities)

        # 2a. documents -------------------------------------------------------
        doc_evs = []
        if decision.needs_pdf:
            t = time.perf_counter()
            doc_evs = doc_tool.retrieve(question, entities, self.graph,
                                        self.retriever, trace)
            trace["t_docs_ms"] = round((time.perf_counter() - t) * 1000, 1)
            titles = "; ".join(e.title for e in doc_evs[:3]) or "none"
            step("Document retrieval", f"{len(doc_evs)} chunk(s): {titles}")

        # 2b. graph hop: resolve doc-native entities to DB entities -----------
        # Bulletins surfaced by retrieval (not just the question) can trigger
        # the hop; the "is this new?" check runs against question entities
        # only, so a bulletin chunk that also mentions the product still
        # yields product:* as an expansion for the SQL stage.
        doc_entities = {e for ev in doc_evs
                        for e in self.graph.entities_in_chunk(
                            ev.metadata.get("chunk_id", ""))}
        doc_bulletins = {e for e in doc_entities
                         if self.graph.entity(e).get("type") == "bulletin"}
        expanded = self.graph.expand_entities(entities | doc_bulletins,
                                              known=entities)
        if expanded:
            trace["entity_expansion"] = expanded
            entities |= set(expanded)
            step("Knowledge-graph hop",
                 ", ".join(f"+{e} ({why})" for e, why in expanded.items()))

        # 2c. database ---------------------------------------------------------
        sql_evs = []
        if decision.needs_sql:
            t = time.perf_counter()
            sql_evs = sql_tool.retrieve(question, entities, self.graph, trace,
                                        force_llm=force_llm)
            trace["t_sql_ms"] = round((time.perf_counter() - t) * 1000, 1)
            attempts = trace.get("sql_attempts", [])
            if attempts:
                summary = " → ".join(
                    f"try {a['try']}: " + (a["error"] if a.get("error")
                                           else f"{a.get('rows', 0)} row(s)")
                    for a in attempts)
                step("SQL (LLM-written)", summary)
            elif sql_evs:
                step("SQL", f"{len(sql_evs)} result set(s)")

        # 3+4. merge & verify ---------------------------------------------------
        evidence = citations.assign_ids(sql_evs, doc_evs)
        t = time.perf_counter()
        answer = synthesize.synthesize(question, decision, evidence, trace,
                                       force_llm=force_llm)
        trace["t_synth_ms"] = round((time.perf_counter() - t) * 1000, 1)
        step("Synthesis",
             f"engine={answer.engine}, "
             + ("verified ✅" if answer.verified else "not verified ⚠️"))
        trace["engine"] = answer.engine
        trace["verified"] = answer.verified
        trace["warnings"] = answer.warnings
        trace["evidence"] = [
            {"id": ev.id, "source": ev.source_type, "locator": ev.locator,
             "score": ev.score} for ev in evidence]
        trace["t_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        if save_trace:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            path = settings.runs_dir / f"trace_{stamp}.json"
            path.write_text(json.dumps(trace, indent=2, default=str))
            trace["trace_file"] = str(path)

        answer.trace = trace
        return answer
