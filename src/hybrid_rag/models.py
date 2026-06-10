"""Typed contracts shared by every pipeline stage.

The whole POC hinges on one idea: every stage exchanges *typed* objects, and
every claim in the final answer must point back at an `Evidence` object via a
stable citation id ("S1" for SQLite, "D1" for documents). Verification is then
a pure function over (answer text, evidence list).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Route = Literal["sql", "pdf", "hybrid"]
SourceType = Literal["sqlite", "pdf"]


class RoutingDecision(BaseModel):
    route: Route
    needs_sql: bool
    needs_pdf: bool
    confidence: float = Field(ge=0.0, le=1.0)
    matched_entities: list[str] = []
    signals: dict = {}
    rationale: str = ""
    engine: Literal["heuristic", "llm"] = "heuristic"


class Evidence(BaseModel):
    """A single retrievable unit of proof.

    id          stable citation marker used inline in the answer ("S1", "D2")
    locator     machine-resolvable URI:
                  db://app.db/<tables>?<params>          (structured)
                  pdf://<file>#page=<n>                  (unstructured)
    title       human-readable one-liner for the bibliography
    content     what the synthesizer is allowed to use (row render / chunk text)
    """
    id: str = ""
    source_type: SourceType
    locator: str
    title: str
    content: str
    score: float = 0.0
    metadata: dict = {}


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Evidence]
    routing: RoutingDecision
    verified: bool = False
    warnings: list[str] = []
    engine: str = "extractive"          # "extractive" | "llm"
    trace: dict = {}

    def bibliography(self) -> list[str]:
        lines = []
        for ev in self.citations:
            if ev.source_type == "sqlite":
                sql = ev.metadata.get("sql", "").replace("\n", " ").strip()
                rows = ev.metadata.get("row_count", "?")
                ids = ev.metadata.get("row_ids", "")
                extra = f" · ids={ids}" if ids else ""
                lines.append(f"[{ev.id}] SQLite · {ev.title} · rows={rows}{extra}\n"
                             f"      SQL: {sql}")
            else:
                via = ev.metadata.get("expansion")
                via_s = f" · added via knowledge-graph ({via})" if via else ""
                lines.append(f"[{ev.id}] PDF · {ev.title}{via_s}\n"
                             f"      {ev.locator}")
        return lines
