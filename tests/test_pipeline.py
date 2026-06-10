import re

from hybrid_rag import citations
from hybrid_rag.models import Evidence, RoutingDecision

HYBRID_Q = ("Acme Corp has reported compressor failures on their AeroFlow "
            "X200 units. How many X200 units have they purchased since 2025, "
            "and are these failures covered under warranty? Include any "
            "relevant service bulletins.")


def test_hybrid_question_end_to_end(pipeline):
    a = pipeline.ask(HYBRID_Q, save_trace=False)
    assert a.routing.route == "hybrid"
    assert a.verified, a.warnings
    used = set(re.findall(r"\[([SD]\d+)\]", a.text))
    assert any(m.startswith("S") for m in used), "no DB citation"
    assert any(m.startswith("D") for m in used), "no document citation"
    # the merged claim: at least one sentence cites both source types
    sentences = re.split(r"(?<=[.!?])\s+", a.text)
    assert any(re.search(r"\[S\d+\]", s) and re.search(r"\[D\d+\]", s)
               for s in sentences), "no cross-source merged sentence"
    # correct figure from the seeded data (5 units from 2024 excluded)
    assert "18" in a.text


def test_doc_to_db_hop_end_to_end(pipeline):
    a = pipeline.ask(
        "Which customers purchased the product affected by service bulletin "
        "FSB-2025-03, and what corrective action does it recommend?",
        save_trace=False)
    assert a.verified, a.warnings
    assert a.trace.get("entity_expansion"), "graph hop did not fire"
    assert "Acme Corp" in a.text
    assert "RK-114" in a.text


def test_pure_sql_question(pipeline):
    a = pipeline.ask("What was Acme Corp's total order value in 2025?",
                     save_trace=False)
    assert a.verified, a.warnings
    assert "$29,876.00" in a.text
    assert not re.search(r"\[D\d+\]", a.text)


def test_pure_pdf_question(pipeline):
    a = pipeline.ask("What does the warranty policy say about units operated "
                     "below 5 degrees C?", save_trace=False)
    assert a.verified, a.warnings
    assert re.search(r"\[D\d+\]", a.text)
    assert not re.search(r"\[S\d+\]", a.text)


def test_bibliography_lists_only_cited_evidence(pipeline):
    a = pipeline.ask(HYBRID_Q, save_trace=False)
    used = set(re.findall(r"\[([SD]\d+)\]", a.text))
    assert {ev.id for ev in a.citations} == used


def _ev(id_, source):
    return Evidence(id=id_, source_type=source, locator="x", title="t",
                    content="c", score=1.0,
                    metadata={"row_count": 1} if source == "sqlite" else {})


def test_verifier_rejects_unknown_and_uncited():
    routing = RoutingDecision(route="hybrid", needs_sql=True, needs_pdf=True,
                              confidence=1.0, rationale="t")
    evs = [_ev("S1", "sqlite"), _ev("D1", "pdf")]
    ok, _ = citations.verify("Fact one [S1]. Fact two [D1].", evs, routing)
    assert ok
    ok, warnings = citations.verify("Fact one [S9].", evs, routing)
    assert not ok and warnings
    ok, _ = citations.verify("Uncited claim. Cited claim [S1][D1].", evs, routing)
    assert not ok


def test_prune_and_renumber_keeps_ids_gapless():
    evs = [_ev("S1", "sqlite"), _ev("S2", "sqlite"),
           _ev("D1", "pdf"), _ev("D2", "pdf"), _ev("D3", "pdf")]
    text = "A [S2]. B [D3]. C [S2][D3]."
    new_text, kept = citations.prune_and_renumber(text, evs)
    assert [e.id for e in kept] == ["S1", "D1"]
    assert "[S1]" in new_text and "[D1]" in new_text
    assert "S2" not in new_text and "D3" not in new_text
