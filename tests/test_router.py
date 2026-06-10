from hybrid_rag import router


def test_hybrid_question_routes_hybrid(graph):
    d = router.route(
        "Acme Corp has reported compressor failures on their AeroFlow X200 "
        "units. How many X200 units have they purchased since 2025, and are "
        "these failures covered under warranty?", graph)
    assert d.route == "hybrid" and d.needs_sql and d.needs_pdf
    assert "product:AF-X200" in d.matched_entities
    assert "customer:1" in d.matched_entities


def test_aggregation_question_routes_sql(graph):
    d = router.route("What was Acme Corp's total order value in 2025?", graph)
    assert d.route == "sql" and d.needs_sql and not d.needs_pdf


def test_policy_question_routes_pdf(graph):
    d = router.route(
        "What does the warranty policy say about units operated below "
        "5 degrees C?", graph)
    assert d.route == "pdf" and d.needs_pdf and not d.needs_sql


def test_bridged_entity_forces_hybrid(graph):
    # Mentions an entity that exists in BOTH the DB and the documents, with
    # signal words from both sides -> must consult both sources.
    d = router.route("Summarise orders and warranty coverage for the "
                     "AeroFlow X200", graph)
    assert d.route == "hybrid"
    assert "product:AF-X200" in d.signals.get("bridged_entities", [])
