from hybrid_rag import doc_tool


def test_compressor_coverage_ranks_warranty_p2_first(pipeline):
    hits = pipeline.retriever.search(
        "Is compressor failure covered under warranty?")
    top_chunk = hits[0][0]
    assert top_chunk["doc"] == "AeroFlow_Warranty_Policy.pdf"
    assert top_chunk["page"] == 2


def test_citation_locator_is_page_exact(pipeline, graph):
    evs = doc_tool.retrieve("warranty coverage for compressor", set(), graph,
                            pipeline.retriever, trace={})
    assert all(e.locator.startswith("pdf://") and "#page=" in e.locator
               for e in evs)


def test_graph_resolves_bulletin_to_product(graph):
    added = graph.expand_entities({"bulletin:FSB-2025-03"})
    assert "product:AF-X200" in added


def test_graph_bridges_x200(graph):
    assert graph.has_db_record("product:AF-X200")
    assert graph.doc_mention_count("product:AF-X200") > 0


def test_kg_expansion_scores_below_lexical(pipeline, graph):
    trace: dict = {}
    evs = doc_tool.retrieve(
        "How many AeroFlow X200 units did Acme purchase and is the "
        "compressor covered?", {"product:AF-X200"}, graph,
        pipeline.retriever, trace)
    lexical = [e.score for e in evs if "expansion" not in e.metadata]
    expanded = [e.score for e in evs if "expansion" in e.metadata]
    if expanded:  # expansion only fires when lexical retrieval missed chunks
        assert max(expanded) < min(lexical)
