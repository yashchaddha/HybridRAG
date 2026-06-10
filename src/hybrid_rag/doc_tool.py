"""Unstructured retrieval over the PDF chunk store.

Hybrid lexical retrieval: BM25 and TF-IDF cosine ranked lists are fused with
reciprocal rank fusion (RRF). A `Retriever` exposes exactly three methods —
`search`, `score`, `by_chunk_id` — so the lexical backend can be swapped for a
neural embedder + vector DB without touching the pipeline.

The knowledge graph then performs *evidence expansion*: chunks that MENTION an
entity already established by the question or by SQL results are injected as
candidates even when lexical scoring missed them. Expanded evidence is tagged
in metadata so the trace (and the bibliography) show exactly why it is there.
"""
from __future__ import annotations

import json
import pickle

from sklearn.metrics.pairwise import cosine_similarity

from .config import settings
from .ingest import BM25_PATH, CHUNKS_PATH, TFIDF_PATH, tokenize
from .kg import KnowledgeGraph
from .models import Evidence


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[dict] = json.loads(CHUNKS_PATH.read_text())
        self.by_id: dict[str, dict] = {c["chunk_id"]: c for c in self.chunks}
        self.bm25 = pickle.loads(BM25_PATH.read_bytes())
        tf = pickle.loads(TFIDF_PATH.read_bytes())
        self.vectorizer, self.matrix = tf["vectorizer"], tf["matrix"]

    # -- ranking ----------------------------------------------------------
    def _ranked_lists(self, query: str) -> tuple[list[int], list[int], dict]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        qv = self.vectorizer.transform([query])
        cos = cosine_similarity(qv, self.matrix).ravel()
        bm25_rank = sorted(range(len(self.chunks)), key=lambda i: -bm25_scores[i])
        cos_rank = sorted(range(len(self.chunks)), key=lambda i: -cos[i])
        return bm25_rank, cos_rank, {"bm25": bm25_scores, "cos": cos}

    def search(self, query: str, k: int | None = None) -> list[tuple[dict, float, dict]]:
        k = k or settings.doc_candidates
        bm25_rank, cos_rank, raw = self._ranked_lists(query)
        fused: dict[int, float] = {}
        for rank_list in (bm25_rank, cos_rank):
            for rank, idx in enumerate(rank_list):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(self.chunks[i], round(fused[i], 5),
                 {"bm25": round(float(raw["bm25"][i]), 3),
                  "cosine": round(float(raw["cos"][i]), 3)}) for i in order]

    def score(self, query: str, chunk_id: str) -> float:
        idx = next(i for i, c in enumerate(self.chunks) if c["chunk_id"] == chunk_id)
        qv = self.vectorizer.transform([query])
        return float(cosine_similarity(qv, self.matrix[idx]).ravel()[0])

    def by_chunk_id(self, chunk_id: str) -> dict | None:
        return self.by_id.get(chunk_id)


def _to_evidence(chunk: dict, score: float, extra_meta: dict) -> Evidence:
    return Evidence(
        source_type="pdf",
        locator=f"pdf://{chunk['doc']}#page={chunk['page']}",
        title=f"{chunk['doc']} · p.{chunk['page']} · §{chunk['section']}",
        content=chunk["text"],
        score=score,
        metadata={"doc": chunk["doc"], "page": chunk["page"],
                  "section": chunk["section"], "chunk_id": chunk["chunk_id"],
                  **extra_meta},
    )


def retrieve(question: str, entities: set[str], graph: KnowledgeGraph,
             retriever: Retriever, trace: dict) -> list[Evidence]:
    # 1. lexical candidates ------------------------------------------------
    hits = retriever.search(question)
    trace["doc_candidates"] = [
        {"chunk": c["chunk_id"], "rrf": s, **m} for c, s, m in hits]

    evidence: list[Evidence] = []
    seen: set[str] = set()
    per_doc: dict[str, int] = {}
    for chunk, score, meta in hits:
        if per_doc.get(chunk["doc"], 0) >= settings.per_doc_cap:
            continue
        evidence.append(_to_evidence(chunk, score, {"retriever": meta}))
        seen.add(chunk["chunk_id"])
        per_doc[chunk["doc"]] = per_doc.get(chunk["doc"], 0) + 1
        if len(evidence) >= settings.doc_top_k:
            break

    # 2. knowledge-graph expansion ------------------------------------------
    # Cosine similarity lives on a different scale than RRF scores, so
    # expansion evidence is pinned just below the weakest lexical hit:
    # it supplements the ranked list, never reorders it.
    floor = min((e.score for e in evidence), default=0.02)
    expansions = []
    linked = graph.chunks_for_entities(entities)
    candidates: list[tuple[str, str]] = [
        (cid, ent) for ent, cids in linked.items() for cid in cids
        if cid not in seen]
    scored = sorted(((retriever.score(question, cid), cid, ent)
                     for cid, ent in candidates), reverse=True)
    for j, (sim, cid, ent) in enumerate(scored[:settings.kg_expansion_max]):
        chunk = retriever.by_chunk_id(cid)
        if not chunk:
            continue
        ent_name = graph.entity(ent).get("name", ent)
        evidence.append(_to_evidence(chunk, round(floor * (0.9 - 0.1 * j), 5),
                                     {"expansion": f"mentions {ent_name}",
                                      "expansion_entity": ent,
                                      "cosine": round(sim, 3)}))
        seen.add(cid)
        expansions.append({"chunk": cid, "via": ent, "cosine": round(sim, 3)})
    trace["kg_expansion"] = expansions

    evidence.sort(key=lambda e: -e.score)
    return evidence[: settings.doc_top_k + settings.kg_expansion_max]
