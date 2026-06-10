"""Offline ingestion: PDFs -> page-aware chunks -> lexical indexes + graph.

Design choices (and why they are fine for a production-grade *POC*):

- Page-level chunking. Pages in the corpus are short and topically coherent;
  the page is also the natural citation unit ("Warranty_Policy.pdf, p.2"), so
  chunk boundaries and citation boundaries coincide and citations are exact by
  construction. Long pages are split on sentence groups but keep their page id.
- Hybrid *lexical* retrieval (BM25 + TF-IDF cosine, fused with reciprocal rank
  fusion). The retriever sits behind a 3-method interface, so swapping in a
  neural embedder (Voyage, OpenAI, sentence-transformers + a vector DB) is a
  one-file change and nothing downstream moves.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pdfplumber
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from . import kg, sample_data
from .config import settings

CHUNKS_PATH = settings.index_dir / "chunks.json"
BM25_PATH = settings.index_dir / "bm25.pkl"
TFIDF_PATH = settings.index_dir / "tfidf.pkl"

MAX_CHUNK_CHARS = 1400


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


# ---------------------------------------------------------------------------
# PDF -> chunks
# ---------------------------------------------------------------------------

def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    return (0 < len(line) <= 60 and len(line.split()) <= 7
            and not line.endswith((".", ":", ";")))


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sents = re.split(r"(?<=[.!?])\s+", text)
    parts, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > MAX_CHUNK_CHARS:
            parts.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    return parts


def parse_pdfs(pdf_dir: Path) -> list[dict]:
    chunks: list[dict] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                if not lines:
                    continue
                headings = [ln for ln in lines if _looks_like_heading(ln)]
                section = headings[0] if headings else lines[0]
                # Body = prose only. Headings stay out of the display text
                # (they otherwise glue onto the next sentence) but their terms
                # still count for retrieval and entity extraction via
                # `index_text`.
                body_lines = [ln for ln in lines if ln not in headings]
                text = re.sub(r"\s+", " ", " ".join(body_lines)).strip() or section
                heading_terms = " ".join(headings)
                for i, part in enumerate(_split_long(text)):
                    suffix = f"-{i + 1}" if i else ""
                    chunks.append({
                        "chunk_id": f"{pdf_path.stem}-p{page_no}{suffix}",
                        "doc": pdf_path.name,
                        "page": page_no,
                        "section": section,
                        "text": part,
                        "index_text": f"{heading_terms}. {part}".strip(". "),
                    })
    return chunks


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def build_indexes(chunks: list[dict]) -> None:
    # Index section heading + body (the stored body no longer repeats the
    # heading, but its terms should still count towards retrieval).
    corpus = [c.get("index_text", c["text"]) for c in chunks]
    bm25 = BM25Okapi([tokenize(t) for t in corpus])
    vec = TfidfVectorizer(lowercase=True, stop_words="english",
                          ngram_range=(1, 2), sublinear_tf=True)
    matrix = vec.fit_transform(corpus)

    CHUNKS_PATH.write_text(json.dumps(chunks, indent=1))
    BM25_PATH.write_bytes(pickle.dumps(bm25))
    TFIDF_PATH.write_bytes(pickle.dumps({"vectorizer": vec, "matrix": matrix}))


def run_ingest(verbose: bool = True) -> dict:
    db_path = sample_data.build_database()
    pdf_paths = sample_data.build_pdfs()
    chunks = parse_pdfs(settings.pdf_dir)
    build_indexes(chunks)
    graph = kg.build_graph(db_path, chunks)
    summary = {
        "db": str(db_path),
        "pdfs": [p.name for p in pdf_paths],
        "chunks": len(chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_ingest()
