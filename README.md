# Hybrid Retrieval POC — SQLite + PDF, one cited answer

Routes a question to a SQLite database, a PDF corpus, or both; merges the
evidence with help from a lightweight knowledge graph; and returns **one
answer where every sentence is cited** — `[S#]` resolves to executed SQL +
row ids, `[D#]` resolves to an exact PDF page. Runs fully offline; an
optional LLM upgrades routing/synthesis behind a verified-or-fallback gate.

See **ARCHITECTURE.md** for the full design, walkthroughs, and detailed steps.

## Quickstart

```bash
pip install -r requirements.txt
export PYTHONPATH=src

python -m hybrid_rag.cli ingest    # builds DB, 3 PDFs, indexes, graph
python -m hybrid_rag.cli demo      # 4 questions: hybrid / sql / pdf / doc→DB hop
python -m hybrid_rag.cli ask "Is compressor failure on the X200 covered?" --trace

python -m pytest tests/ -q         # 27 passed
```

Every run writes an audit trace to `runs/trace_*.json` (routing decision,
candidate rankings, graph hops, executed SQL, timings, verification result).

Optional LLM path: set `OPENAI_API_KEY` (model via `LLM_MODEL`, default
`gpt-4o-mini`). Anything failing the JSON contract or citation verification
falls back to the deterministic engines.
