# Hybrid Retrieval POC — SQLite + PDF with Knowledge-Graph-Assisted Routing, Evidence Merging, and Verified Citations

A single-workflow proof of concept that answers questions requiring **both** a relational database and a PDF corpus: the system routes the question, retrieves from each source, joins the evidence (with help from a lightweight knowledge graph), and produces **one merged answer in which every sentence is cited** — `[S#]` markers resolve to SQL result sets, `[D#]` markers resolve to exact PDF pages. The goal of the POC is to validate the **routing and evidence-merging layer**, not a UI.

The pipeline runs fully offline and deterministically. If `ANTHROPIC_API_KEY` is set, an LLM upgrades two stages (routing classification and answer synthesis) behind a *verified-or-fallback* gate; if the LLM output fails mechanical citation verification, it is discarded and the deterministic path produces the answer instead. The architecture is therefore testable in CI without model access, while remaining LLM-ready.

---

## 1. Design goals

1. **Correct routing.** SQL-only questions must not drag in documents; policy questions must not fire SQL; questions that genuinely need both must consult both. Routing decisions carry a confidence score and a written rationale, and are recorded in a trace.
2. **Verifiable grounding, not "probably grounded".** Citation checking is a pure function over the answer text and the retrieved evidence set — no model judges whether the model cited correctly. An answer is only marked *verified* if every sentence carries a legal marker and both required source types are represented.
3. **Exact locators on both sides.** DB citations resolve to the executed SQL, parameters, and the row ids that produced the fact (`db://app.db?kind=...`, `ids=1002,1003,1005`). Document citations resolve to file + page (`pdf://AeroFlow_Warranty_Policy.pdf#page=2`), which is why chunking is page-aligned.
4. **Cross-source claims cite both sources.** The most valuable sentences in a merged answer are the ones that *join* a database fact with a document clause ("all 18 units … are inside the compressor coverage window **[S1][D1]**"). The POC makes these joins explicit, named **merge policies**, rather than hoping a model produces them.
5. **Safety by construction.** The database can only be read: a semantic layer of parameterised SQL templates is the default; optional LLM-generated SQL must pass a validation gate (single statement, `SELECT`/`WITH` only, keyword denylist, enforced `LIMIT`) and even then executes on a `mode=ro` SQLite connection.
6. **Auditability.** Every run writes a structured trace JSON (`runs/trace_*.json`) capturing the routing decision, candidate rankings, graph hops, executed SQL, synthesis engine, verification result and per-stage timings. The trace — not the console — is the artefact that proves the routing/merging layer works.

Out of scope by design: UI, authentication, streaming, multi-tenant concerns, neural embeddings (the retriever interface is built so they can be swapped in — see §12).

---

## 2. Architecture

```
                                 ┌─────────────────────────────┐
                                 │          Question           │
                                 └──────────────┬──────────────┘
                                                ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 1. QUERY ROUTER (router.py)                                   │
        │    • entity extraction via KG alias index                     │
        │    • weighted lexical signals (SQL vocab vs document vocab)   │
        │    • graph signal: "bridged" entity (in DB *and* docs) ⇒      │
        │      hybrid bias                                              │
        │    • optional LLM classifier (JSON contract) → heuristic      │
        │      fallback                                                 │
        │    Output: RoutingDecision{route, needs_sql, needs_pdf,       │
        │            confidence, rationale, signals}                    │
        └───────────────┬───────────────────────────┬───────────────────┘
                needs_pdf                       needs_sql
                        ▼                           ▼ (runs after docs)
   ┌────────────────────────────────┐   ┌────────────────────────────────┐
   │ 2a. PDF RETRIEVAL (doc_tool)   │   │ 2c. SQL RETRIEVAL (sql_tool)   │
   │  • BM25 + TF-IDF cosine        │   │  • semantic layer: param-      │
   │    fused by RRF (k=60)         │   │    eterised query templates    │
   │  • page-level chunks ⇒ exact   │   │    (units, value, customers,   │
   │    page citations              │   │    tickets) over a typed       │
   │  • per-doc cap for diversity   │   │    schema doc                  │
   │  • KG EXPANSION: inject        │   │  • optional NL→SQL via LLM,    │
   │    entity-linked chunks the    │   │    gated by validate_sql()     │
   │    lexical pass missed,        │   │  • read-only connection        │
   │    scored below lexical hits   │   │    (file:…?mode=ro)            │
   └───────────────┬────────────────┘   └───────────────┬────────────────┘
                   │      ┌──────────────────────┐      │
                   └─────▶│ 2b. KNOWLEDGE GRAPH  │◀─────┘
                          │ (kg.py, networkx)    │
                          │ entity ⟷ chunk ⟷ doc │
                          │ + PURCHASED edges    │
                          │ Graph hop: bulletin  │
                          │  →co-mention→product │
                          │  feeds SQL planning  │
                          └──────────┬───────────┘
                                     ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 3. MERGE & SYNTHESIZE (citations.py + synthesize.py)          │
        │    • assign ids: SQL→S1.., PDF→D1.. (score-ordered)           │
        │    • LLM synthesizer (optional, verified-or-fallback)         │
        │    • extractive composer (default): SQL fact sentences,       │
        │      best-sentence-per-chunk with source lead-ins, and        │
        │      MERGE POLICIES producing [S#][D#] cross-source claims    │
        │    • prune & renumber: bibliography = exactly what is cited   │
        └──────────────────────────────┬────────────────────────────────┘
                                       ▼
        ┌───────────────────────────────────────────────────────────────┐
        │ 4. VERIFY (citations.verify — pure function)                  │
        │    unknown ids? uncited sentences? required source types      │
        │    covered? ⇒ Answer{text, citations, verified, warnings}     │
        └──────────────────────────────┬────────────────────────────────┘
                                       ▼
                       Answer + bibliography + runs/trace_*.json
```

Document retrieval runs **before** SQL on purpose: chunks surfaced by the question can reveal doc-native entities (e.g. a service bulletin) that the graph then resolves to DB entities (the affected product), which in turn parameterise the SQL stage. Each side informs the other — that is the evidence-merging layer this POC exists to validate.

---

## 3. End-to-end walkthrough (demo question 1)

> *"Acme Corp has reported compressor failures on their AeroFlow X200 units. How many X200 units have they purchased since 2025, and are these failures covered under warranty? Include any relevant service bulletins."*

**Route.** The alias index matches `acme corp → customer:1` and `x200 → product:AF-X200`. SQL signals fire ("how many", "purchased"), PDF signals fire ("covered", "warranty", "service bulletins"). `product:AF-X200` is *bridged* — it has a DB record **and** document mentions — which independently forces `hybrid`. Decision: `hybrid`, confidence 0.95, with the rationale string stored in the trace.

**Documents.** BM25 and TF-IDF rankings are fused with reciprocal rank fusion. Top hits: Warranty p.2 (§Compressor Coverage), Bulletin p.1/p.2, Warranty p.3 (§Claims). The knowledge graph then injects the two X200 specification pages because they MENTION `product:AF-X200` — tagged `expansion` in metadata and pinned *below* the weakest lexical score so supplementation never reorders relevance.

**Graph hop.** Retrieved chunks mention `bulletin:FSB-2025-03`; the graph resolves bulletins to co-mentioned products. Here the product was already known from the question, so no new entity is added (the hop is the star of demo question 4 instead).

**SQL.** The planner selects two templates from the semantic layer: *units purchased* (customer × product × date window "since 2025" → 18 units across orders 1002/1003/1005, correctly excluding the 2024 order) and *support tickets* (issue keyword `compressor` → 3 rows, one citing serial `X2-25A-0142`). Both execute as parameterised read-only queries.

**Merge.** Evidence gets ids (S1, S2, D1–D7). The extractive composer writes one sentence per SQL fact, one best-sentence per top-3 document chunk, then applies merge policies: **Policy A** joins the purchase window [S1] with the five-year compressor coverage clause [D1] into a single `[S1][D1]` claim; **Policy B** joins the ticket serial [S2] with the bulletin's affected `X2-25A` range [D2]. Uncited evidence (the KG-injected spec pages, here) is pruned from the bibliography — it remains in the trace — and ids are renumbered gaplessly.

**Verify.** Every sentence carries a marker, no unknown ids, both source types represented ⇒ `verified`. The full pass — route, rankings, hop, SQL, engine, warnings, timings — lands in `runs/trace_*.json`.

---

## 4. Component deep-dives

### 4.1 Ingestion (`ingest.py`, `sample_data.py`)

`run_ingest()` builds everything from code so the POC is reproducible from a clean checkout: it (1) creates and seeds `data/app.db` (customers, products, orders, order_items, support_tickets), (2) generates three realistic PDFs with reportlab — an X200 specification, a warranty policy, and field service bulletin FSB-2025-03 — with deliberate page boundaries so page citations are meaningful, (3) parses the PDFs back with pdfplumber into **page-level chunks**, and (4) builds the lexical indexes and the knowledge graph.

Chunking detail that matters for answer quality: heading lines are detected and **removed from the display text** (otherwise they glue onto the first sentence and produce garbage like "Exclusions This warranty does not cover…"), but their terms are preserved in a separate `index_text` field that feeds BM25, TF-IDF *and* graph entity extraction. The page heading becomes the chunk's `section` label shown in the bibliography. Long pages are sentence-split at 1,400 characters with `-2` suffixed chunk ids.

### 4.2 Knowledge graph (`kg.py`)

A networkx `MultiDiGraph` persisted to JSON, small by design. Nodes: entities (customers, products, bulletins — with `db_table`/`db_id` when a DB record exists), chunks, and documents. Edges: `doc —HAS_CHUNK→ chunk`, `chunk —MENTIONS→ entity`, `customer —PURCHASED→ product` (derived from order rows). An alias index (`x200`, `acme`, `fsb-2025-03`, …) powers longest-match entity extraction from question text.

The graph is used three ways, each cheap and each visible in the trace:

1. **Routing signal.** `bridged entity` = has a DB record *and* document mentions ⇒ the question likely needs both sources, so the router biases to hybrid even if lexical signals are one-sided.
2. **Evidence expansion.** Chunks that MENTION an established entity are injected as document evidence even when lexical scoring missed them (capped, tagged `expansion`, scored below the weakest lexical hit).
3. **Cross-source joining (the graph hop).** Doc-native entities resolve to DB entities via co-mention: `bulletin:FSB-2025-03 ←MENTIONS— chunk —MENTIONS→ product:AF-X200`. The hop's "is this new?" check runs against *question* entities only, so a bulletin chunk that also names the product still yields the product as an expansion for SQL planning — that subtlety is what makes demo question 4 work.

### 4.3 Query router (`router.py`)

Deterministic core: weighted phrase dictionaries (`SQL_HINTS`: "how many", "revenue", "order value", … / `PDF_HINTS`: "warranty", "specification", "bulletin", "say", …) produce two scores; thresholds map them to `sql` / `pdf` / `hybrid`, with "nothing fired" defaulting to both (recall over precision for a POC). Bridged entities override to hybrid. Confidence combines signal strength and score gap; the rationale is a human-readable sentence stored on the decision.

If an API key is present, an LLM classifier runs first under a strict JSON contract (`{route, confidence, rationale}`); malformed output falls back to the heuristic. The `RoutingDecision` records which engine decided.

### 4.4 SQL retrieval (`sql_tool.py`)

Two planners share one execution path:

- **Semantic layer (default).** Four parameterised templates — `units_purchased`, `order_value`, `customers_for_product`, `support_tickets` — selected by intent regexes plus the resolved entities, with a date-window parser ("since 2025", "in 2025", "last year"). Templates are the production-realistic pattern: analysts vet the SQL once; the runtime only binds parameters. Each template's intent gate is deliberately narrow (e.g. `units_purchased` requires units-language, so "total order value" doesn't drag in a units row).
- **NL→SQL (optional, LLM).** Generated SQL goes through `validate_sql()`: single statement, must start `SELECT`/`WITH`, denylist (`insert|update|delete|drop|alter|create|attach|pragma|…`), `LIMIT` injected if absent. Defense in depth: execution always uses `sqlite3.connect("file:…?mode=ro", uri=True)`, so even a missed pattern cannot write.

Evidence content is a rendered result table; metadata carries the exact SQL, bound params, row count, the contributing row ids (surfaced in the bibliography as `ids=…`), and up to 10 raw rows for the synthesizer.

### 4.5 PDF retrieval (`doc_tool.py`)

`Retriever` exposes exactly `search`, `score`, `by_chunk_id` — the seam where BM25+TF-IDF can be replaced by a neural embedder + vector store without touching the pipeline. `search` fuses the two ranked lists with RRF (`k=60`), which is robust to the two scorers' different scales; a per-document cap keeps the top-k diverse. KG expansion then runs as described in §4.2. Every candidate ranking (with raw BM25 and cosine numbers) is written to the trace.

### 4.6 Synthesis (`synthesize.py`)

Both engines honour the same contract: *every sentence ends with ≥1 citation marker; cross-source claims cite both sides; no facts beyond the evidence.*

- **Extractive composer (default, deterministic).** SQL evidence renders through per-kind sentence templates ("Acme Corp has purchased 18 AeroFlow X200 units since 2025 across 3 orders … [S1]"). For the top-3 document chunks, the best sentence is chosen by lexical overlap with the question (with a digit bonus, a stub filter, and an answer-type heuristic: action-seeking questions boost imperative sentences — that is how "what corrective action…" selects the RK-114 instruction rather than the adjacent warranty paragraph). Field labels like "Subject:" are stripped and a source-aware lead-in is prefixed ("The warranty policy states: …"). Then **merge policies** add the cross-source sentences: Policy A (purchase window × coverage term) and Policy B (ticket serial × bulletin affected range). Policies are domain rules here; with an LLM they generalise — the mechanism and verification stay identical.
- **LLM synthesizer (optional).** Receives the evidence blocks verbatim plus the hard rules; its output is verified *before* acceptance. Failure ⇒ trace records `llm_synthesis_rejected` and the extractive answer ships. *Verified or replaced, never "probably fine".*

After composition, `prune_and_renumber` makes the bibliography exactly the cited set with gapless ids; the full retrieval set stays in the trace for audit.

### 4.7 Citation contract & verifier (`citations.py`)

Contract: `S#` ⇒ SQLite evidence, `D#` ⇒ PDF evidence; locator URIs `db://app.db?kind=…` and `pdf://<file>#page=<n>`; markers `[S1]`, `[D2]` at sentence ends. `verify(text, evidence, routing)` is a pure function returning `(verified, warnings)`; it fails an answer for unknown ids, any uncited sentence, or a missing required source type (S required when routing needed SQL and rows exist; D likewise). Because it is model-free, a passing answer is *mechanically* grounded.

### 4.8 Trace (`pipeline.py`)

One JSON per run: question, routing decision, doc candidate rankings, KG expansions, the entity-expansion hop, executed SQL plans/errors, synthesis engine, rejection flags, verification warnings, final evidence list, and per-stage millisecond timings. Validating the POC = reading traces, not trusting prose.

---

## 5. Production-grade practices already present

Parameterised SQL only, read-only DB connection, statement validation gate, typed Pydantic models at every boundary, deterministic fallback for every LLM touchpoint, mechanical citation verification, full per-run tracing with timings, page-exact document locators, row-id-exact DB locators, graceful degradation without network/API access, reproducible synthetic data generation, and a 27-test pytest suite covering routing, SQL safety, retrieval ranking, the graph hop, the citation contract, and all four end-to-end flows.

---

## 6. Build & run — detailed steps

Requires Python 3.11+ (developed on 3.12). No API key needed for any step.

**Step 1 — Get the code and enter the project.**
```bash
unzip hybrid-rag-poc.zip && cd hybrid-rag-poc
```

**Step 2 — Install dependencies.**
```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

**Step 3 — Make the package importable.** Either export the path for each command (used below) or install editable:
```bash
export PYTHONPATH=src          # option A
# pip install -e .             # option B, if you add a pyproject
```

**Step 4 — Ingest.** Builds the SQLite DB, generates the three PDFs, parses them into page chunks, builds BM25 + TF-IDF indexes, and constructs the knowledge graph:
```bash
python -m hybrid_rag.cli ingest
# → {"db": "...app.db", "pdfs": [...3 files...], "chunks": 8,
#    "graph_nodes": 21, "graph_edges": 23}
```

**Step 5 — Run the four-question demo** (hybrid, pure-SQL, pure-PDF, doc→DB hop):
```bash
python -m hybrid_rag.cli demo
```

**Step 6 — Ask your own question**, optionally dumping the trace inline:
```bash
python -m hybrid_rag.cli ask "Is compressor failure on the X200 covered?" --trace
python -m hybrid_rag.cli ask "..." --json        # machine-readable Answer
```

**Step 7 — Run the test suite.**
```bash
python -m pytest tests/ -q        # 27 passed
```

**Step 8 — Inspect the audit artefacts.** Each `ask`/`demo` run writes `runs/trace_<timestamp>.json`; open one to see the routing decision, candidate rankings, the SQL that executed, and stage timings.

**Step 9 (optional) — Enable the LLM path.**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export LLM_MODEL=claude-sonnet-4-6      # default
python -m hybrid_rag.cli demo
```
Routing/synthesis now try the LLM first; anything failing the JSON contract or citation verification falls back to the deterministic engines, and the trace says so.

---

## 7. Demo questions and what each one proves

| # | Question (abridged) | Route | What it validates |
|---|---------------------|-------|-------------------|
| 1 | Acme's X200 compressor failures: units since 2025, warranty coverage, relevant bulletins | hybrid | Both retrievers fire; merge policies produce `[S#][D#]` cross-source claims; date filter excludes the 2024 order (18, not 23). |
| 2 | Acme's total order value in 2025 | sql | SQL-only routing; no document noise; exact figure $29,876.00 with contributing order ids. |
| 3 | Warranty policy on operation below 5 °C | pdf | PDF-only routing despite the word "units"; exclusion clause cited to Warranty p.2. |
| 4 | Which customers bought the product affected by FSB-2025-03, and what corrective action? | hybrid | **Doc→DB graph hop**: bulletin (no DB record) resolves to `product:AF-X200`, which parameterises the customers query; corrective action cited to Bulletin p.2. |

---

## 8. Real output (from `demo_output.txt`, this build)

Question 1 — hybrid with cross-source merge sentences:

```
Route   : HYBRID  (confidence 0.95, engine heuristic)
Entities: customer:1, product:AF-X200
Bridged : product:AF-X200 (present in DB and documents)

Answer  (extractive, verified):
  Acme Corp has purchased 18 AeroFlow X200 units since 2025 across 3 orders (first on
  2025-01-15, most recent on 2026-02-10) [S1]. Acme Corp has logged 3 support tickets for
  the AeroFlow X200 citing compressor failure (opened between 2025-06-03 and 2026-01-12)
  [S2]. The warranty policy states: Compressor failures attributable to manufacturing
  defects, including start relay defects, are covered for the full five-year term [D1].
  ... Because compressor coverage runs five years from delivery [D1], all 18 units
  delivered between 2025-01-15 and 2026-02-10 are inside the compressor coverage window
  [S1][D1]. Ticket evidence cites serial X2-25A-0142 [S2], which falls in the affected
  X2-25A range, so repairs on that unit are covered at no charge under the bulletin
  regardless of the standard term [D2].

Sources:
  [S1] SQLite · orders ⋈ order_items ⋈ customers ⋈ products (unit totals) · rows=1 ·
       ids=1002,1003,1005     SQL: SELECT c.name AS customer, ...
  [D1] PDF · AeroFlow_Warranty_Policy.pdf · p.2 · §Compressor Coverage
       pdf://AeroFlow_Warranty_Policy.pdf#page=2
```

Question 4 — the doc→DB hop:

```
Route   : HYBRID  (confidence 0.95, engine heuristic)
Entities: bulletin:FSB-2025-03
KG hop  : +product:AF-X200  (co-mentioned with bulletin:FSB-2025-03)

Answer  (extractive, verified):
  3 customers have purchased AeroFlow X200: Acme Corp (23 units), Borealis Labs (3 units)
  and Cascade Foods (2 units) [S1]. ... Field Service Bulletin FSB-2025-03 reports:
  Certified technicians must install Relay Retrofit Kit RK-114 and update controller
  firmware to version 2.4.1 on all affected units [D2].
```

The full transcript for all four questions ships as `demo_output.txt`.

---

## 9. How to evaluate this layer

The trace files make three metrics straightforward to compute offline: **routing accuracy** (label a question set, compare `trace.routing.route`), **citation precision/recall** (precision = cited ids whose evidence actually supports the sentence, judged by spot-check or an LLM grader; recall = required source types covered — the verifier already enforces the floor), and **merge coverage** (fraction of genuinely cross-source questions whose answer contains ≥1 `[S#][D#]` sentence — `test_pipeline.py` asserts this for the demo set). Faithfulness checks can diff answer claims against `metadata.rows` and chunk text since both ride along in the trace.

---

## 10. Productionization roadmap

In rough order of value: swap the lexical retriever for neural embeddings behind the existing 3-method `Retriever` interface (pgvector/Qdrant), with RRF now fusing lexical + dense; replace regex SQL validation with AST validation (sqlglot) plus per-role row-level security and a query-cost budget; move the graph to a real store (or SQLite edge tables) with an entity-resolution pass replacing the hand alias map; generalise merge policies into an LLM-driven "join planner" whose outputs still pass the same pure-function verifier; add OpenTelemetry spans mirroring the trace keys; wire the trace-based metrics of §9 into CI as regression evals; expand the date/intent parsers or delegate them to the LLM planner under the same validation gate.

---

## 11. Repository layout

```
hybrid-rag-poc/
├── ARCHITECTURE.md            ← this document
├── README.md                  ← quickstart
├── requirements.txt
├── demo_output.txt            ← captured run of the 4-question demo
├── src/hybrid_rag/
│   ├── config.py              # paths + tunables (k, caps, thresholds, model)
│   ├── models.py              # RoutingDecision / Evidence / Answer (Pydantic)
│   ├── sample_data.py         # DB seed + PDF content + builders
│   ├── ingest.py              # PDF→page chunks, BM25/TF-IDF indexes
│   ├── kg.py                  # alias index, graph build, hop/expansion API
│   ├── router.py              # signals + optional LLM classifier
│   ├── sql_tool.py            # semantic layer, validation gate, ro executor
│   ├── doc_tool.py            # RRF hybrid retriever + KG expansion
│   ├── citations.py           # id assignment, prune/renumber, verifier
│   ├── synthesize.py          # extractive composer, merge policies, LLM synth
│   ├── pipeline.py            # orchestration + trace persistence
│   ├── llm.py                 # optional Anthropic client helpers
│   └── cli.py                 # ingest / ask / demo
├── tests/                     # 27 tests: routing, SQL safety, retrieval,
│   └── ...                    #   graph hop, citation contract, e2e
├── data/                      # generated: app.db, seed_pdfs/, index/
└── runs/                      # generated: trace_*.json per question
```
