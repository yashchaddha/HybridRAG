"""Simple multi-view UI for the Hybrid Retrieval POC.

Sidebar switches the main area between three views:

  💬 Chat               ask a question, get one grounded, cited answer
  📄 Documents Ingested  the parsed/indexed PDF corpus + status + upload
  🕸 Knowledge Graph     the live networkx graph (nodes / edges / relations)

Note: retrieval is lexical (BM25 + TF-IDF) — there is no vector DB. "Ingested"
means a PDF has been parsed into page-chunks and added to the indexes + graph.
Ingestion is a full rebuild (`run_ingest`), so uploading a PDF saves it into
the corpus and re-ingests everything.

Run from the project root with the project's venv:

    .venv/bin/streamlit run app.py        # -> http://localhost:8501
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Make `src/` importable without exporting PYTHONPATH (mirrors tests/conftest).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

# On Streamlit Community Cloud, secrets live in st.secrets; mirror them into the
# environment so config.py / llm.py (which read os.getenv) pick them up.
try:
    for _k in ("OPENAI_API_KEY", "LLM_MODEL", "APP_PASSWORD"):
        if _k in st.secrets:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from hybrid_rag import ingest, llm, router, sample_data
from hybrid_rag.config import settings

MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-4-turbo", "Custom…"]

INDEX_FILES = ("graph.json", "aliases.json", "chunks.json", "bm25.pkl", "tfidf.pkl")
CHUNKS_PATH = settings.index_dir / "chunks.json"
SEED_PDFS = set(sample_data.PDF_DOCS)          # the three generated demo PDFs

SUGGESTIONS = [
    "How many AeroFlow X200 units has Acme Corp purchased since 2025?",
    "Is compressor failure on the X200 covered under warranty?",
    "What was Acme Corp's total order value in 2025?",
    "What corrective action does bulletin FSB-2025-03 recommend?",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def artifacts_ready() -> bool:
    return (settings.db_path.exists()
            and all((settings.index_dir / f).exists() for f in INDEX_FILES))


@st.cache_resource(show_spinner="Loading…")
def load_pipeline():
    from hybrid_rag.pipeline import Pipeline
    return Pipeline()


def chunk_counts() -> dict[str, int]:
    """doc filename -> number of indexed chunks."""
    if not CHUNKS_PATH.exists():
        return {}
    counts: dict[str, int] = {}
    for c in json.loads(CHUNKS_PATH.read_text()):
        counts[c["doc"]] = counts.get(c["doc"], 0) + 1
    return counts


def list_pdfs() -> list[Path]:
    return sorted(settings.pdf_dir.glob("*.pdf"))


def reingest() -> dict:
    """Full rebuild, then drop the cached pipeline so views reload fresh."""
    summary = ingest.run_ingest(verbose=False)
    st.cache_resource.clear()
    st.cache_data.clear()
    return summary


@st.cache_data(show_spinner=False)
def load_tables() -> dict[str, list[dict]]:
    """Read-only snapshot of every table in the SQLite database."""
    conn = sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        return {n: [dict(r) for r in conn.execute(f'SELECT * FROM "{n}" LIMIT 1000')]
                for n in names}
    finally:
        conn.close()


def graph_to_dot(g) -> str:
    """Render the networkx knowledge graph as a Graphviz DOT string."""
    ENTITY_FILL = {"product": "#cfe8ff", "customer": "#d7f5d3", "bulletin": "#ffe0b2"}
    EDGE_COLOR = {"PURCHASED": "#1a73e8", "MENTIONS": "#9aa0a6", "HAS_CHUNK": "#cdcdcd"}
    out = [
        "digraph KG {",
        '  rankdir=LR; bgcolor="transparent"; pad=0.2; nodesep=0.3; ranksep=0.7;',
        '  node [style=filled, fontname="Helvetica", fontsize=10, color="#00000022"];',
        '  edge [fontname="Helvetica", fontsize=8];',
    ]
    for node, a in g.nodes(data=True):
        kind = a.get("kind")
        if kind == "entity":
            fill = ENTITY_FILL.get(a.get("type"), "#eeeeee")
            label, shape = a.get("name", node), "box"
        elif kind == "doc":
            fill, shape, label = "#e8eaed", "note", a.get("name", node)
        else:  # chunk
            fill, shape = "#f5f5f5", "ellipse"
            label = f"p.{a.get('page', '?')} {(a.get('section') or '')[:16]}".strip()
        label = str(label).replace('"', "'")
        out.append(f'  "{node}" [label="{label}", fillcolor="{fill}", shape={shape}];')
    for u, v, d in g.edges(data=True):
        rel = d.get("rel", "")
        out.append(f'  "{u}" -> "{v}" [label="{rel}", '
                   f'color="{EDGE_COLOR.get(rel, "#9aa0a6")}"];')
    out.append("}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Citation rendering — downloadable PDF + the exact chunk used
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def pdf_bytes(name: str):
    """Raw bytes of a corpus PDF, for the download button (None if missing)."""
    p = settings.pdf_dir / name
    return p.read_bytes() if p.exists() else None


def citation_payload(answer) -> list[dict]:
    """Compact, session-storable view of the cited evidence.

    For PDF evidence we keep `chunk_text` (== Evidence.content) so the exact
    chunk the answer was based on can always be shown.
    """
    out: list[dict] = []
    for ev in answer.citations:
        if ev.source_type == "pdf":
            out.append({"id": ev.id, "source_type": "pdf",
                        "doc": ev.metadata.get("doc"),
                        "page": ev.metadata.get("page"),
                        "section": ev.metadata.get("section"),
                        "chunk_text": ev.content,
                        "expansion": ev.metadata.get("expansion")})
        else:
            out.append({"id": ev.id, "source_type": "sqlite",
                        "title": ev.title,
                        "row_count": ev.metadata.get("row_count"),
                        "row_ids": ev.metadata.get("row_ids"),
                        "sql": ev.metadata.get("sql"),
                        "rows": ev.metadata.get("rows", [])})
    return out


def render_citations(msg_idx: int, cites: list[dict]) -> None:
    for c in cites:
        if c["source_type"] == "pdf":
            head = f"**[{c['id']}]** 📄 {c['doc']} · p.{c['page']} · §{c['section']}"
            if c.get("expansion"):
                head += f"  ·  _via knowledge graph ({c['expansion']})_"
            st.markdown(head)
            data = pdf_bytes(c["doc"])
            if data is not None:
                # Clicking the PDF name downloads the file.
                st.download_button(f"⬇ {c['doc']}", data=data, file_name=c["doc"],
                                   mime="application/pdf",
                                   key=f"dl_{msg_idx}_{c['id']}")
            else:
                st.caption("(file not found on disk)")
            st.markdown("_Exact chunk the answer used:_")
            st.markdown(f"> {c['chunk_text']}")
        else:
            line = f"**[{c['id']}]** 🗄 {c['title']} · rows={c['row_count']}"
            if c.get("row_ids"):
                line += f" · ids={c['row_ids']}"
            st.markdown(line)
            if c.get("sql"):
                st.code(c["sql"], language="sql")
            rows = c.get("rows")
            if rows:
                st.caption("Rows returned by this query:")
                st.dataframe(rows, width="stretch", hide_index=True)
            elif c.get("row_count") == 0:
                st.caption("Query returned no rows.")


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def stream_text(text: str, delay: float = 0.015):
    """Yield `text` word-by-word so st.write_stream renders a typing effect."""
    for token in text.split(" "):
        yield token + " "
        time.sleep(delay)


def render_chat(pipe) -> None:
    st.subheader("💬 Chat")

    # Reset the input box on the run after a question was submitted.
    if st.session_state.pop("clear_box", False):
        st.session_state["qbox"] = ""

    messages = st.session_state.setdefault("messages", [])
    for idx, m in enumerate(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("caption"):
                st.caption(m["caption"])
            if m.get("steps"):
                with st.expander("🧠 Reasoning"):
                    for s in m["steps"]:
                        st.markdown(f"**{s['title']}** — {s['detail']}"
                                    if s.get("detail") else f"**{s['title']}**")
            if m.get("citations"):
                # Expand the latest answer's sources so the chunk is always
                # visible; older ones stay collapsed to keep history tidy.
                with st.expander("Sources", expanded=(idx == len(messages) - 1)):
                    render_citations(idx, m["citations"])
            elif m.get("sources"):              # legacy messages from older runs
                with st.expander("Sources"):
                    st.text("\n".join(m["sources"]))

    # When the latest turn is an unanswered question, generate the answer and
    # reveal it with a typing effect. We stream the *verified* answer the
    # pipeline returns — true token streaming can't run ahead of the
    # citation-verification gate, so we type out the final, checked text.
    if messages and messages[-1]["role"] == "user":
        q = messages[-1]["content"]
        with st.chat_message("assistant"):
            steps: list[dict] = []
            try:
                gate = router.triage(q)                 # chat vs. knowledge base
                if gate["needs_retrieval"]:
                    # Stream the pipeline's reasoning, one step at a time.
                    with st.status("Reasoning…", expanded=True) as status:
                        def on_step(title, detail=""):
                            st.markdown(f"**{title}** — {detail}" if detail
                                        else f"**{title}**")
                            steps.append({"title": title, "detail": detail})
                        answer = pipe.ask(q, force_llm=True, on_step=on_step)
                        status.update(label="Reasoning complete",
                                      state="complete", expanded=False)
                else:
                    answer = None
            except Exception as exc:                    # LLM-only: no fallback
                msg = (f"⚠️ Couldn't get an LLM answer: {exc}\n\n"
                       "Check your OpenAI key, the selected model, and your quota.")
                st.error(msg)
                messages.append({"role": "assistant", "content": msg})
            else:
                if answer is None:                      # conversational — RAG not invoked
                    reply = gate["reply"] or ("Hi! I can answer questions about your orders, "
                                              "products, warranty, specs, and service bulletins.")
                    st.write_stream(stream_text(reply))
                    st.caption("💬 chat")
                    messages.append({"role": "assistant", "content": reply,
                                     "caption": "💬 chat"})
                else:
                    st.write_stream(stream_text(answer.text))
                    caption = f"{answer.routing.route} · " + (
                        "✅ verified" if answer.verified else "⚠️ not verified")
                    st.caption(caption)
                    cites = citation_payload(answer)
                    if cites:
                        with st.expander("Sources", expanded=True):
                            render_citations(len(messages), cites)
                    messages.append({"role": "assistant", "content": answer.text,
                                     "caption": caption, "citations": cites,
                                     "steps": steps})

    # Suggestion bubbles — clicking one drops the question into the box below.
    st.caption("Try one of these:")
    chip_cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if chip_cols[i % 2].button(suggestion, key=f"chip_{i}", width="stretch"):
            st.session_state["qbox"] = suggestion

    # A form so pressing Enter in the box submits; Send sits on the right.
    with st.form("ask_form", clear_on_submit=False, border=False):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        question = c1.text_input(
            "Ask a question", key="qbox", label_visibility="collapsed",
            placeholder="Ask about orders, warranty, specs, or bulletins…")
        sent = c2.form_submit_button("Send ▸", type="primary", width="stretch")
    if sent and question.strip():
        messages.append({"role": "user", "content": question.strip()})
        st.session_state["clear_box"] = True
        st.rerun()


def render_documents() -> None:
    st.subheader("📄 Documents Ingested")
    st.caption("Retrieval is lexical (BM25 + TF-IDF) — there is no vector DB. "
               "“Ingested” = parsed into page-chunks and added to the indexes + graph.")

    counts = chunk_counts()
    rows = [{
        "Document": f.name,
        "Status": "✅ Ingested" if counts.get(f.name) else "🕒 Pending ingest",
        "Chunks": counts.get(f.name, 0),
        "Size (KB)": round(f.stat().st_size / 1024, 1),
        "Type": "seed" if f.name in SEED_PDFS else "uploaded",
    } for f in list_pdfs()]
    st.dataframe(rows, width="stretch", hide_index=True)

    st.divider()
    st.markdown("**Upload your own PDFs**")
    uploads = st.file_uploader("Add PDFs to the corpus", type=["pdf"],
                               accept_multiple_files=True, key="uploader")
    if uploads and st.button("Save & re-ingest", type="primary"):
        for uf in uploads:
            (settings.pdf_dir / Path(uf.name).name).write_bytes(uf.getbuffer())
        with st.spinner("Re-ingesting the corpus (rebuilding indexes + graph)…"):
            summary = reingest()
        st.success(f"Done — {summary['chunks']} chunks across {len(list_pdfs())} PDFs, "
                   f"graph: {summary['graph_nodes']} nodes / {summary['graph_edges']} edges.")
        st.rerun()

    extra = [f.name for f in list_pdfs() if f.name not in SEED_PDFS]
    if extra:
        st.markdown("**Remove an uploaded PDF**")
        choice = st.selectbox("Uploaded documents", ["—"] + extra,
                              label_visibility="collapsed")
        if choice != "—" and st.button("Remove & re-ingest"):
            (settings.pdf_dir / choice).unlink(missing_ok=True)
            with st.spinner("Re-ingesting…"):
                reingest()
            st.rerun()


def render_sql() -> None:
    st.subheader("🗄 SQL Data")
    st.caption(f"Read-only view of the SQLite database · {settings.db_path.name} "
               "— the structured source that [S#] citations resolve to.")
    if not settings.db_path.exists():
        st.info("No database found — build the index first.")
        return
    tables = load_tables()
    tabs = st.tabs([f"{name} ({len(rows)})" for name, rows in tables.items()])
    for tab, (name, rows) in zip(tabs, tables.items()):
        with tab:
            if rows:
                st.dataframe(rows, width="stretch", hide_index=True)
            else:
                st.caption("(empty table)")


def render_graph(pipe) -> None:
    st.subheader("🕸 Knowledge Graph")
    g = pipe.graph.g
    st.caption(f"{g.number_of_nodes()} nodes · {g.number_of_edges()} edges · "
               "networkx MultiDiGraph  ·  "
               "🟦 product 🟩 customer 🟧 bulletin ⬜ doc/chunk")
    st.graphviz_chart(graph_to_dot(g), width="stretch")

    with st.expander("Nodes"):
        st.dataframe(
            [{"Node": n, "Kind": a.get("kind"),
              "Type": a.get("type", ""), "Name/Section": a.get("name") or a.get("section", "")}
             for n, a in g.nodes(data=True)],
            width="stretch", hide_index=True)
    with st.expander("Edges (relationships)"):
        st.dataframe(
            [{"From": u, "Relationship": d.get("rel"), "To": v}
             for u, v, d in g.edges(data=True)],
            width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def require_password() -> None:
    """Optional gate — active only if an APP_PASSWORD secret/env var is set."""
    expected = os.getenv("APP_PASSWORD")
    if not expected or st.session_state.get("_authed"):
        return
    st.title("🔒 Hybrid RAG")
    pw = st.text_input("Enter password to continue", type="password")
    if pw == expected:
        st.session_state["_authed"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    st.stop()


st.set_page_config(page_title="Hybrid RAG", page_icon="💬", layout="wide")
require_password()

with st.sidebar:
    st.title("💬 Hybrid RAG")
    view = st.radio("View",
                    ["💬 Chat", "📄 Documents Ingested", "🗄 SQL Data", "🕸 Knowledge Graph"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("**LLM-only mode** — every query is routed, queried (LLM-written SQL), "
               "and answered by the LLM.")
    if os.getenv("OPENAI_API_KEY"):
        default = settings.llm_model if settings.llm_model in MODELS else "Custom…"
        pick = st.selectbox("Model", MODELS, index=MODELS.index(default))
        chosen = (st.text_input("Custom model id", value=settings.llm_model).strip()
                  if pick == "Custom…" else pick)
        llm.set_model(chosen)
        st.caption(f"🔌 OpenAI · `{chosen or settings.llm_model}`")
    else:
        st.error("LLM-only mode needs `OPENAI_API_KEY` — set it in `.env`, then rerun.")
    if view == "💬 Chat":
        st.divider()
        if st.button("🗑 Clear chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

if not artifacts_ready():
    with st.spinner("Building the knowledge base (first run)…"):
        reingest()
    st.rerun()

pipe = load_pipeline()

if view == "💬 Chat":
    render_chat(pipe)
elif view == "📄 Documents Ingested":
    render_documents()
elif view == "🗄 SQL Data":
    render_sql()
else:
    render_graph(pipe)
