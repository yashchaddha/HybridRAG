"""Central configuration for the hybrid retrieval POC.

Everything is overridable via environment variables so the same code runs
locally, in CI, and in a container without edits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load a local .env (if present) so OPENAI_API_KEY / LLM_MODEL / retrieval
# tunables can be supplied without exporting them by hand. Guarded so the
# package still imports if python-dotenv isn't installed (minimal/offline).
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    # --- paths ---------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    db_path: Path = PROJECT_ROOT / "data" / "app.db"
    pdf_dir: Path = PROJECT_ROOT / "data" / "seed_pdfs"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    runs_dir: Path = PROJECT_ROOT / "runs"

    # --- retrieval tunables -------------------------------------------
    doc_top_k: int = int(os.getenv("DOC_TOP_K", "5"))
    doc_candidates: int = int(os.getenv("DOC_CANDIDATES", "10"))
    per_doc_cap: int = int(os.getenv("PER_DOC_CAP", "3"))
    rrf_k: int = 60                      # reciprocal-rank-fusion constant
    kg_expansion_max: int = 2            # max chunks injected by the graph
    sql_row_limit: int = 50              # hard cap on rows returned to the LLM
    sql_max_retries: int = int(os.getenv("SQL_MAX_RETRIES", "3"))  # LLM SQL self-correction

    # --- LLM (optional) ------------------------------------------------
    # When OPENAI_API_KEY is set, the router / NL->SQL / synthesizer use
    # OpenAI. Without it, every stage degrades to a deterministic fallback
    # so the full pipeline still runs offline (CI, air-gapped demos).
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # --- routing thresholds --------------------------------------------
    route_threshold: float = 1.0         # min signal score to engage a source

    sources_label: dict = field(default_factory=lambda: {
        "sqlite": "SQLite", "pdf": "PDF"})


settings = Settings()

for _p in (settings.data_dir, settings.pdf_dir, settings.index_dir, settings.runs_dir):
    _p.mkdir(parents=True, exist_ok=True)
