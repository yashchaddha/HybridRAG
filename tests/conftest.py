import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hybrid_rag import ingest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ingested():
    """Build DB, PDFs, indexes and graph once for the whole test session."""
    return ingest.run_ingest()


@pytest.fixture(scope="session")
def pipeline(ingested):
    from hybrid_rag.pipeline import Pipeline
    return Pipeline()


@pytest.fixture(scope="session")
def graph(pipeline):
    return pipeline.graph
