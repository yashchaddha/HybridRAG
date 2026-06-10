# Hybrid RAG — Streamlit app container.
# Streamlit needs a persistent server (not serverless), so this image runs on
# any container host: Render, Railway, Fly.io, Cloud Run, etc.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PORT=8501

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Bake the demo corpus (DB + 3 seed PDFs + lexical indexes + knowledge graph)
# into the image so the app boots ready. Deterministic and offline — no key.
RUN python -m hybrid_rag.cli ingest

EXPOSE 8501

# Hosts inject $PORT (Render/Railway); fall back to 8501 for local runs.
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
