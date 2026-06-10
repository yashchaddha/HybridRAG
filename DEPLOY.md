# Deploying the Hybrid RAG app

This is a **Streamlit** app — it runs a persistent WebSocket server, so it needs a
container/VM host, **not** a serverless platform like Vercel. The repo ships a
`Dockerfile`, so any container host works.

## What you need
- **`OPENAI_API_KEY`** as a secret — the app's LLM-only mode requires it.
- Optional **`LLM_MODEL`** (default `gpt-4o-mini`; the UI also has a model picker).
- The image runs `ingest` at **build time**, baking the demo DB / 3 seed PDFs /
  indexes / knowledge graph in, so the app boots ready (no key needed to build).

## Local (Docker)
```bash
docker build -t hybrid-rag .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... -e PORT=8501 hybrid-rag
# → http://localhost:8501
```

## Render (uses the included `render.yaml`)
1. Push the repo to GitHub.
2. Render → **New → Blueprint** → select the repo (reads `render.yaml`).
3. In the service's **Environment**, set `OPENAI_API_KEY` (marked `sync: false`).
Render injects `$PORT`; the Dockerfile reads it. Health check: `/_stcore/health`.

## Railway
1. **New Project → Deploy from GitHub repo** (Railway auto-detects the `Dockerfile`).
2. Add variables `OPENAI_API_KEY` (and optionally `LLM_MODEL`).
Railway sets `$PORT` automatically.

## Fly.io
```bash
fly launch --no-deploy            # generates fly.toml; set internal_port = 8501
fly secrets set OPENAI_API_KEY=sk-...
fly deploy
```

## Notes
- **Ephemeral filesystem.** Uploaded PDFs and new traces reset on restart; the seed
  corpus is rebuilt on boot. To persist uploads, mount a volume at `/app/data`
  (Fly `fly volumes create`, Railway volume, Render disk on a paid plan).
- **WebSocket behind a proxy.** If the page loads but stays "connecting," append to the
  Dockerfile `CMD`: `--server.enableCORS=false --server.enableXsrfProtection=false`.
- **Cost.** With a key set, every question makes a few OpenAI calls (triage + routing +
  LLM SQL + synthesis). Use `gpt-4o-mini` (default) to keep it cheap.
