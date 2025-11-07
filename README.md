# Quantum ArXiv Newsletter (local, OSS, Ollama)

Create a short daily newsletter of **arXiv papers matching “quantum” in the last N hours**.  
Runs fully **locally** on a single machine: arXiv ingest → embeddings → clustering → two summaries per topic → Markdown newsletter.

- **Local LLMs** via **Ollama** (choose your model)
- **Embeddings** via Sentence-Transformers (default: `all-MiniLM-L6-v2`)
- **No telemetry by default** (see “Telemetry & Offline”)
- Optional **MCP** fetch and an **LLM “editor”** that curates which clusters to publish

---

## What it does (end-to-end)

1. **Fetch**: pull arXiv results for a query (default `all:quantum`) in a **rolling** time window (e.g. last 24 hours), keeping title, authors, abstract, ID, links.
2. **Embed**: build a text field (`title + abstract`) and compute normalized embeddings.
3. **Cluster**: group similar papers using agglomerative clustering on **cosine distance** (`1 - cosine_similarity`).
4. **Select (optional)**: an **editor** agent ranks clusters by size & cohesion (or prefers diversity) and picks the top K.
5. **Summarize** (two styles per selected cluster):
   - **Scientific**: exactly **3 bullets**, concise and precise, inline arXiv ID citations.
   - **Science-comm**: 1 short paragraph + final “Why it matters:” sentence; citations appended.
6. **Render**: output a Markdown newsletter using `newsletter/template.md.j2`.



---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Ensure Ollama is running and a model is pulled
ollama serve &
ollama pull llama3.2     # or your preferred local model

# Run (basic)
python -m src.pipeline \
  --query "all:quantum" \
  --since-hours 24 \
  --threshold 0.33 \
  --max-items 4 \
  --min-cluster-size 2 \
  --max-clusters 6
```

> Output appears in `output/newsletter_<YYYY-MM-DD>.md`.

### With the LLM “editor”

```bash
python -m src.pipeline \
  --query "all:quantum" \
  --since-hours 24 \
  --threshold 0.55 \
  --max-items 4 \
  --min-cluster-size 2 \
  --max-clusters 6 \
  --editor \
  --cohesion-mode tight     # or: diverse
```

* `tight` favors **high cohesion** clusters (well-formed topics).
* `diverse` favors **lower cohesion** (broader themes).
  Scoring: `0.65*size_norm + 0.35*(cohesion or 1-cohesion)` depending on mode.



---

## CLI flags you’ll actually tune

* `--since-hours INT`
  Rolling window size (default 24). Typical: `24–72`.

* `--threshold FLOAT`
  **Agglomerative clustering distance threshold**. We cluster on **cosine distance** (`1 - cos_sim`).
  **Larger** threshold ⇒ **more merging** (fewer, larger clusters).
  **Smaller** threshold ⇒ stricter merges (more singletons).
  Try: `0.33` (conservative), `0.5–0.6` (merge more).

* `--min-cluster-size INT`
  Drop tiny clusters. Use `1` to include singletons. If filtering removes everything, the pipeline **falls back to including singletons** to avoid empty newsletters.

* `--max-clusters INT`
  Keep only the top-K clusters (after optional editor).

* `--max-items INT`
  Max papers per cluster to send to the summarizers. Lower = crisper summaries.

* `--editor` and `--cohesion-mode {tight,diverse}`
  Enable the editor and choose how cohesion is treated in selection.

* `--use-mcp`
  Fetch via the MCP server instead of the Python `arxiv` client.

---

## Config (.env)

You can set these in `.env` or override via CLI:

* `OLLAMA_BASE_URL` (default `http://localhost:11434`)
* `OLLAMA_MODEL` (default `llama3.2`)
  *(Set to any local Ollama model you like.)*
* `QUERY` (default `all:quantum`)
* `SINCE_HOURS` (default `24`)
* `EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
* `CLUSTER_THRESHOLD` (default `0.35`)
* `MAX_PAPERS_PER_CLUSTER` (default `6`)

---

## Telemetry & Offline

This project is set to **avoid egress by default**:

* `LANGSMITH_TRACING=false`, `LANGCHAIN_TRACING_V2=false`
* `HF_HUB_DISABLE_TELEMETRY=1`
* (If you later add CrewAI: `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`)

The only network calls are:

* the arXiv API during fetch
* the first-time download of embedding model files (cached afterwards)

For fully offline runs after the first install, you may set `HF_HUB_OFFLINE=1`.

---

## Troubleshooting

* **“No new quantum papers …”**
  Usually caused by over-filtering. Fix by:

  * increasing `--threshold` (e.g., `0.55–0.6`) to merge more,
  * or lowering `--min-cluster-size` to `1` to include singletons.

* **arXiv pagination error** (`UnexpectedEmptyPageError`)
  We cap to a single page (≤100) and treat empty second pages as end-of-results.

* **LLM outputs too long or chatty**
  The prompts are schema-locked and we clamp generation length. You can further lower `num_predict` in `src/summarizers.py`’s `make_llm(...)` or reduce `--max-items`.

---

## Project layout

```
quantum-arxiv-newsletter/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ newsletter/
│  └─ template.md.j2              # Markdown Jinja2 template
├─ src/
│  ├─ __init__.py
│  ├─ config.py                   # env + telemetry hardening
│  ├─ fetch.py                    # arXiv ingest (rolling window)
│  ├─ embed.py                    # sentence-transformers embeddings
│  ├─ cluster.py                  # agglomerative clustering helpers
│  ├─ editor.py                   # LLM/heuristic selection of clusters (optional)
│  ├─ summarizers.py              # scientific + sci-comm summaries
│  ├─ render.py                   # Jinja2 render + file write
│  └─ pipeline.py                 # end-to-end CLI
└─ output/                        # generated newsletters
```

---

## Extending (ideas)

* Swap `EMBED_MODEL` for a domain model (e.g., SPECTER-style) for tighter clusters.
* Add a critic pass that auto-fixes summaries that violate length or citation format.
* Emit HTML alongside Markdown (simple Jinja2 template add).
* Schedule via `cron` and email the MD/HTML (outside this repo).

---

## Requirements

* Python 3.9+
* Ollama running locally with your chosen model
* `pip install -r requirements.txt` 

---


