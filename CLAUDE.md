# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run API (local dev):**
```bash
uvicorn src.main:app --reload --port 8000
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_retrieval.py          # single file
pytest tests/test_retrieval.py::test_fn # single test
pytest --cov=src tests/
```

**Ingest demo documents:**
```bash
bash scripts/ingest_demo_docs.sh
```

**Run evaluation:**
```bash
bash scripts/run_evaluation.sh
python scripts/check_quality_gate.py    # exits 1 if metrics below threshold
```

**Docker (ChromaDB only):**
```bash
docker compose up -d chromadb           # starts ChromaDB on port 8001
docker compose up -d                    # starts ChromaDB + rag-api container
```

## Environment

Copy `.env.example` → `.env`. Required keys: `OPENAI_API_KEY`, `COHERE_API_KEY`.

ChromaDB defaults: `localhost:8001` (local dev) or `chromadb:8000` (inside Docker network).

## Architecture

Request flow: `POST /query` → `HybridRetriever.search()` → `rerank()` → `generate_answer()` → citation validation → `QueryResponse`

**Ingestion pipeline** (`src/ingestion/`):
- `document_loader.py` — extracts pages from PDF/markdown/txt into `{"text", "source", "page"}` dicts
- `chunker.py` — splits pages into chunks (700 chars, 100 overlap)
- `embedder.py` — embeds via `all-MiniLM-L6-v2`, stores in ChromaDB collection `rag_documents`; `get_collection()` returns the singleton collection

**Retrieval** (`src/retrieval/`):
- `vector_search.py` — ChromaDB cosine similarity
- `hybrid_search.py` — `HybridRetriever` merges BM25 + vector via Reciprocal Rank Fusion (RRF, k=60); BM25 index rebuilt in-memory on every query from ChromaDB; singleton via `get_hybrid_retriever()`

**Reranking** (`src/reranking/reranker.py`): Cohere `rerank-english-v3.0`, top-5 from top-20 candidates

**Generation** (`src/generation/`):
- `prompt_templates.py` — loads `prompts/rag_prompts.yaml` at startup; `get_template("qa_with_citations")` returns system+user strings
- `answer_generator.py` — sends chunks + question to `gpt-4o-mini` (temp=0); validates that all cited chunk IDs (`[chunk-xxxxxxxx]`) exist in the provided set — hallucinated IDs cause answer replacement with refusal

**Evaluation** (`src/evaluation/evaluator.py`): Ragas metrics `faithfulness` and `context_precision@5`; thresholds `0.85` / `0.70` enforced by CI via `scripts/check_quality_gate.py` hitting `GET /evaluate`

**Config** (`src/config.py`): Single `Settings` pydantic-settings class; all tunable params (chunk size, top-k, model names, thresholds) readable from env.

**Prompt versioning**: `prompts/rag_prompts.yaml` has a `version` field. Edit prompts here, not in code.

## Key invariants

- ChromaDB collection name is hardcoded as `rag_documents` in `embedder.py` — `get_collection()` must be used everywhere to get it
- Citation format is `[chunk-xxxxxxxx]` (8 hex chars) — regex in `answer_generator.py` enforces this
- BM25 index is in-memory only; it rebuilds from ChromaDB on each query, so it always reflects current state but adds latency
