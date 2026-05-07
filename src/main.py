"""FastAPI application: ingest, query, and evaluate endpoints."""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.config import settings
from src.evaluation.evaluator import load_golden_dataset, run_evaluation
from src.generation.answer_generator import generate_answer
from src.generation.prompt_templates import load_templates
from src.ingestion.chunker import chunk_pages
from src.ingestion.document_loader import load_document
from src.ingestion.embedder import embed_and_store
from src.models import EvaluationResult, IngestResponse, QueryRequest, QueryResponse
from src.reranking.reranker import rerank
from src.retrieval.hybrid_search import get_hybrid_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load prompt templates and warm up retriever."""
    # Strip whitespace from API keys so Ragas/OpenAI SDK read clean values
    for key in ("OPENAI_API_KEY", "COHERE_API_KEY"):
        if key in os.environ:
            os.environ[key] = os.environ[key].strip()
    load_templates()
    get_hybrid_retriever()
    logger.info("RAG API startup complete")
    yield


app = FastAPI(
    title="Production RAG System",
    description="Hybrid retrieval, re-ranking, citation enforcement, and evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "healthy", "service": "rag-api", "version": "1.0.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a document (PDF, markdown, txt), chunk, embed, and store.
    Returns number of chunks created.
    """
    content = await file.read()
    try:
        pages = load_document(file_path=file.filename, content=content, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_pages(pages)
    stored = embed_and_store(chunks)

    return IngestResponse(
        status="ok",
        filename=file.filename,
        chunks_created=stored,
        message=f"Ingested {stored} chunks from {file.filename}",
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Answer a question using hybrid retrieval, re-ranking, and LLM generation.
    Returns answer with citations and confidence score.
    """
    retriever = get_hybrid_retriever()
    candidates = retriever.search(request.question, top_k=settings.top_k_retrieval)

    if not candidates:
        return QueryResponse(
            question=request.question,
            answer="I cannot answer based on the provided documents.",
            citations=[],
            confidence=0.0,
            retrieval_method="hybrid",
        )

    reranked = rerank(request.question, candidates, top_k=request.top_k)
    response = generate_answer(request.question, reranked)
    return response


@app.get("/evaluate", response_model=EvaluationResult)
async def evaluate() -> EvaluationResult:
    """
    Run evaluation suite on the golden dataset and check quality thresholds.
    Returns Ragas metrics and pass/fail status.
    """
    golden = load_golden_dataset()
    metrics = run_evaluation(golden)

    passed = (
        metrics.get("faithfulness", 0.0) >= settings.min_faithfulness
        and metrics.get("context_precision@5", 0.0) >= settings.min_precision_at_5
    )

    return EvaluationResult(
        metrics=metrics,
        thresholds={
            "min_faithfulness": settings.min_faithfulness,
            "min_precision_at_5": settings.min_precision_at_5,
        },
        passed=passed,
        timestamp=datetime.now().isoformat(),
    )
