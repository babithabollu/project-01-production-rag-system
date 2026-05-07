"""Pydantic schemas for the RAG API."""
from pydantic import BaseModel, Field
from typing import List, Optional


class Citation(BaseModel):
    """A single citation referencing a source chunk."""

    chunk_id: str
    text: str
    source: str
    page: Optional[int] = None


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    question: str = Field(..., min_length=5, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""

    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    retrieval_method: str


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    status: str
    filename: str
    chunks_created: int
    message: str


class EvaluationResult(BaseModel):
    """Result of running the evaluation suite."""

    metrics: dict
    thresholds: dict
    passed: bool
    timestamp: str
