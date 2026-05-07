"""Re-rank candidate chunks with Cohere cross-encoder."""
import logging
from typing import List, Dict, Any

import cohere

from src.config import settings

logger = logging.getLogger(__name__)

_client: cohere.Client = None


def _get_client() -> cohere.Client:
    """Lazy-init Cohere client."""
    global _client
    if _client is None:
        _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
    """
    Re-rank chunks using Cohere's cross-encoder model.

    Returns top_k chunks sorted by relevance score (highest first).
    Each chunk gains a 'rerank_score' key.
    """
    if top_k is None:
        top_k = settings.top_k_rerank

    if not chunks:
        return []

    client = _get_client()
    documents = [c["text"] for c in chunks]

    response = client.rerank(
        model=settings.rerank_model,
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
    )

    reranked: List[Dict[str, Any]] = []
    for result in response.results:
        chunk = dict(chunks[result.index])
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    logger.debug("Re-ranked %d → %d chunks", len(chunks), len(reranked))
    return reranked
