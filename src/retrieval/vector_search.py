"""Semantic vector search against ChromaDB using cosine similarity."""
import logging
from typing import List, Dict, Any

from src.ingestion.embedder import get_collection, get_embedding_model
from src.config import settings

logger = logging.getLogger(__name__)


def vector_search(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Embed query and retrieve top_k chunks by cosine similarity.

    Returns list of dicts with: chunk_id, text, source, page, score.
    """
    if top_k is None:
        top_k = settings.top_k_retrieval

    model = get_embedding_model()
    collection = get_collection()

    query_embedding = model.encode([query], show_progress_bar=False)[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return chunks

    for i, chunk_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1.0 - distance  # cosine distance → similarity
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "page": results["metadatas"][0][i].get("page"),
                "score": score,
            }
        )

    logger.debug("Vector search returned %d chunks for query: %.50s", len(chunks), query)
    return chunks
