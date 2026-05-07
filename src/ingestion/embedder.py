"""Embed chunks with sentence-transformers and persist to ChromaDB."""
import logging
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.config import settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_documents"

_model: SentenceTransformer = None
_collection = None


def _get_model() -> SentenceTransformer:
    """Lazy-load embedding model (shared singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Loaded embedding model: %s", settings.embedding_model)
    return _model


def _get_collection():
    """Lazy-connect to ChromaDB and return the documents collection."""
    global _collection
    if _collection is None:
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Connected to ChromaDB collection: %s", COLLECTION_NAME)
    return _collection


def embed_and_store(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed chunks and upsert into ChromaDB.

    Returns number of chunks stored.
    """
    if not chunks:
        return 0

    model = _get_model()
    collection = _get_collection()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Stored %d chunks in ChromaDB", len(chunks))
    return len(chunks)


def get_collection():
    """Expose collection for retrieval modules."""
    return _get_collection()


def get_embedding_model() -> SentenceTransformer:
    """Expose model for retrieval modules."""
    return _get_model()
