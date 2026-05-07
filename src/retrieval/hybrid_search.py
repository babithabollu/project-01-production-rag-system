"""Hybrid retrieval: BM25 keyword search + vector search merged via RRF."""
import logging
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from src.ingestion.embedder import get_collection
from src.retrieval.vector_search import vector_search
from src.config import settings

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF constant — standard default


class HybridRetriever:
    """
    Combines BM25 keyword ranking with dense vector ranking via
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(self) -> None:
        self._corpus: List[str] = []
        self._corpus_ids: List[str] = []
        self._corpus_meta: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi = None

    def _build_bm25_index(self) -> None:
        """Load all documents from ChromaDB and build an in-memory BM25 index."""
        collection = get_collection()
        count = collection.count()
        if count == 0:
            self._bm25 = None
            return

        result = collection.get(include=["documents", "metadatas"])
        self._corpus = result["documents"]
        self._corpus_ids = result["ids"]
        self._corpus_meta = result["metadatas"]

        tokenized = [doc.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 index built over %d documents", count)

    def ingest(self, texts: List[str]) -> None:
        """
        Ingest raw text strings into the retriever (for unit testing).
        Rebuilds BM25 index after ingestion.
        """
        from src.ingestion.chunker import chunk_pages
        from src.ingestion.embedder import embed_and_store

        pages = [{"text": t, "source": "test", "page": 1} for t in texts]
        chunks = chunk_pages(pages)
        embed_and_store(chunks)
        self._build_bm25_index()

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Hybrid search: RRF merge of BM25 and vector results.

        Returns up to top_k chunks sorted by RRF score.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        # Rebuild BM25 index on every query so newly ingested docs are included.
        self._build_bm25_index()

        vector_results = vector_search(query, top_k=top_k)

        if self._bm25 is None or not self._corpus:
            # No BM25 index available — fall back to vector only.
            for r in vector_results:
                r["retrieval_method"] = "vector-only"
            return vector_results[:top_k]

        # BM25 ranking
        tokens = query.lower().split()
        bm25_scores = self._bm25.get_scores(tokens)
        bm25_ranked = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]

        # Build id→rank maps
        vector_rank: Dict[str, int] = {
            r["chunk_id"]: i for i, r in enumerate(vector_results)
        }
        bm25_rank: Dict[str, int] = {
            self._corpus_ids[i]: rank for rank, i in enumerate(bm25_ranked)
        }

        # Collect all candidate ids
        candidate_ids = set(vector_rank) | set(bm25_rank)

        rrf_scores: Dict[str, float] = {}
        for cid in candidate_ids:
            vr = vector_rank.get(cid, len(self._corpus))
            br = bm25_rank.get(cid, len(self._corpus))
            rrf_scores[cid] = 1.0 / (RRF_K + vr) + 1.0 / (RRF_K + br)

        # Build chunk lookup from vector results
        chunk_lookup: Dict[str, Dict[str, Any]] = {r["chunk_id"]: r for r in vector_results}
        # Add BM25-only chunks not in vector results
        for rank, idx in enumerate(bm25_ranked):
            cid = self._corpus_ids[idx]
            if cid not in chunk_lookup:
                chunk_lookup[cid] = {
                    "chunk_id": cid,
                    "text": self._corpus[idx],
                    "source": self._corpus_meta[idx].get("source", ""),
                    "page": self._corpus_meta[idx].get("page"),
                    "score": bm25_scores[idx],
                }

        merged = sorted(candidate_ids, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in merged:
            chunk = dict(chunk_lookup[cid])
            chunk["score"] = rrf_scores[cid]
            chunk["retrieval_method"] = "hybrid"
            results.append(chunk)

        logger.debug("Hybrid search returned %d chunks", len(results))
        return results


# Module-level singleton
_retriever: HybridRetriever = None


def get_hybrid_retriever() -> HybridRetriever:
    """Return shared HybridRetriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
