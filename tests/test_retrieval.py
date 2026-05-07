"""Integration tests for hybrid retrieval."""
import pytest

from src.retrieval.hybrid_search import HybridRetriever


@pytest.fixture()
def retriever_with_docs():
    """Fresh retriever with climate-domain test documents."""
    r = HybridRetriever()
    docs = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The primary greenhouse gas driving climate change is carbon dioxide (CO2).",
        "Renewable energy sources like solar and wind can reduce carbon emissions significantly.",
        "Solar photovoltaic panels convert sunlight directly into electricity using silicon cells.",
        "Wind turbines generate electricity when wind rotates their blades, driving a generator.",
    ]
    r.ingest(docs)
    return r


def test_hybrid_search_returns_results(retriever_with_docs):
    """Search returns non-empty list of chunks."""
    results = retriever_with_docs.search("What causes climate change?", top_k=3)
    assert len(results) > 0


def test_hybrid_search_top_result_relevant(retriever_with_docs):
    """Top result for climate question contains relevant keywords."""
    results = retriever_with_docs.search("What causes climate change?", top_k=3)
    top_text = results[0]["text"].lower()
    assert any(kw in top_text for kw in ["greenhouse", "emissions", "climate", "carbon"])


def test_hybrid_search_score_positive(retriever_with_docs):
    """All returned chunks have a positive score."""
    results = retriever_with_docs.search("How does solar energy work?", top_k=5)
    for r in results:
        assert r["score"] > 0, f"Expected positive score, got {r['score']}"


def test_hybrid_search_respects_top_k(retriever_with_docs):
    """Result count does not exceed requested top_k."""
    results = retriever_with_docs.search("renewable energy", top_k=2)
    assert len(results) <= 2


def test_hybrid_search_chunk_metadata(retriever_with_docs):
    """Each result has required metadata fields."""
    results = retriever_with_docs.search("greenhouse gas", top_k=3)
    for r in results:
        assert "chunk_id" in r
        assert "text" in r
        assert "source" in r
        assert "score" in r


def test_hybrid_search_returns_method_label(retriever_with_docs):
    """Result carries retrieval_method label."""
    results = retriever_with_docs.search("carbon dioxide emissions", top_k=3)
    for r in results:
        assert r.get("retrieval_method") in {"hybrid", "vector-only"}
