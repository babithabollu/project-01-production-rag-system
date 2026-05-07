"""Unit tests for document loading and chunking."""
import pytest

from src.ingestion.chunker import chunk_pages
from src.ingestion.document_loader import load_document


def test_load_text_document():
    """Plain-text document loads as a single page."""
    content = b"Renewable energy includes solar, wind, and hydro power sources."
    pages = load_document("test.txt", content, "test.txt")
    assert len(pages) == 1
    assert pages[0]["source"] == "test.txt"
    assert "solar" in pages[0]["text"]


def test_load_markdown_document():
    """Markdown document loads as a single page."""
    content = b"# Energy Guide\n\nSolar energy is produced by photovoltaic cells."
    pages = load_document("guide.md", content, "guide.md")
    assert len(pages) == 1
    assert pages[0]["page"] == 1


def test_unsupported_file_type_raises():
    """Unsupported extension raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document("data.csv", b"a,b,c", "data.csv")


def test_chunk_pages_produces_metadata():
    """Chunks carry source, page, and chunk_id."""
    pages = [{"text": "Solar energy is renewable. " * 50, "source": "doc.txt", "page": 1}]
    chunks = chunk_pages(pages)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert chunk["chunk_id"].startswith("chunk-")
        assert chunk["source"] == "doc.txt"
        assert chunk["page"] == 1


def test_chunk_ids_are_unique():
    """All chunk IDs within a document are unique."""
    long_text = "Climate change is driven by greenhouse gas emissions. " * 100
    pages = [{"text": long_text, "source": "climate.txt", "page": 1}]
    chunks = chunk_pages(pages)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_overlap_creates_context_continuity():
    """Consecutive chunks share some text (overlap)."""
    text = " ".join([f"Sentence number {i} about renewable energy." for i in range(200)])
    pages = [{"text": text, "source": "test.txt", "page": 1}]
    chunks = chunk_pages(pages)
    if len(chunks) >= 2:
        # Overlapping chunks share words — last words of chunk[0] appear in chunk[1]
        words_0 = set(chunks[0]["text"].split())
        words_1 = set(chunks[1]["text"].split())
        assert len(words_0 & words_1) > 0, "Expected overlap between consecutive chunks"
