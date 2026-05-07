"""Unit tests for answer generation and citation enforcement."""
from unittest.mock import MagicMock, patch

import pytest

from src.generation.answer_generator import generate_answer
from src.models import QueryResponse


def _make_chunks(ids_and_texts):
    """Helper: build minimal chunk dicts for tests."""
    return [
        {
            "chunk_id": cid,
            "text": text,
            "source": "test_doc.txt",
            "page": 1,
            "score": 0.9,
            "retrieval_method": "hybrid",
        }
        for cid, text in ids_and_texts
    ]


@patch("src.generation.answer_generator._get_client")
def test_citation_enforcement_valid(mock_client):
    """Valid citations pass through unchanged."""
    chunks = _make_chunks([
        ("chunk-aa001122", "Renewable energy includes solar and wind power."),
        ("chunk-bb334455", "Solar panels convert sunlight to electricity."),
    ])

    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = (
        "Renewable energy includes solar and wind [chunk-aa001122]. "
        "Solar panels convert sunlight [chunk-bb334455]."
    )
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    response = generate_answer("What is renewable energy?", chunks)

    assert isinstance(response, QueryResponse)
    assert "cannot answer" not in response.answer.lower()
    assert len(response.citations) == 2


@patch("src.generation.answer_generator._get_client")
def test_citation_enforcement_invalid_id(mock_client):
    """Answer referencing non-existent chunk ID triggers refusal."""
    chunks = _make_chunks([
        ("chunk-aa001122", "Renewable energy includes solar and wind power."),
    ])

    mock_completion = MagicMock()
    # LLM hallucinates a chunk ID that doesn't exist
    mock_completion.choices[0].message.content = (
        "Nuclear fusion is a future energy source [chunk-ffffffff]."
    )
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    response = generate_answer("What is nuclear fusion?", chunks)

    assert "cannot answer" in response.answer.lower()
    assert len(response.citations) == 0


@patch("src.generation.answer_generator._get_client")
def test_empty_chunks_returns_refusal(mock_client):
    """No chunks → immediate refusal without LLM call."""
    response = generate_answer("What is solar energy?", [])

    mock_client.return_value.chat.completions.create.assert_not_called()
    assert "cannot answer" in response.answer.lower()
    assert response.confidence == 0.0


@patch("src.generation.answer_generator._get_client")
def test_response_confidence_calculated(mock_client):
    """Confidence equals mean rerank_score of provided chunks."""
    chunks = _make_chunks([
        ("chunk-cc112233", "Solar panels use photovoltaic cells."),
    ])
    chunks[0]["rerank_score"] = 0.76

    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = (
        "Solar panels use photovoltaic cells [chunk-cc112233]."
    )
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    response = generate_answer("How do solar panels work?", chunks)

    assert abs(response.confidence - 0.76) < 0.01


@patch("src.generation.answer_generator._get_client")
def test_retrieval_method_propagated(mock_client):
    """Retrieval method from chunks propagates to response."""
    chunks = _make_chunks([("chunk-dd667788", "Wind turbines generate electricity.")])
    chunks[0]["retrieval_method"] = "hybrid"

    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = (
        "Wind turbines generate electricity [chunk-dd667788]."
    )
    mock_client.return_value.chat.completions.create.return_value = mock_completion

    response = generate_answer("How does wind energy work?", chunks)
    assert response.retrieval_method == "hybrid"
