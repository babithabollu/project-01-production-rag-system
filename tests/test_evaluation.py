"""Tests for evaluation pipeline and golden dataset."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


def test_golden_dataset_loads():
    """Golden dataset JSON is valid and has required fields."""
    path = Path("src/evaluation/golden_dataset.json")
    assert path.exists(), "golden_dataset.json must exist"
    with path.open() as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= 10, "Golden dataset must have at least 10 entries"
    for item in data:
        assert "question" in item
        assert "ground_truth_answer" in item
        assert "source_documents" in item


def test_golden_dataset_questions_non_empty():
    """All questions are non-empty strings."""
    path = Path("src/evaluation/golden_dataset.json")
    with path.open() as f:
        data = json.load(f)
    for item in data:
        assert isinstance(item["question"], str)
        assert len(item["question"].strip()) > 5


@patch("src.evaluation.evaluator.get_hybrid_retriever")
@patch("src.evaluation.evaluator.generate_answer")
@patch("src.evaluation.evaluator.evaluate")
def test_run_evaluation_returns_metrics(mock_evaluate, mock_generate, mock_retriever):
    """run_evaluation returns dict with all expected metric keys."""
    from src.evaluation.evaluator import run_evaluation

    mock_retriever.return_value.search.return_value = [
        {"chunk_id": "chunk-aa001122", "text": "Solar energy is renewable.", "source": "doc.txt", "page": 1, "score": 0.9}
    ]

    mock_response = MagicMock()
    mock_response.answer = "Solar energy is renewable [chunk-aa001122]."
    mock_generate.return_value = mock_response

    mock_evaluate.return_value = {
        "faithfulness": 0.88,
        "answer_relevancy": 0.82,
        "context_precision": 0.74,
        "context_recall": 0.77,
    }

    golden = [
        {
            "question": "What is solar energy?",
            "ground_truth_answer": "Solar energy is energy from the sun.",
            "source_documents": ["renewable_energy_guide.md"],
        }
    ]

    scores = run_evaluation(golden)

    assert "faithfulness" in scores
    assert "context_precision@5" in scores
    assert "answer_relevancy" in scores
    assert "context_recall" in scores
    assert all(0.0 <= v <= 1.0 for v in scores.values())
