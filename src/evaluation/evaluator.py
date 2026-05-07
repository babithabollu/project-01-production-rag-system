"""Run Ragas evaluation metrics against the golden dataset."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.retrieval.hybrid_search import get_hybrid_retriever
from src.generation.answer_generator import generate_answer

logger = logging.getLogger(__name__)

_GOLDEN_PATH = Path("src/evaluation/golden_dataset.json")
_EVALUATION_FEATURES = Features(
    {
        "question": Value("string"),
        "answer": Value("string"),
        "contexts": Sequence(Value("string")),
        "ground_truth": Value("string"),
    }
)


def load_golden_dataset() -> List[Dict[str, Any]]:
    """Load the hand-verified Q&A pairs from disk."""
    with _GOLDEN_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(golden_dataset: List[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Run Ragas evaluation on golden_dataset.

    Returns dict: {faithfulness, answer_relevancy, context_precision, context_recall}.
    """
    if golden_dataset is None:
        golden_dataset = load_golden_dataset()

    retriever = get_hybrid_retriever()

    questions: List[str] = []
    answers: List[str] = []
    ground_truths: List[str] = []
    contexts: List[List[str]] = []

    for item in golden_dataset:
        question = item["question"]
        chunks = retriever.search(question, top_k=5)
        response = generate_answer(question, chunks)

        questions.append(str(question))
        answers.append(str(response.answer))
        ground_truths.append(str(item["ground_truth_answer"]))
        contexts.append([str(c["text"]) for c in chunks])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        },
        features=_EVALUATION_FEATURES,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision@5": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }
    logger.info("Evaluation scores: %s", scores)
    return scores
