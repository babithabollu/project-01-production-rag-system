"""Generate cited answers via OpenAI LLM with citation validation."""
import logging
import re
from typing import List, Dict, Any

from openai import OpenAI

from src.config import settings
from src.generation.prompt_templates import get_template
from src.models import Citation, QueryResponse

logger = logging.getLogger(__name__)

_client: OpenAI = None

# Citation pattern: [chunk-xxxxxxxx]
_CITATION_RE = re.compile(r"\[chunk-[0-9a-f]{8}\]", re.IGNORECASE)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks as numbered context block."""
    parts = []
    for c in chunks:
        parts.append(f"[{c['chunk_id']}] (source: {c['source']}, page: {c.get('page', 'N/A')})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> QueryResponse:
    """
    Call OpenAI LLM with retrieved chunks and enforce citation validity.

    If the generated answer references chunk IDs not in the provided set,
    the answer is replaced with a refusal message.
    """
    if not chunks:
        return QueryResponse(
            question=question,
            answer="I cannot answer based on the provided documents.",
            citations=[],
            confidence=0.0,
            retrieval_method="hybrid",
        )

    template = get_template("qa_with_citations")
    context_block = _format_chunks(chunks)

    user_message = template["user"].format(
        chunks=context_block,
        question=question,
    )

    client = _get_client()
    completion = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": user_message},
        ],
    )

    raw_answer = completion.choices[0].message.content.strip()

    # Extract cited chunk IDs
    cited_ids = {m.strip("[]").lower() for m in _CITATION_RE.findall(raw_answer)}
    valid_ids = {c["chunk_id"].lower() for c in chunks}
    invalid_ids = cited_ids - valid_ids

    if invalid_ids:
        logger.warning("Answer cited invalid chunk IDs: %s", invalid_ids)
        raw_answer = "I cannot answer based on the provided documents."
        cited_ids = set()

    # Build Citation objects for IDs actually referenced
    chunk_map = {c["chunk_id"].lower(): c for c in chunks}
    citations = [
        Citation(
            chunk_id=cid,
            text=chunk_map[cid]["text"],
            source=chunk_map[cid]["source"],
            page=chunk_map[cid].get("page"),
        )
        for cid in cited_ids
        if cid in chunk_map
    ]

    # Confidence = mean rerank_score if available, else mean vector score
    scores = [c.get("rerank_score", c.get("score", 0.0)) for c in chunks]
    confidence = sum(scores) / len(scores) if scores else 0.0

    retrieval_method = chunks[0].get("retrieval_method", "hybrid") if chunks else "hybrid"

    return QueryResponse(
        question=question,
        answer=raw_answer,
        citations=citations,
        confidence=round(confidence, 4),
        retrieval_method=retrieval_method,
    )
