"""Split document pages into overlapping chunks with metadata."""
import hashlib
import logging
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split pages into overlapping chunks.

    Each chunk dict has: chunk_id, text, source, page.
    chunk_id is a deterministic SHA-1 hex digest of source+page+offset.
    """
    chunks: List[Dict[str, Any]] = []
    for page in pages:
        splits = _splitter.split_text(page["text"])
        for idx, text in enumerate(splits):
            raw_id = f"{page['source']}:{page['page']}:{idx}"
            chunk_id = "chunk-" + hashlib.sha1(raw_id.encode()).hexdigest()[:8]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    logger.info("Created %d chunks from %d pages", len(chunks), len(pages))
    return chunks
