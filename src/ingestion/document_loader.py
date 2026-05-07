"""Load PDF, Markdown, and plain-text documents into raw text + metadata."""
import logging
from pathlib import Path
from typing import List, Dict, Any

import pypdf

logger = logging.getLogger(__name__)


def load_document(file_path: str, content: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text pages from a document.

    Returns a list of dicts with keys: text, source, page.
    Supports .pdf, .md, .txt extensions.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(content, filename)
    elif suffix in {".md", ".txt"}:
        return _load_text(content, filename)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Extract pages from a PDF binary."""
    import io
    reader = pypdf.PdfReader(io.BytesIO(content))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": filename, "page": i + 1})
    logger.info("Loaded %d pages from %s", len(pages), filename)
    return pages


def _load_text(content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Load plain text or markdown as a single page."""
    text = content.decode("utf-8", errors="replace")
    return [{"text": text, "source": filename, "page": 1}]
