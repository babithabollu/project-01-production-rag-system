"""Load and cache versioned prompt templates from YAML."""
import logging
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)

_templates: Dict[str, Any] = {}
_PROMPTS_PATH = Path("prompts/rag_prompts.yaml")


def load_templates() -> None:
    """Load prompt templates from YAML file into module cache."""
    global _templates
    if not _PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Prompts file not found: {_PROMPTS_PATH}")
    with _PROMPTS_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    _templates = data.get("templates", {})
    logger.info(
        "Loaded %d prompt templates (version %s)",
        len(_templates),
        data.get("version", "unknown"),
    )


def get_template(name: str) -> Dict[str, str]:
    """
    Return a named template dict with 'system' and 'user' keys.
    Raises KeyError if template not found.
    """
    if not _templates:
        load_templates()
    return _templates[name]
