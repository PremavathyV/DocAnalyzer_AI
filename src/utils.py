"""
utils.py — Shared helpers: logging setup, file-type detection, text cleaning.
"""

import logging
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}


def detect_file_type(filename: str) -> str:
    """
    Return a normalised file-type string based on the file extension.
    Raises ValueError for unsupported types.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return SUPPORTED_EXTENSIONS[ext]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove excessive whitespace / control characters while preserving
    paragraph breaks so downstream models receive readable input.
    """
    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
