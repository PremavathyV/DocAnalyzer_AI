"""
summarizer.py — AI-powered text summarization.

Uses HuggingFace's facebook/bart-large-cnn by default.
Falls back to a simple extractive summary if the model is unavailable.
"""

from __future__ import annotations

from functools import lru_cache

from src.utils import get_logger

logger = get_logger(__name__)

# Maximum input tokens the summarisation model can handle
_MAX_INPUT_CHARS = 3000  # ~1024 tokens; trim longer docs before sending


@lru_cache(maxsize=1)
def _load_summarizer():
    """Lazy-load the summarisation pipeline (cached after first call)."""
    from transformers import pipeline  # deferred import — heavy load

    logger.info("Loading summarisation model (facebook/bart-large-cnn)…")
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn",
    )


def _extractive_fallback(text: str, max_sentences: int = 3) -> str:
    """
    Simple extractive fallback: return the first N sentences.
    Used when the transformer model cannot be loaded.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:max_sentences])


def summarize_text(text: str) -> str:
    """
    Generate a concise abstractive summary of *text*.

    Args:
        text: The document text to summarise.

    Returns:
        A summary string.
    """
    if not text or not text.strip():
        raise ValueError("Cannot summarise empty text.")

    # Trim to model's comfortable input window
    trimmed = text[:_MAX_INPUT_CHARS]

    try:
        summarizer = _load_summarizer()
        # Dynamic length bounds: summary should be 20–130 tokens
        max_len = min(130, max(30, len(trimmed.split()) // 4))
        min_len = min(20, max_len - 5)

        result = summarizer(
            trimmed,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
        )
        summary = result[0]["summary_text"].strip()
        logger.info("Summarisation complete (%d chars)", len(summary))
        return summary

    except Exception as exc:
        logger.warning("Transformer summariser failed (%s); using extractive fallback.", exc)
        return _extractive_fallback(text)
