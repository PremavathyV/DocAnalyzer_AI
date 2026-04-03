"""
sentiment.py — Sentiment analysis using a HuggingFace transformer.

Model: distilbert-base-uncased-finetuned-sst-2-english
  - Outputs POSITIVE / NEGATIVE
  - We map scores near 0.5 to NEUTRAL for nuance.
"""

from __future__ import annotations

from functools import lru_cache

from src.utils import get_logger

logger = get_logger(__name__)

# Input is trimmed to this length before sending to the model
_MAX_INPUT_CHARS = 512

# If the winning label's confidence is below this threshold → neutral
_NEUTRAL_THRESHOLD = 0.65


@lru_cache(maxsize=1)
def _load_sentiment_pipeline():
    """Lazy-load the sentiment pipeline (cached after first call)."""
    from transformers import pipeline  # deferred import

    logger.info("Loading sentiment model (distilbert-base-uncased-finetuned-sst-2-english)…")
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


def analyze_sentiment(text: str) -> str:
    """
    Classify the overall sentiment of *text*.

    Args:
        text: Document text.

    Returns:
        One of: 'positive', 'negative', 'neutral'.
    """
    if not text or not text.strip():
        raise ValueError("Cannot analyse sentiment of empty text.")

    trimmed = text[:_MAX_INPUT_CHARS]

    try:
        classifier = _load_sentiment_pipeline()
        result = classifier(trimmed)[0]

        label: str = result["label"].lower()   # 'positive' or 'negative'
        score: float = result["score"]

        # Treat low-confidence predictions as neutral
        if score < _NEUTRAL_THRESHOLD:
            sentiment = "neutral"
        else:
            sentiment = label  # already 'positive' or 'negative'

        logger.info("Sentiment: %s (score=%.3f)", sentiment, score)
        return sentiment

    except Exception as exc:
        logger.error("Sentiment analysis failed: %s", exc)
        raise RuntimeError(f"Sentiment analysis failed: {exc}") from exc
