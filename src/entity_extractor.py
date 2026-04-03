"""
entity_extractor.py — Named Entity Recognition using spaCy.

Extracts:
  - PERSON       → people
  - ORG          → organizations
  - GPE / LOC    → locations
  - DATE / TIME  → dates
  - MONEY        → money
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

from src.utils import get_logger

logger = get_logger(__name__)

# spaCy label → output key mapping
_LABEL_MAP: Dict[str, str] = {
    "PERSON": "people",
    "ORG": "organizations",
    "GPE": "locations",
    "LOC": "locations",
    "DATE": "dates",
    "TIME": "dates",
    "MONEY": "money",
}

_EMPTY_RESULT: Dict[str, List[str]] = {
    "people": [],
    "organizations": [],
    "locations": [],
    "dates": [],
    "money": [],
}


@lru_cache(maxsize=1)
def _load_nlp():
    """Lazy-load the spaCy model (cached after first call)."""
    import spacy  # deferred import

    model = "en_core_web_sm"
    logger.info("Loading spaCy model '%s'…", model)
    try:
        return spacy.load(model)
    except OSError:
        logger.warning(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            model,
            model,
        )
        raise RuntimeError(
            f"spaCy model '{model}' is not installed. "
            f"Run: python -m spacy download {model}"
        )


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from *text*.

    Uses spaCy NER + regex fallbacks for money and dates to maximise recall.

    Returns:
        Dict with keys: people, organizations, locations, dates, money.
        Each value is a deduplicated list of entity strings.
    """
    import re

    if not text or not text.strip():
        return dict(_EMPTY_RESULT)

    try:
        nlp = _load_nlp()
        chunk = text[:100_000]
        doc = nlp(chunk)

        result: Dict[str, List[str]] = {k: [] for k in _EMPTY_RESULT}

        for ent in doc.ents:
            key = _LABEL_MAP.get(ent.label_)
            if key:
                value = ent.text.strip()
                if value and value not in result[key]:
                    result[key].append(value)

        # ── Regex fallbacks for money & dates (improves recall) ──────────
        # Money: $1,000 / $1.5M / ₹50,000 / USD 200 / 500 dollars
        money_pattern = re.compile(
            r'(?:USD|INR|EUR|GBP|₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|thousand|M|B|K))?'
            r'|\d[\d,]*(?:\.\d+)?\s?(?:dollars?|euros?|rupees?|pounds?)',
            re.IGNORECASE,
        )
        for m in money_pattern.findall(text):
            v = m.strip()
            if v and v not in result["money"]:
                result["money"].append(v)

        # Dates: 12/03/2024, 3 Jan 2024, January 2024, 2024-01-15
        date_pattern = re.compile(
            r'\b(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}'
            r'|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}'
            r'|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}'
            r'|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}'
            r'|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
            re.IGNORECASE,
        )
        for m in date_pattern.findall(text):
            v = m.strip()
            if v and v not in result["dates"]:
                result["dates"].append(v)

        logger.info(
            "Entity extraction complete — %s",
            {k: len(v) for k, v in result.items()},
        )
        return result

    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Entity extraction failed: %s", exc)
        return dict(_EMPTY_RESULT)
