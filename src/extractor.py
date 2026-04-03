
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Premavathy\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
"""
extractor.py — Multi-format text extraction.

Supports:
  - PDF  → PyMuPDF (fitz)
  - DOCX → python-docx
  - Image → pytesseract OCR
"""

import io
import os
from typing import Union

import fitz  # PyMuPDF
import pytesseract
from docx import Document
from PIL import Image, ImageFilter

from src.utils import clean_text, get_logger

logger = get_logger(__name__)

# Allow overriding the Tesseract binary path via env var (useful on Windows)
_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf(data: bytes) -> str:
    """Extract text from a PDF byte stream using PyMuPDF."""
    logger.info("Extracting text from PDF")
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(pages)


def _extract_docx(data: bytes) -> str:
    """Extract text from a DOCX byte stream using python-docx."""
    logger.info("Extracting text from DOCX")
    doc = Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def _extract_image(data: bytes) -> str:
    return "OCR not supported in deployed version"

    try:
        print("Using Tesseract at:", pytesseract.pytesseract.tesseract_cmd)

        image = Image.open(io.BytesIO(data)).convert("L")
        image = image.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(
            image,
            config="--psm 6",
            timeout=5
        )

        print("OCR DONE")
        return text

    except Exception as e:
        print("OCR ERROR:", str(e))
        raise RuntimeError(f"OCR failed: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(data: bytes, file_type: str) -> str:
    """
    Extract raw text from document bytes.

    Args:
        data:      Raw file bytes.
        file_type: One of 'pdf', 'docx', 'image'.

    Returns:
        Cleaned extracted text string.

    Raises:
        ValueError: For unsupported file types.
        RuntimeError: If extraction fails.
    """
    extractors = {
        "pdf": _extract_pdf,
        "docx": _extract_docx,
        "image": _extract_image,
    }

    if file_type not in extractors:
        raise ValueError(f"No extractor available for file type: {file_type}")

    try:
        raw = extractors[file_type](data)
        cleaned = clean_text(raw)
        if not cleaned:
            raise RuntimeError("Extraction produced empty text. Check the document content.")
        logger.info("Extraction complete — %d characters", len(cleaned))
        return cleaned
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"Text extraction failed: {exc}") from exc
