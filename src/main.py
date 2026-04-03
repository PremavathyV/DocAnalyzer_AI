"""
main.py — FastAPI entry point for the Document Analysis & Extraction API.

Endpoint:
  POST /analyze
    - Accepts a multipart file upload
    - Returns summary, entities, and sentiment as JSON
"""

import io
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from src.entity_extractor import extract_entities
from src.extractor import extract_text
from src.sentiment import analyze_sentiment
from src.summarizer import summarize_text
from src.utils import detect_file_type, get_logger

load_dotenv()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# API Key auth
# ---------------------------------------------------------------------------
# Set API_KEY in your .env file. If not set, a random one is generated and
# printed at startup — copy it for your submission form.
_API_KEY: str = os.getenv("API_KEY") or secrets.token_hex(16)


def verify_api_key(x_api_key: str = Header(default=None)):
    """Dependency: validate X-API-Key header."""
    if not x_api_key or not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass it as header: X-API-Key")


# ---------------------------------------------------------------------------
# App lifecycle — pre-warm models on startup (optional but speeds up first req)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Document Analysis API…")
    logger.info("=" * 50)
    logger.info("  API KEY: %s", _API_KEY)
    logger.info("  Pass as header → X-API-Key: %s", _API_KEY)
    logger.info("=" * 50)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Document Analysis & Extraction API",
    description="Upload PDF, DOCX, or image files to extract text, entities, summary, and sentiment.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (frontend UI)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def ui():
    """Serve the frontend UI."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h2>UI not found. Place index.html in src/static/</h2>")


@app.post("/analyze", tags=["Analysis"])
async def analyze(file: UploadFile = File(...)):
    """
    Analyze an uploaded document.

    Accepts: PDF, DOCX, PNG, JPG, JPEG, TIFF, BMP, WEBP

    Returns:
        summary      — AI-generated abstractive summary
        entities     — Named entities (people, orgs, locations, dates, money)
        sentiment    — positive | negative | neutral
    """
    # ── 1. Validate file type ──────────────────────────────────────────────
    try:
        file_type = detect_file_type(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    # ── 2. Read file bytes ─────────────────────────────────────────────────
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info("Received file '%s' (%d bytes, type=%s)", file.filename, len(data), file_type)

    # ── 3. Extract text ────────────────────────────────────────────────────
    try:
        text = extract_text(data, file_type)
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {exc}")

    # ── 4. Run AI pipeline (summarise, entities, sentiment) ────────────────
    try:
        summary = summarize_text(text)
    except Exception as exc:
        logger.error("Summarisation error: %s", exc)
        summary = "Summary unavailable."

    try:
        entities = extract_entities(text)
    except Exception as exc:
        logger.error("Entity extraction error: %s", exc)
        entities = {"people": [], "organizations": [], "locations": [], "dates": [], "money": []}

    try:
        sentiment = analyze_sentiment(text)
    except Exception as exc:
        logger.error("Sentiment error: %s", exc)
        sentiment = "neutral"

    # ── 5. Return structured response ──────────────────────────────────────
    return JSONResponse(content={
        "summary": summary,
        "entities": entities,
        "sentiment": sentiment,
        "text": text,          # needed for entity highlighting in UI
    })


@app.post("/download/json", tags=["Export"])
async def download_json(file: UploadFile = File(...)):
    """Re-analyze and return result as a downloadable JSON file."""
    try:
        file_type = detect_file_type(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))

    data = await file.read()
    text = extract_text(data, file_type)
    result = {
        "summary": summarize_text(text),
        "entities": extract_entities(text),
        "sentiment": analyze_sentiment(text),
        "text": text,
    }
    json_bytes = json.dumps(result, indent=2).encode("utf-8")
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=analysis_result.json"},
    )


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/")
def home():
    return {"message": "API running"}