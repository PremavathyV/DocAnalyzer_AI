# Document Analysis & Extraction API

An AI-powered backend that extracts, analyzes, and summarizes content from PDF, DOCX

---

## Features

- Multi-format support: PDF, DOCS
- Text extraction: PyMuPDF (PDF), python-docx (DOCX), Tesseract OCR (images)
- AI summarization: `facebook/bart-large-cnn` via HuggingFace Transformers
- Named entity recognition: spaCy `en_core_web_sm` — people, orgs, locations, dates, money
- Sentiment analysis: `distilbert-base-uncased-finetuned-sst-2-english`
- Async FastAPI backend with clean modular architecture

---

## Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI + Uvicorn |
| PDF | PyMuPDF (fitz) |
| DOCX | python-docx |
| OCR | pytesseract + Pillow |
| Summarization | HuggingFace Transformers (BART) |
| NER | spaCy |
| Sentiment | HuggingFace Transformers (DistilBERT) |

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Install Tesseract OCR

- **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`
- **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  Then set `TESSERACT_CMD` in your `.env` file.

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (e.g. set TESSERACT_CMD on Windows)
```

---

## Running the API

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at: http://localhost:8000/docs

---

## API Usage

### Health check

```bash
curl http://localhost:8000/
```

### Analyze a document

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/document.pdf"
```

### Example response

```json
{
  "summary": "The document outlines quarterly financial results showing a 12% revenue increase...",
  "entities": {
    "people": ["John Smith", "Jane Doe"],
    "organizations": ["Acme Corp", "SEC"],
    "locations": ["New York", "California"],
    "dates": ["January 2024", "Q3 2023"],
    "money": ["$1.2 million", "$500,000"]
  },
  "sentiment": "positive"
}
```

---

## Project Structure

```
├── README.md
├── requirements.txt
├── .env.example
└── src/
    ├── main.py             # FastAPI app & /analyze endpoint
    ├── extractor.py        # PDF / DOCX / image text extraction
    ├── summarizer.py       # HuggingFace BART summarization
    ├── entity_extractor.py # spaCy NER
    ├── sentiment.py        # HuggingFace DistilBERT sentiment
    └── utils.py            # Logging, file-type detection, text cleaning
```

---

## Deployment (Render / Railway / EC2)

1. Push repo to GitHub.
2. Set environment variables (`TESSERACT_CMD` if on Windows host, `AI_BACKEND`, etc.).
3. Set start command: `uvicorn src.main:app --host 0.0.0.0 --port 8000`
4. Ensure Tesseract is installed on the server (`apt install tesseract-ocr`).

For Docker deployment, add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
