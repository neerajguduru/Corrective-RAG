FROM python:3.12-slim

# System deps: tesseract for OCR of scanned PDFs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/

# Uploaded/extracted files + vector store live here (mount as volume)
RUN mkdir -p data/documents data/eval

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
