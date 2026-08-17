FROM python:3.11-slim

WORKDIR /app

# System deps for pypdf / sentence-transformers (torch wheel needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ /app/
COPY frontend/ /frontend/

RUN mkdir -p data/uploads data/chroma

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
