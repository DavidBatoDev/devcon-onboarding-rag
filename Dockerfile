FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp1: runtime lib some scientific wheels (onnxruntime via chromadb) expect
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app, including the prebuilt index under app/storage/
COPY . .

# Cloud Run injects PORT (default 8080); main.py reads it and binds 0.0.0.0
EXPOSE 8080
CMD ["python", "main.py"]
