# Production Dockerfile for Data Center Predictive Cooling System
FROM python:3.11-slim

# Prevent Python from writing bytecode and enable unbuffered standard I/O
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KERAS_HOME=/app/.keras \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=5000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure runtime directories exist
RUN mkdir -p storage/uploads storage/models storage/predictions .keras

# Expose port
EXPOSE 5000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-5000}/health || exit 1

# Production server execution via Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 'backend.app:app'"]
