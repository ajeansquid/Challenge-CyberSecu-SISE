# Dockerfile for CyberSec ML Toolkit
# Challenge SISE-OPSIE 2026

FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better layer caching
COPY pyproject.toml .
# Install dependencies (no project itself yet — faster cache layer)
RUN uv sync --no-install-project

# Copy application code
COPY . .

# Final sync to install the project itself
RUN uv sync

# Expose Streamlit default port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit via uv
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
