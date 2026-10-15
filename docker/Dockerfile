# Multi-stage production Dockerfile for OmniSight-VLM
# Base image: NVIDIA CUDA 12.4 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies & Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY README.md pytest.ini ./

# Expose FastAPI Command Center port
EXPOSE 8000

# Default entrypoint: Launch FastAPI streaming server
CMD ["python", "src/api/app.py"]
