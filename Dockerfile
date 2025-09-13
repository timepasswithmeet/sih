# Multi-stage Docker build for ATC project
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p artifacts data/synthetic/images

# Production stage
FROM base as production

WORKDIR /app

# Copy requirements and install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    detectron2>=0.6 \
    opencv-python>=4.8.0 \
    pillow>=9.5.0 \
    numpy>=1.24.0 \
    streamlit>=1.28.0 \
    pycocotools>=2.0.6 \
    pyyaml>=6.0 \
    scikit-learn>=1.3.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    tqdm>=4.65.0 \
    onnx>=1.14.0 \
    onnxruntime>=1.15.0

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY assets/ ./assets/

# Create necessary directories
RUN mkdir -p artifacts data/synthetic/images

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
