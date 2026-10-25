# CPU image for the multimodal inference service
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# curl for the health check
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching (CPU-only torch on Linux; see requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY manifest.yaml manifest_candidate.yaml manifest_candidate_no_camera.yaml ./
COPY rlbench_data.py model.py train.py service.py ./

# Create non-root user for security
RUN useradd -m -u 1000 inference && \
    chown -R inference:inference /app
USER inference

# Bake the ResNet-18 weights into the image, so the container needs no network at startup
ENV TORCH_HOME=/home/inference/.cache/torch
RUN python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# The data and trained weights are mounted at run time:
#   docker run -p 8000:8000 -v $(pwd)/rlbench-data:/app/rlbench-data:ro -v $(pwd)/models:/app/models:ro multimodal-inference
# INFERENCE_MANIFEST=manifest_candidate.yaml serves the candidate instead.
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
