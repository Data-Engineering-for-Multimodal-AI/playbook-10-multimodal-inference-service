import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Info, generate_latest
from pydantic import BaseModel

from model import ModelManager, find_latest_image
from rlbench_data import DATA_ROOT, REPO_ROOT, frame_step

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The manifest to serve (for example manifest_candidate.yaml for the candidate version)
MANIFEST = os.environ.get("INFERENCE_MANIFEST", "manifest.yaml")

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_ERRORS = Counter('inference_errors_total', 'Inference requests that failed', ['status'])
REQUEST_LATENCY = Histogram('inference_request_duration_seconds', 'Inference request latency')
MODEL_INFO = Info('model_version', 'Model version information')


# Pydantic models for API
class InferenceRequest(BaseModel):
    rgb_path: str   # a frame (front_rgb/<step>.png) or a front_rgb folder (its latest frame)
    pkl_path: str   # the episode's low_dim_obs.pkl


class InferenceResponse(BaseModel):
    action: List[float]   # predicted change of the 7 joint positions (radians)
    model_version: str
    step: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    trained: bool


# Global model manager
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    global model_manager
    model_manager = ModelManager(os.path.join(REPO_ROOT, MANIFEST))
    config = model_manager.get_config()
    logger.info(f"Service started with model version {model_manager.get_version()}")

    # Set Prometheus model info
    MODEL_INFO.info({key: str(config[key]) for key in ('version', 'vision', 'telemetry', 'policy', 'fusion')})

    # Log the active manifest
    logger.info("Active manifest configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    yield


# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Inference Service",
    description="RLBench multimodal inference service for robotic action prediction",
    version="2026.10",
    lifespan=lifespan,
)


def data_path(path: str) -> str:
    """Resolve a client path; only files under the data folder (RLBENCH_DATA) are served.

    `..` can't leave the folder. Symlinks inside it are followed: whoever runs the service put them there.
    """
    resolved = os.path.abspath(os.path.join(REPO_ROOT, path))
    if os.path.commonpath([resolved, DATA_ROOT]) != DATA_ROOT:
        raise HTTPException(status_code=400, detail=f"Path outside the data folder: {path}")
    if not os.path.exists(resolved):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    return resolved


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return HealthResponse(status="healthy", model_version=model_manager.get_version(),
                          trained=model_manager.trained)


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Main inference endpoint.

    Takes a camera frame (or a front_rgb folder) and the episode's telemetry, and returns the
    predicted joint deltas for that frame. Runs in FastAPI's thread pool, so requests don't
    block each other.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    REQUEST_COUNT.inc()
    start_time = time.time()
    try:
        rgb_path = data_path(request.rgb_path)
        pkl_path = data_path(request.pkl_path)
        try:
            if os.path.isdir(rgb_path):
                # If it's a directory, use the latest timestep image
                rgb_path = find_latest_image(rgb_path)
                logger.info(f"Using latest timestep image: {rgb_path}")
            action = model_manager.get_model().predict(rgb_path, pkl_path)
        except (ValueError, OSError, pickle.UnpicklingError, EOFError) as e:
            # Report paths as the client sent them, not where the server keeps them
            raise HTTPException(status_code=422,
                                detail=f"Can't use these inputs: {str(e).replace(REPO_ROOT + os.sep, '')}")
        except Exception as e:
            logger.exception("Inference error")
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    except HTTPException as e:
        REQUEST_ERRORS.labels(status=str(e.status_code)).inc()
        raise

    # Record latency
    REQUEST_LATENCY.observe(time.time() - start_time)
    return InferenceResponse(action=action.tolist(), model_version=model_manager.get_version(),
                             step=frame_step(rgb_path))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def root():
    """Root endpoint with basic service information."""
    if model_manager is None:
        return {"status": "initializing"}
    return {
        "service": "Multimodal Inference Service",
        "version": model_manager.get_version(),
        "endpoints": {
            "inference": "/infer",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Set to True for development
    )
