from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import logging
import time
from typing import List, Dict
import os
from prometheus_client import Counter, Histogram, Info, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import Response

from model import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total number of inference requests')
REQUEST_LATENCY = Histogram('inference_request_duration_seconds', 'Inference request latency')
MODEL_INFO = Info('model_version', 'Model version information')

# Pydantic models for API
class InferenceRequest(BaseModel):
    rgb_path: str
    pkl_path: str

class InferenceResponse(BaseModel):
    action: List[float]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Inference Service",
    description="RLBench multimodal inference service for robotic action prediction",
    version="1.0.0"
)

# Global model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global model_manager
    
    try:
        model_manager = ModelManager("manifest.yaml")
        logger.info(f"Service started with model version {model_manager.get_version()}")
        
        # Set Prometheus model info
        MODEL_INFO.info({
            'version': model_manager.get_version(),
            'vision': model_manager.get_config()['vision'],
            'telemetry': model_manager.get_config()['telemetry'],
            'policy': model_manager.get_config()['policy'],
            'fusion': model_manager.get_config()['fusion']
        })
        
        # Log the active manifest
        config = model_manager.get_config()
        logger.info("Active manifest configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise e

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return HealthResponse(
        status="healthy",
        model_version=model_manager.get_version()
    )

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    """
    Main inference endpoint.
    
    Takes RGB image path and pickle file path, returns predicted action vector.
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        # Validate input paths
        if not os.path.exists(request.rgb_path):
            raise HTTPException(status_code=400, detail=f"RGB path not found: {request.rgb_path}")
        
        if not os.path.exists(request.pkl_path):
            raise HTTPException(status_code=400, detail=f"Pickle path not found: {request.pkl_path}")
        
        # Handle directory paths for RGB (timestep sequences)
        rgb_path = request.rgb_path
        if os.path.isdir(rgb_path):
            # If it's a directory, find the latest timestep image
            from model import find_latest_image
            rgb_path = find_latest_image(rgb_path)
            logger.info(f"Using latest timestep image: {rgb_path}")
        
        # Run inference
        model = model_manager.get_model()
        action = model.predict(rgb_path, request.pkl_path)
        
        # Record latency
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        return InferenceResponse(
            action=action.tolist(),
            model_version=model_manager.get_version()
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get("/")
async def root():
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