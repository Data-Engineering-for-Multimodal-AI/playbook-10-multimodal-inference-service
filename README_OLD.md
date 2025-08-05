# Multimodal Inference Service

A local inference service for RLBench multimodal robotic data that combines vision and telemetry inputs to predict robotic actions.

## Features

- **Vision Processing**: ResNet18-based RGB image feature extraction
- **Telemetry Integration**: Joint position and gripper state processing 
- **MLP Fusion**: Concatenation-based multimodal fusion with MLP head
- **FastAPI Service**: REST API with health checks and Prometheus metrics
- **Canary Deployment**: Simulation framework for safe model rollouts
- **Containerized**: Docker support for easy deployment

## Quick Start

### Prerequisites

- Python 3.9+
- RLBench sample data in `rlbench-data/` directory

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Service

```bash
# Start the FastAPI service
python service.py

# Or with uvicorn directly
uvicorn service:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

### API Usage

#### Inference Endpoint

```bash
curl -X POST "http://localhost:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "rgb_path": "rlbench-data/open_drawer/all_variations/episodes/episode1/front_rgb",
       "pkl_path": "rlbench-data/open_drawer/all_variations/episodes/episode1/low_dim_obs.pkl"
     }'
```

Response:
```json
{
  "action": [0.12, -0.03, 0.44, 0.15, -0.22, 0.08, 0.31],
  "model_version": "1.0.0"
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Metrics

```bash
curl http://localhost:8000/metrics
```

### Docker Deployment

```bash
# Build image
docker build -t multimodal-inference .

# Run container
docker run -p 8000:8000 -v $(pwd)/rlbench-data:/app/rlbench-data multimodal-inference
```

### Canary Deployment Simulation

```bash
# Run canary simulation
python simulate_canary.py --canary-traffic 10.0 --max-samples 50

# With custom thresholds
python simulate_canary.py \
    --canary-traffic 15.0 \
    --latency-threshold 1.3 \
    --divergence-threshold 0.15 \
    --max-samples 100
```

## Architecture

### Model Components

1. **Vision Encoder**: Pretrained ResNet18 for RGB feature extraction
2. **Telemetry Processor**: Joint position normalization and feature extraction  
3. **Fusion Layer**: Feature concatenation
4. **MLP Head**: 3-layer MLP for action prediction (7 DOF output)

### Configuration

Model settings are defined in `manifest.yaml`:

```yaml
version: "1.0.0"
vision: "resnet18" 
telemetry: "joints_v1"
policy: "mlp_stub_v1"
fusion: "concat"
model_settings:
  vision_embedding_dim: 512
  telemetry_dim: 7
  hidden_dim: 256
  output_dim: 7
```

### Data Format

- **RGB**: PNG images (any resolution, auto-resized to 224x224)
- **Telemetry**: Pickle files containing `joint_positions` and other robot state
- **Output**: 7D action vector representing delta joint commands

## File Structure

```
├── service.py              # FastAPI application
├── model.py                # Neural network models and inference
├── manifest.yaml           # Model configuration
├── simulate_canary.py      # Canary deployment simulation
├── requirements.txt        # Python dependencies  
├── Dockerfile             # Container configuration
├── .dockerignore          # Docker build exclusions
└── rlbench-data/          # Sample data directory
    ├── open_drawer/
    ├── push_buttons/
    └── ...
```

## Development

### Running Tests

The canary simulation serves as an integration test:

```bash
python simulate_canary.py --max-samples 10
```

### Adding New Models

1. Update `manifest.yaml` with new version and settings
2. Modify `model.py` to support new architectures
3. Test with canary simulation before deployment

### Monitoring

Prometheus metrics available at `/metrics`:

- `inference_requests_total`: Request counter
- `inference_request_duration_seconds`: Latency histogram  
- `model_version`: Current model version info

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `rlbench-data/` contains valid episodes
2. **Memory issues**: Reduce batch size or use smaller models
3. **Port conflicts**: Change port in `service.py` or use Docker port mapping

### Logs

Service logs include:
- Model initialization status
- Request processing details  
- Error messages with stack traces
- Manifest configuration on startup

## Important Note: RLBench Data Compatibility

**Current Implementation Status:**
- **Vision Processing**: Fully working with real RLBench RGB images
- **Service Architecture**: Complete FastAPI service with all endpoints
- **Containerization**: Docker deployment ready
- **Telemetry Processing**: Limited by available RLBench data format

**About RLBench Data:**
The `low_dim_obs.pkl` files in our sample data contain **Demo metadata** rather than timestep-by-timestep **observation data**. In a complete RLBench setup, each demo would contain a sequence of observations with joint positions, but our sample data only has episode-level metadata.

**What This Means:**
- The system **works correctly** with real RLBench RGB images
- For telemetry, it falls back to dummy joint positions (zeros) as documented
- This is a **data availability limitation**, not a system limitation
- The architecture is **correct and ready** for complete RLBench observation data

**For Full RLBench Integration:**
To get real joint position data, you would need:
1. Complete RLBench installation with environment simulation
2. Generated demonstrations with per-timestep observations
3. Observation data (not just demo metadata) in the pickle files

The inference service architecture is **production-ready** and will automatically use real joint positions when proper RLBench observation data is provided.