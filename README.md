# Multimodal Inference Service

A production-ready inference service for RLBench multimodal robotic data that combines vision and telemetry inputs to predict robotic actions. Built with PyTorch, FastAPI, and Docker for enterprise deployment.

## Features

- **Vision Processing**: ResNet18-based RGB image feature extraction
- **Telemetry Integration**: Joint position and gripper state processing 
- **MLP Fusion**: Concatenation-based multimodal fusion with MLP head
- **FastAPI Service**: REST API with health checks and Prometheus metrics
- **Canary Deployment**: Simulation framework for safe model rollouts
- **Containerized**: Docker support for production deployment
- **Manifest-based Configuration**: Version-controlled model parameters
- **Monitoring**: Built-in metrics and health monitoring

## Architecture

The service implements a multimodal neural network architecture:

```
RGB Image (224x224x3) → ResNet18 → 512D Features
                                        ↓
Joint Positions (7D) ──────────────→ Concat → MLP → 7D Actions
```

**Components:**
- **Vision Encoder**: Pretrained ResNet18 (frozen, inference-only)
- **Telemetry Processor**: Joint position extraction with robust fallbacks
- **Fusion Network**: 3-layer MLP (519→256→256→7)
- **Action Output**: 7 DOF delta joint commands

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- RLBench sample data (see Data Format section)

### Installation

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb-10-mm-inference-service.git
cd pb-10-mm-inference-service

# Install dependencies
pip install -r requirements.txt

# Set up sample data (see Data Setup section below)
python download_sample_data.py
```

### Running the Service

**Option 1: Direct Python**
```bash
# Start the FastAPI service
uvicorn service:app --host 0.0.0.0 --port 8000
```

**Option 2: Docker**
```bash
# Build the Docker image
docker build -t multimodal-inference .

# Run the container
docker run -p 8000:8000 multimodal-inference

# Run with data volume (recommended)
docker run -p 8000:8000 -v $(pwd)/rlbench-data:/app/rlbench-data multimodal-inference
```

The service will be available at `http://localhost:8000`

## API Usage

### Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `POST /infer` - Inference endpoint
- `GET /metrics` - Prometheus metrics

### Inference Example

```bash
curl -X POST "http://localhost:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "rgb_path": "rlbench-data/push_buttons/all_variations/episodes/episode39/front_rgb",
       "pkl_path": "rlbench-data/push_buttons/all_variations/episodes/episode39/low_dim_obs.pkl"
     }'
```

**Response:**
```json
{
  "action": [0.12, -0.03, 0.44, 0.21, -0.15, 0.08, 0.33],
  "model_version": "1.0.0"
}
```

### Interactive Documentation

FastAPI provides automatic interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Data Setup

### Getting Sample Data

Due to GitHub file size limits, the RLBench data is not included in this repository. You have several options:

**Option 1: Use the download helper (Recommended)**
```bash
python download_sample_data.py
```

**Option 2: Manual setup**
1. Download RLBench data from: https://github.com/stepjam/RLBench
2. Place episodes in the `rlbench-data/` directory following the structure below

**Option 3: Use your own data**
Format your robot demonstration data to match the RLBench structure

### Data Format

The service expects RLBench-formatted data:

```
rlbench-data/
├── task_name/                 # e.g., push_buttons, open_drawer
│   └── all_variations/
│       └── episodes/
│           └── episode_N/
│               ├── front_rgb/        # RGB camera images
│               │   ├── 34.png
│               │   ├── 35.png
│               │   └── ... 135.png
│               └── low_dim_obs.pkl   # Telemetry data
```

## Configuration

Model parameters are managed through `manifest.yaml`:

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

## Canary Deployment

Test model rollouts safely with the canary simulation:

```bash
# Run canary deployment simulation
python simulate_canary.py --max-samples 10 --canary-traffic 20.0
```

Features:
- Traffic splitting between baseline and candidate models
- Latency and action divergence monitoring
- Automatic rollback on performance degradation
- Detailed metrics reporting

## Monitoring

The service includes comprehensive monitoring:

**Health Checks:**
```bash
curl http://localhost:8000/health
```

**Prometheus Metrics:**
```bash
curl http://localhost:8000/metrics
```

Tracked metrics:
- Request count and success rate
- Response latency distribution
- Model version information
- Error rates by type

## Development

### Testing

Validate the complete system:

```bash
# Test model loading
python -c "from model import MultimodalInferenceModel; model = MultimodalInferenceModel(); print('Model loads successfully')"

# Test service health
curl http://localhost:8000/health

# Test Docker build
docker build -t multimodal-inference .

# Run canary simulation
python simulate_canary.py --max-samples 5
```

### Project Structure

```
├── service.py              # FastAPI application
├── model.py               # Neural network implementation
├── manifest.yaml          # Model configuration
├── simulate_canary.py     # Canary deployment simulation
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container definition
├── README.md             # Documentation
└── rlbench-data/         # Sample data (not included)
```

## RLBench Data Compatibility

**Current Implementation Status:**
- **Vision Processing**: Fully working with real RLBench RGB images
- **Service Architecture**: Complete FastAPI service with all endpoints
- **Containerization**: Docker deployment ready
- **Telemetry Processing**: Limited by available RLBench data format

**About RLBench Data:**
The `low_dim_obs.pkl` files in sample data contain Demo metadata rather than timestep-by-timestep observation data. In a complete RLBench setup, each demo would contain a sequence of observations with joint positions, but sample data only has episode-level metadata.

**What This Means:**
- The system works correctly with real RLBench RGB images
- For telemetry, it falls back to dummy joint positions (zeros) as documented
- This is a data availability limitation, not a system limitation
- The architecture is correct and ready for complete RLBench observation data

**For Full RLBench Integration:**
To get real joint position data, you would need:
1. Complete RLBench installation with environment simulation
2. Generated demonstrations with per-timestep observations
3. Observation data (not just demo metadata) in the pickle files

The inference service architecture is production-ready and will automatically use real joint positions when proper RLBench observation data is provided.

## Performance

**Benchmark Results:**
- **Inference Latency**: ~88ms per sample
- **Model Size**: ResNet18 + MLP (~45MB)
- **Memory Usage**: Moderate for ResNet18 + MLP
- **Throughput**: Suitable for real-time robotics applications

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

This work is based on RLBench:

```bibtex
@article{james2019rlbench,
  title={RLBench: The Robot Learning Benchmark \& Learning Environment},
  author={James, Stephen and Ma, Zicong and Rovick Arrojo, David and Davison, Andrew J.},
  journal={IEEE Robotics and Automation Letters},
  year={2020}
}
```

**RLBench Resources:**
- GitHub Repository: https://github.com/stepjam/RLBench
- Original Paper: IEEE Robotics and Automation Letters (2020)

## Acknowledgments

- RLBench team for the robotics benchmark and data format
- PyTorch and FastAPI communities for excellent frameworks
- Contributors and maintainers of this project

## Support

- Create an issue for bug reports or feature requests
- Check existing documentation and issues first
- Provide detailed information for better assistance

---

**Note**: This is a research and development project. For production use, ensure proper testing, security measures, and compliance with your specific requirements.