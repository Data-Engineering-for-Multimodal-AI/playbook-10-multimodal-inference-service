#!/usr/bin/env python3
"""
Download script for RLBench sample data.
This script downloads a subset of RLBench data for testing the multimodal inference service.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_file(url: str, filename: str):
    """Download a file with progress indication."""
    print(f"Downloading {filename}...")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
    
    urllib.request.urlretrieve(url, filename, progress_hook)
    print("\nDownload complete!")

def main():
    """Download and extract sample RLBench data."""
    
    # Create data directory
    data_dir = Path("rlbench-data")
    data_dir.mkdir(exist_ok=True)
    
    print("RLBench Sample Data Downloader")
    print("=" * 40)
    print("This script downloads sample episodes for testing the inference service.")
    print("Note: For the full RLBench dataset, visit: https://github.com/stepjam/RLBench")
    print()
    
    # Option 1: Download from a cloud storage (you would need to upload sample data here)
    # sample_url = "https://your-cloud-storage.com/rlbench-sample.zip"
    
    # Option 2: Instructions for manual download
    print("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. The full RLBench dataset is large (several GB)")
    print("2. For testing, you need at least one episode with:")
    print("   - RGB images in front_rgb/ directory")
    print("   - low_dim_obs.pkl file")
    print()
    print("Sample data structure needed:")
    print("rlbench-data/")
    print("├── push_buttons/")
    print("│   └── all_variations/")
    print("│       └── episodes/")
    print("│           └── episode39/")
    print("│               ├── front_rgb/")
    print("│               │   ├── 34.png")
    print("│               │   ├── 35.png")
    print("│               │   └── ... 135.png")
    print("│               └── low_dim_obs.pkl")
    print()
    print("You can:")
    print("a) Download RLBench data from: https://github.com/stepjam/RLBench")
    print("b) Use your own robot demonstration data in the same format")
    print("c) Contact the repository maintainers for sample data access")
    print()
    
    # Create minimal directory structure for demonstration
    demo_dir = data_dir / "push_buttons" / "all_variations" / "episodes" / "episode_demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a README in the data directory
    readme_content = """# RLBench Sample Data

This directory should contain RLBench demonstration data in the following format:

```
rlbench-data/
├── task_name/                 # e.g., push_buttons, open_drawer
│   └── all_variations/
│       └── episodes/
│           └── episode_N/
│               ├── front_rgb/        # RGB camera images
│               │   ├── 34.png       # Timestep images
│               │   ├── 35.png
│               │   └── ... 135.png
│               └── low_dim_obs.pkl   # Robot telemetry data
```

## Getting Sample Data

1. **Full RLBench Dataset**: https://github.com/stepjam/RLBench
2. **Custom Data**: Use your own robot demonstration data in this format
3. **Test Data**: Contact repository maintainers for access to sample episodes

## Usage

Once you have data in this directory, you can:

```bash
# Test the inference service
python -c "from model import MultimodalInferenceModel; model = MultimodalInferenceModel()"

# Start the service
uvicorn service:app --host 0.0.0.0 --port 8000

# Run inference
curl -X POST "http://localhost:8000/infer" \\
     -H "Content-Type: application/json" \\
     -d '{
       "rgb_path": "rlbench-data/push_buttons/all_variations/episodes/episode39/front_rgb",
       "pkl_path": "rlbench-data/push_buttons/all_variations/episodes/episode39/low_dim_obs.pkl"
     }'
```
"""
    
    with open(data_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Created data directory structure in: {data_dir.absolute()}")
    print(f"See {data_dir}/README.md for detailed instructions")

if __name__ == "__main__":
    main()