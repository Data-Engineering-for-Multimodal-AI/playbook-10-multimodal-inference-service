# Multimodal Inference Service

**Chapter 10 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026). Page numbers to be confirmed.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## What you'll build

An inference service for a robot arm that combines two modalities: what the front camera sees and where the seven joints are. You'll serve it with versioned manifests, watch it with Prometheus metrics, and roll out new model versions with a canary that can roll them back.

- **A multimodal model trained on real demonstrations**:
  - A frozen ResNet-18 embeds the camera frame (512 numbers), and the joint positions add 7 more.
  - A fusion MLP predicts the joint change the expert made next.
  - It is trained on 20 RLBench `push_buttons` demonstrations and tested on 5 others.
- **A FastAPI service**:
  - `/infer`, `/health` and Prometheus `/metrics`.
  - Inputs are restricted to the data folder, and the telemetry loader refuses pickles that aren't RLBench data.
  - A Docker image runs it as a non-root user.
- **Versioned manifests**: each version is a YAML file naming its inputs, architecture, training split and weights.
- **Canary rollouts with real verdicts**:
  - A share of the traffic goes to the candidate, and the baseline answers the same requests in shadow.
  - The two are compared on latency, errors, and distance from what the expert actually did.
  - Two candidates ship. A wider network (1.1.0) passes. A version that drops the camera to save latency (1.2.0) is 5 times faster but gets rolled back.

## Quick start

You need Python 3.12 or newer and about 1 GB of disk space: the environment (700 MB) and the data. The data downloads as 349 MB, of which 78 MB is kept.

```bash
# Clone the repository
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-10-multimodal-inference-service.git
cd playbook-10-multimodal-inference-service

# Create and activate a virtual environment, then install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download the RLBench demonstrations (25 episodes of push_buttons)
python download_sample_data.py
```

With [uv](https://docs.astral.sh/uv/): `uv sync`, then prefix commands with `uv run`. The Makefile has shortcuts: `make train`, `make serve`, `make canary`, `make canary-no-camera`, `make smoke`.

### 1. Train the model versions

```bash
python train.py
```

This trains the three manifests: `manifest.yaml` (1.0.0), `manifest_candidate.yaml` (1.1.0) and `manifest_candidate_no_camera.yaml` (1.2.0).

- It embeds about 2,500 camera frames once, cached in `.cache/`, then trains each fusion head.
- It takes 2 to 3 minutes on a laptop CPU and downloads ResNet-18 (45 MB).
- The weights go to `models/`.

The service and the canary train any missing version themselves, so you can skip this step.

### 2. Serve it

```bash
uvicorn service:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST "http://localhost:8000/infer" \
     -H "Content-Type: application/json" \
     -d '{
       "rgb_path": "rlbench-data/push_buttons/all_variations/episodes/episode21/front_rgb/10.png",
       "pkl_path": "rlbench-data/push_buttons/all_variations/episodes/episode21/low_dim_obs.pkl"
     }'
```

- `rgb_path` is a camera frame, or a `front_rgb` folder (meaning its latest frame).
- `pkl_path` is the episode's telemetry. The service reads the joint positions at that frame.
- The response is the predicted change of the 7 joints in radians, plus the model version and the frame used.
- Paths must be inside `rlbench-data/` (or `RLBENCH_DATA`); anything else gets a 400.
- `INFERENCE_MANIFEST=manifest_candidate.yaml uvicorn service:app` serves a different version.

Interactive docs are at http://localhost:8000/docs. The metrics are at `/metrics`:

- `inference_requests_total`
- `inference_errors_total` by status
- `inference_request_duration_seconds`
- `model_version_info`

### 3. Roll out a new version

```bash
python simulate_canary.py --max-samples 400 --canary-traffic 20
python simulate_canary.py --candidate manifest_candidate_no_camera.yaml --max-samples 400 --canary-traffic 20
```

The canary replays requests from the 5 held-out demonstrations and sends a share of them to the candidate (seeded, so runs repeat). The baseline answers the candidate's requests too, in shadow. It rolls back as soon as the candidate:

- fails more often (success rate below 95% of the baseline's), or
- is slower: p95 latency over 1.5 times the baseline's on the same requests, or
- moves further from what the expert did: error more than 10% above the baseline's, or
- drifts from the baseline's actions (divergence above 0.2).

If none of these happen, and at least 10 requests went to the candidate, it can be promoted.

Every check has a command-line option (`--help`). The printed `python simulate_canary.py --max-samples 10 --canary-traffic 20.0` still runs. It sends only one or two requests to the candidate, so it reports that there isn't enough traffic to decide.

### 4. Docker

```bash
docker build -t multimodal-inference .
docker run -p 8000:8000 -v $(pwd)/rlbench-data:/app/rlbench-data:ro -v $(pwd)/models:/app/models:ro multimodal-inference
```

- The image is Python 3.12 with CPU-only PyTorch (1.5 GB); ResNet-18's weights are inside it.
- The demonstrations and trained weights are mounted, so train first.
- The container runs as the non-root user `inference` and has a health check.

### Tests

```bash
make smoke    # or: uv run pytest -q
```

The tests check:

- the safe loader rejects a malicious pickle
- the data and the training scores
- every service endpoint, including rejected paths and files
- both canary verdicts and the printed canary command

CI runs them on Python 3.12 and 3.13. It also runs the printed `pip install -r requirements.txt`, `uvicorn` and `curl` commands, and builds and queries the Docker image.

## Tested on

- 2026-10-25: macOS 26.5.1 (Apple M3, 8 GB), Python 3.12.15, torch 2.14.1, FastAPI 0.162.0, Docker 28.0.1.
- Ubuntu (GitHub Actions) with Python 3.12 and 3.13: see the `smoke` workflow.

## Differences from the printed book

The code as printed is tagged `first-edition`. What changed and why:

- **The model is trained.**
  - In the first edition, the fusion head kept its random starting weights: there was no training step, so actions were random numbers and changed every time the service restarted.
  - `train.py` now trains it by behaviour cloning on real demonstrations.
  - The weights file named in the manifest makes a version reproducible.
- **The telemetry is read.**
  - The joint positions are inside each `low_dim_obs.pkl` (`Demo._observations[i].joint_positions`). The first edition's loader skipped attributes starting with `_`, so every request used zeros.
  - The README of the time called this a data limitation. It was a bug.
- **Telemetry and frame match.** The service reads the joint positions at the frame it was given, not always the last ones.
- **Safer inputs.**
  - `/infer` opened any path on the server and unpickled it, so a crafted pickle could run code.
  - Paths are now limited to the data folder, and the loader accepts only RLBench and NumPy classes.
  - Errors return 400, 404 or 422 instead of 500.
- **The canary compares trained models on real outcomes.**
  - The first edition compared two untrained networks, so its rollback decisions were noise.
  - Now the comparison is per request against the baseline in shadow, scored against the expert's actions, and seeded.
  - Added: the `--candidate`, `--quality-tolerance`, `--min-samples` and `--seed` options, and a no-camera candidate.
  - The candidate manifest is a committed file now rather than generated on each run.
- **Current APIs and runtime.**
  - FastAPI lifespan replaces `@app.on_event`, and `weights=` replaces torchvision's `pretrained=True`.
  - `/metrics` sends the Prometheus text content type (it sent the OpenMetrics type with text content).
  - Inference runs in FastAPI's thread pool.
  - Docker: Python 3.9 reached end of life in October 2025. The image is now Python 3.12 with CPU-only PyTorch, with the weights baked in.
- **Data.** `download_sample_data.py` now downloads and verifies the demonstrations; it used to create an empty folder.
- **The curl example.** It now uses `episode21`. The first edition's `episode39` isn't in the 25-episode sample; it is in the 100-episode `train` split of the same dataset.
- **Manifests.**
  - `manifest.yaml` gains `weights` and `training` sections.
  - `policy` changes from `mlp_stub_v1` to `mlp_bc_v1` (behaviour cloning).
- **Packaging.** Python 3.12+, `pyproject.toml`, `uv.lock` and a pinned `requirements.txt` (CPU PyTorch on Linux), plus smoke tests, CI and a Makefile.
- **Removed.**
  - `README_GITHUB.md` and `README_OLD.md`, older copies of this README.
  - `SYSTEM_TEST_RESULTS.md`: its results weren't reproducible.
  - `setup_github.sh`: notes from the first upload.

## Data and licenses

| Data | Source | License |
|---|---|---|
| `rlbench-data/push_buttons/` (downloaded, not committed): 25 demonstrations with front camera frames (128x128 PNG), `low_dim_obs.pkl`, `variation_descriptions.pkl` and `variation_number.pkl` | RLBench 18-task set (PerAct protocol), `test/push_buttons.tar.xz` from [huggingface.co/datasets/shihao1895/rlbench](https://huggingface.co/datasets/shihao1895/rlbench) at revision `b44678905fb2`; SHA-256 `9a331a2f…63fcdd8cb`, checked by the script | MIT as tagged by that dataset. The demonstrations were generated with [RLBench](https://github.com/stepjam/RLBench), whose software licence (Imperial College London) is for non-commercial use. |
| ResNet-18 ImageNet weights (downloaded by torchvision) | [torchvision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause (torchvision) |

The script keeps only the files the service reads; the other cameras, depth maps and masks in the archive are skipped. No data is committed to this repository.

RLBench: S. James, Z. Ma, D. Rovick Arrojo and A. J. Davison, "RLBench: The Robot Learning Benchmark & Learning Environment", *IEEE Robotics and Automation Letters*, 2020.

## Expected output

Recorded 2026-10-25 on macOS 26.5.1 (Apple M3), Python 3.12.15. Other CPUs give slightly different numbers, because floating-point results differ in training. For example, on Ubuntu with Python 3.12 in CI (same day):

- R²: 0.533 (1.0.0), 0.523 (1.1.0) and −0.592 (1.2.0).
- Canary verdicts: the same; the no-camera rollback is at +24%.

`python train.py`:

```
Test scores on held-out demonstrations (the joint change the expert made next):
version   inputs                            MAE (rad)      R²  MAE if no movement
1.0.0     front camera + joint positions      0.00586   0.560             0.01042
1.1.0     front camera + joint positions      0.00597   0.543             0.01042
1.2.0     joint positions                     0.00953  -0.565             0.01042

1870 training frames from 20 episodes, 591 test frames from 5 other episodes.
```

R² compares the error with always predicting the average move.

- With the camera, the model explains over half the variation, and its error is about half that of predicting no movement.
- Without the camera, it scores worse than the average move, because it can't see where the buttons are. Each demonstration places them differently, for example "push the maroon button, then push the green button, then push the silver button".

The `curl` request above (episode 21, frame 10):

```json
{"action":[0.00009854892414296046,-0.018847042694687843,0.0002839352819137275,-0.040501486510038376,-0.00027835473883897066,0.01927071437239647,0.00046435065451078117],"model_version":"1.0.0","step":10}
```

The expert's actual move at that frame, rounded: `[0.0018, -0.0199, 0.0016, -0.0414, -0.0006, 0.0195, 0.0033]`.

`python simulate_canary.py --max-samples 400 --canary-traffic 20`:

```
Baseline Version: 1.0.0
Candidate Version: 1.1.0 (manifest_candidate.yaml)
Samples Processed: 400

Canary deployment SUCCESSFUL - no rollback needed; the candidate can be promoted

Final Metrics:
  Baseline                          319 samples, p95 latency   20.1 ms, error vs expert 0.00591 rad, success 1.000
  Candidate                          81 samples, p95 latency   20.2 ms, error vs expert 0.00511 rad, success 1.000
  Baseline on candidate requests     81 samples, p95 latency   19.6 ms, error vs expert 0.00513 rad, success 1.000
  Action Divergence: 0.027
```

On its 81 requests the candidate matches the baseline. Over all 591 test frames it is 2% worse (`train.py` above), which is within the 10% tolerance. A canary sees only part of the traffic, so its verdict means "no worse beyond the tolerance", not "better".

`python simulate_canary.py --candidate manifest_candidate_no_camera.yaml --max-samples 400 --canary-traffic 20`:

```
Baseline Version: 1.0.0
Candidate Version: 1.2.0 (manifest_candidate_no_camera.yaml)
Samples Processed: 38

ROLLBACK TRIGGERED: Worse actions: error vs expert 0.00591 rad vs 0.00458 rad (+29%)

Final Metrics:
  Baseline                           28 samples, p95 latency   20.7 ms, error vs expert 0.00546 rad, success 1.000
  Candidate                          10 samples, p95 latency    4.0 ms, error vs expert 0.00591 rad, success 1.000
  Baseline on candidate requests     10 samples, p95 latency   19.2 ms, error vs expert 0.00458 rad, success 1.000
  Action Divergence: 1.138
```

Dropping the camera makes requests 5 times faster, and a latency check alone would pass it. The rollback comes after 10 candidate requests, because its actions are 29% further from the expert's.

## License

MIT; see [LICENSE](LICENSE). The data has its own licenses (above).
