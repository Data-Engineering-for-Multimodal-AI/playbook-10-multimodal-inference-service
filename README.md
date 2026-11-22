# Playbook 10: Serve a robot model with canary rollout and rollback
**How to build and operate a multimodal inference service.** Chapter 10 of [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026), pages 407–413.

A robot's model can't be swapped blindly: a new version has to roll out to a few robots first and roll back if it gets worse. In this repo, we explore this with Eighteen Labs, a fictional robotics company, and a small RLBench robot policy. Follow along to wrap the model in a service with extra inputs (text, sound, telemetry), canary rollout, rollback, telemetry and drift checks.

**The problem.** Updating a model in a live system can break it, so every change has to be shipped in a way you can watch and undo.

**What you'll work with.** You'll package the model in a container with a deployment manifest, send a share of traffic to the new version, roll back when it gets worse, and watch telemetry and drift.

> Following the book? Its Steps 1–4 are in [Run it the way the book does](#run-it-the-way-the-book-does) and still work as printed. Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## Run it

You need [uv](https://docs.astral.sh/uv/) and `make`; Docker only for the optional image in [Run it the way the book does](#run-it-the-way-the-book-does). Or open the repo in a GitHub Codespace, where both are ready. On Windows without `make`, use a Codespace or the book's steps below.

The times below are from a Mac (Apple M3, 8 GB) that was running other jobs at the same time, so they are slower than on an idle machine.

**1. Get the code.**

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-10-multimodal-inference-service.git
cd playbook-10-multimodal-inference-service
```

**2. Download the data, train the three versions and replay a short canary.** Installs the locked dependencies, downloads the 25 RLBench `push_buttons` demonstrations (`python download_sample_data.py`, 349 MB, 78 MB kept), trains the three manifests (`python train.py`) and runs the book's short canary (`python simulate_canary.py --max-samples 5 --canary-traffic 20.0`). On a fresh clone this took 10 min 34 s on the Mac, including the download. In CI, with the data cached, it took 36 s.

```bash
make run
```

You should see the test scores of the three versions, then the canary's verdict:

```
version   inputs                            MAE (rad)      R²  MAE if no movement
1.0.0     front camera + joint positions      0.00586   0.560             0.01042
1.1.0     front camera + joint positions      0.00597   0.543             0.01042
1.2.0     joint positions                     0.00953  -0.565             0.01042
...
Not enough candidate traffic to decide (1 of 10 samples): the baseline keeps serving; run with more samples
```

The book's 5-request canary sends one request to the candidate, too few to decide. Step 4 sends 400.

**3. Serve it and send a request.** Starts the API on port 8000 (`uvicorn service:app`). It retrains the three versions first; on the Mac, with the camera embeddings cached, the API answered 75 s after the command. Leave it running.

```bash
make serve
```

In a second terminal, send the book's request (episode 21, frame 10), then another frame:

```bash
make infer
make infer EPISODE=22 FRAME=40
```

```
{"action":[0.00009854892414296046,-0.018847042694687843,0.0002839352819137275,-0.040501486510038376,-0.00027835473883897066,0.01927071437239647,0.00046435065451078117],"model_version":"1.0.0","step":10}
{"action":[-0.0003630587598308921,0.008821416646242142,-0.0008226005593314767,-0.032100338488817215,-0.0005674407002516091,0.03473471850156784,-0.0005583956954069436],"model_version":"1.0.0","step":40}
```

The action is the predicted change of the 7 joints, in radians. Episodes 20 to 24 are the held-out ones; any frame number that exists works. The service counts what it served:

```bash
curl -s localhost:8000/metrics | grep -E "^inference_requests_total|^model_version_info"
```

```
inference_requests_total 2.0
model_version_info{fusion="concat",policy="mlp_bc_v1",telemetry="joints_v1",version="1.0.0",vision="resnet18"} 1.0
```

Stop the API with Ctrl+C.

**4. Roll out the two candidates.** Replays 400 held-out requests with 20% going to the candidate (`python simulate_canary.py --max-samples 400 --canary-traffic 20`), first for 1.1.0 (a wider network), then for 1.2.0 (no camera). Each took under a minute on the Mac.

```bash
make canary
make canary-no-camera
```

```
Canary deployment SUCCESSFUL - no rollback needed; the candidate can be promoted
...
ROLLBACK TRIGGERED: Worse actions: error vs expert 0.00591 rad vs 0.00458 rad (+29%)
  Candidate                          10 samples, p95 latency    5.8 ms, error vs expert 0.00591 rad, success 1.000
  Baseline on candidate requests     10 samples, p95 latency   31.3 ms, error vs expert 0.00458 rad, success 1.000
```

The no-camera version answered 5 times faster and was rolled back after 10 requests, because its actions were 29% further from what the expert did. The full reports are in [Expected output](#expected-output).

**5. Clean up.**

```bash
make clean
```

This deletes the trained weights (`models/`), the cached camera embeddings (`.cache/`), `canary_results.json` and the test cache. The downloaded demonstrations in `rlbench-data/` stay; delete that folder to remove them.

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1 to 4. On the busy Mac, step 2 took 10 min 34 s, most of it the download and the first embedding of the frames. |
| 1 hour | The three [Try this](#try-this) experiments (under 4 minutes each on the busy Mac), then the canary with your own thresholds (`uv run python simulate_canary.py --help`). |
| 1 day | [Use your own data](#use-your-own-data) with your own demonstrations or another RLBench task, the Docker image, and [Before production](#before-production). |

## Try this

Each experiment changes one thing and measures the effect. None needs the server or Docker. The numbers below are from the busy Mac on 2026-11-21; yours will differ.

**1. How much traffic should the canary get?** Runs the canary's own simulator with 5, 10, 20 and 50% of the traffic going to the candidate, with 10 routing seeds each, for both candidates. Each model's answer and latency are measured once per frame, so the 80 runs take under 4 minutes.

```bash
make try-canary-traffic
```

```
Candidate manifest_candidate.yaml: up to 400 requests, 10 seeds per share

canary share decisions                     requests to decide  served by candidate
          5% 8 promote, 2 rollback                        400                   17
         10% 8 promote, 2 rollback                        400                   34
         20% 9 promote, 1 rollback                        400                   82
         50% 9 promote, 1 rollback                        400                  204
rollback reasons: {'Worse actions': 6}

Candidate manifest_candidate_no_camera.yaml: up to 400 requests, 10 seeds per share

canary share decisions                     requests to decide  served by candidate
          5% 10 rollback                                  188                   10
         10% 10 rollback                                  108                   10
         20% 10 rollback                                   42                   10
         50% 10 rollback                                   20                   10
rollback reasons: {'Action divergence too high': 5, 'Worse actions': 35}
```

The bad candidate was always caught after 10 of its own requests; a bigger share only reached those 10 sooner (20 requests at 50%, 188 at 5%). The good candidate, 2% worse than the baseline over all test frames, was rolled back in 6 of 40 runs: the canary checks its error after every request from the 10th on, and on so few requests the noise alone can cross the 10% tolerance. In CI (Ubuntu, where training gives slightly different weights), it was rolled back in 11 of 40 runs. Raise `--min-samples` or `--quality-tolerance` and run it again.

**2. Does the service notice when the camera changes?** Darkens and blurs the held-out frames, as a dimmer room or a dirty lens would, and measures version 1.0.0's error and a drift score that needs no labels: how far the frames' ResNet-18 embeddings are from the training frames' (1.0 is as far as a typical training frame).

```bash
make try-camera-drift
```

```
camera                error vs expert (rad)  change  drift score
as recorded                         0.00572     +0%         1.00
70% brightness                      0.00643    +12%         1.08
40% brightness                      0.00744    +30%         1.30
blurred (radius 2)                  0.00896    +57%         1.95
blurred (radius 4)                  0.01104    +93%         2.44
```

The error rose with every change, and the drift score rose with it, so the score can raise an alarm before anyone has labels. With the strongest blur, the model was worse than predicting no movement (0.01042). The canary can't see this: both versions get the same blurred frames. Add your own change to `CONDITIONS` in `experiments/camera_drift.py`.

**3. Where does a request's time go?** Times each part of `predict()` on 100 held-out frames, for 1.0.0 and the no-camera 1.2.0.

```bash
make try-latency-breakdown
```

```
version                     read telemetry  open frame  ResNet-18  fusion head  whole predict()
1.0.0 (camera + joints)               8.10        1.96      59.63         0.34            77.97
1.2.0 (joints only)                  12.03        2.52          -         0.40            14.06

Each request reads the whole low_dim_obs.pkl (295 KB, every frame of the episode) to use one row of it.
```

ResNet-18 takes most of 1.0.0's time. Without it, 1.2.0 spends most of its time unpickling the whole episode's telemetry for one row of 7 numbers, and the model itself takes 0.4 ms. Keeping the telemetry in memory would remove about 12 of 1.2.0's 14 ms, and 8 of 1.0.0's 78 ms. In CI the parts were faster and the shares similar: reading the telemetry took 3.72 of 1.2.0's 3.92 ms.

## Use your own data

Put your demonstrations in a folder, one folder per episode, in the RLBench layout the service reads (directly, or under `<task>/all_variations/episodes/`):

```
my_demos/
  episode0/
    front_rgb/        0.png 1.png ... N-1.png   the front camera, one frame per step, numbered from 0
    low_dim_obs.pkl                             the pickled RLBench Demo: one observation per frame,
                                                each with joint_positions (7 numbers, radians)
  episode1/ ...
```

Check it first. This reads every file and runs no model:

```bash
make check-data DIR=my_demos
```

```
my_demos: 3 episodes, 330 frames with telemetry. All checks passed.
```

It reports every problem with the file and the reason, and runs nothing if there is one. These are from a copy of that folder with mistakes put in on purpose, including a pickle that would run a shell command when loaded:

```
6 problem(s) in broken; nothing was run:
  episode0/front_rgb: frames don't run 0 to 117 without gaps (missing 5.png)
  episode0: 118 frames but 119 observations in low_dim_obs.pkl
  episode1/front_rgb/frame_extra.png: not named <step>.png
  episode1/front_rgb/3.png: Git LFS pointer, not the image (run git lfs pull)
  episode2/low_dim_obs.pkl: refused by the safe loader (refusing to load posix.system from a pickle)
  episode3: no low_dim_obs.pkl
```

It also checks for observations without 7 finite joint positions, frames that won't open or are smaller than 32x32, and episodes with no `front_rgb/`. Then score the three versions on your demonstrations, the way `train.py` scores the held-out ones:

```bash
make evaluate DIR=my_demos
```

```
Scores on your demonstrations (the joint change each one made next):
version   inputs                            MAE (rad)      R²  MAE if no movement
1.0.0     front camera + joint positions      0.01510  -0.069             0.01268
1.1.0     front camera + joint positions      0.01356   0.109             0.01268
1.2.0     joint positions                     0.12926 -93.972             0.01268

327 frames from 3 episodes. R² below 0: worse than always predicting the training data's average move.
```

This folder holds three demonstrations of another RLBench task, `open_drawer` (`python download_sample_data.py --task open_drawer`, then copy three episodes). The models learned to push buttons, so on a drawer they do worse than predicting no movement, and the joints-only version is far off. Scoring a version on your own episodes like this, before any canary, shows whether it fits your robot at all. `make evaluate` took 31 s for these 330 frames on the Mac. It writes nothing.

## What you end up with

A model service with three versions, metrics and a canary that promotes or rolls back, plus measurements of how the canary decides, what input drift does to it, and where its latency goes.

Skills you'll practice:
- Training and versioning a multimodal model with manifests
- Serving it behind an API with health checks and Prometheus metrics
- Rolling out a version with a canary, a shadow baseline and rollback rules
- Sizing a canary and seeing when its rules give false alarms
- Detecting input drift without labels
- Checking robot data, including unsafe pickles, before a model reads it

## Before production

This playbook runs on one machine with 25 public demonstrations. Before you run something like it for real:

1. **Check a canary's verdict before acting on it.** The rules are checked after every candidate request from the 10th on, so noise can trigger them: a candidate 2% worse overall was rolled back in 6 of 40 runs ([Try this 1](#try-this)). Require more requests, or use a test that accounts for checking repeatedly. *Chapter 10, "Canary Deployment" (page 397).*
2. **Watch the inputs, not only the versions.** A blurred camera nearly doubled the error (+93%), and a canary comparing two versions on the same frames can't see it. The embedding drift score rose from 1.00 to 2.44 with it ([Try this 2](#try-this)). Export a drift score like it to `/metrics` and alert on it. *Chapter 10, "Detecting Drift" (page 384).*
3. **Keep telemetry in memory.** Every request unpickles the whole episode file (295 KB) for one row; that is 12 of the 14 ms of a joints-only request ([Try this 3](#try-this)). In a real robot, telemetry arrives as a stream; read the current state, not a file. *Chapter 10, "Concurrency and Synchronization" (page 389).*
4. **Batch the camera model.** ResNet-18 takes 60 of the 78 ms of a request with the camera, one frame at a time. Batch requests that arrive together, or move the encoder to an accelerator. *Chapter 10, "Batching Across Modalities" (page 391); Chapter 10, "Hardware and Accelerator Scheduling" (page 393).*
5. **Evaluate on your own robot before a canary.** On three `open_drawer` demonstrations, every version did worse than predicting no movement ([Use your own data](#use-your-own-data)). A canary only compares versions; it doesn't tell you whether either one fits. *Chapter 10, "Evaluation Framework and Retraining" (page 386).*
6. **Keep the rollback path ready.** The canary script decides, but nothing switches the served version: the service serves one manifest per process (`INFERENCE_MANIFEST`). Route traffic between two running versions and switch back with one command. *Chapter 10, "Rollback" (page 400).*
7. **Accept only data you trust.** The service limits paths to the data folder and its loader refuses pickles with other classes (`make check-data` showed it refusing `posix.system`), but pickle is still a format that can run code. Prefer a plain format (NumPy, Parquet) for telemetry from outside. *Chapter 11, "Protecting Data Across the Pipeline" (page 421).*
8. **Version the whole model.** A manifest names the weights file but not a checksum of it or of the training data. Record both, so a version can be rebuilt and verified. *Chapter 10, "Version Control for Composite Models" (page 396).*
9. **Run the image with limits.** The Docker image runs as a non-root user with a health check; give it memory and CPU limits and a read-only filesystem in production. *Chapter 10, "CI/CD Orchestration" (page 398).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine | $0. Everything runs locally: no API keys, no paid services. About 0.8 GB of disk on a Mac: 690 MB for the environment, 78 MB of demonstrations, and a few MB of weights and cached embeddings. The Docker image adds 1.5 GB. |
| A GitHub Codespace (2 cores, 8 GB, the default size; this repo asks for none) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` in a clean devcontainer (the data was downloaded while the container was created) took 129 s; the whole job took about 3 minutes. |

## How it works

The model combines what the front camera sees and where the seven joints are.

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

## Run it the way the book does

These are the book's Steps 1–4 (pages 409–412), with the fixes listed in [Differences from the printed book](#differences-from-the-printed-book).

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

With [uv](https://docs.astral.sh/uv/): `uv sync`, then prefix commands with `uv run`. The Makefile has shortcuts: `make train`, `make serve`, `make canary`, `make canary-no-camera`, `make smoke`; `make run`, `make infer`, `make check-data`, `make evaluate` and `make try-*` are described in [Run it](#run-it), [Use your own data](#use-your-own-data) and [Try this](#try-this).

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

Every check has a command-line option (`--help`). The book's `python simulate_canary.py --max-samples 5 --canary-traffic 20.0` (page 411) still runs. It sends one request to the candidate, so it reports that there isn't enough traffic to decide; the book shows a PROMOTE verdict in a different report format.

### 4. Docker

```bash
docker build -t multimodal-inference .
docker run -p 8000:8000 -v $(pwd)/rlbench-data:/app/rlbench-data:ro -v $(pwd)/models:/app/models:ro multimodal-inference
```

- The image is Python 3.12 with CPU-only PyTorch (1.5 GB); ResNet-18's weights are inside it.
- The demonstrations and trained weights are mounted, so train first.
- The book's `docker run -p 8000:8000 multimodal-inference` (page 412), without mounts, also starts, as in the first edition, but with an untrained model: `/health` says `"trained": false`. Without the data, `/infer` has nothing to read.
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
- the checks behind `make check-data` (`tests/test_own_data.py`)

CI runs them on Python 3.12 and 3.13. It also runs the README's `pip install -r requirements.txt`, `uvicorn` and `curl` commands, and builds and queries the Docker image.

## Tested on

- 2026-10-25: macOS 26.5.1 (Apple M3, 8 GB), Python 3.12.15, torch 2.14.1, FastAPI 0.162.0, Docker 28.0.1.
- Ubuntu (GitHub Actions) with Python 3.12 and 3.13: see the `smoke` workflow.

## Differences from the printed book

The code as it was when the book was written is tagged `first-edition`. What changed and why:

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
- **Starting without data.** As in the first edition, `python service.py` and the Docker image start before any data is downloaded, with an untrained model and a warning; `/health` now reports `"trained"`. With the data and no weights, the service trains first.
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

Recorded 2026-10-25 on macOS 26.5.1 (Apple M3), Python 3.12.15. This is the output of this version; it may differ from the output printed in the book, and that's okay. Other CPUs give slightly different numbers, because floating-point results differ in training. For example, on Ubuntu with Python 3.12 in CI (same day):

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
