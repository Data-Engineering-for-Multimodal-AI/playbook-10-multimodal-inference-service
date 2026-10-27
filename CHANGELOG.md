# Changelog

## 2026.10.1 (2026-10-27)

Checked against the book (Early Release, pages 410–416).

### Fixed
- The book's `python service.py` and `docker run -p 8000:8000 multimodal-inference` (pages 414–415)
  failed to start on a fresh clone: since 2026.10 the service trains when it has no weights, and
  there was no data. It now starts with an untrained model and a warning, as the first edition did;
  `/health` reports `"trained"`. CI runs the image without mounts.

### Changed
- The tests and CI run the canary command as printed (`--max-samples 5`).
- README: page numbers; notes on the printed canary and Docker commands.

## 2026.10 (2026-10-25)

Maintenance update. The code as printed in the book is tagged `first-edition`.

### Fixed
- Telemetry was never used.
  - The joint positions in `low_dim_obs.pkl` sit in `Demo._observations`, which the loader skipped, so every request used zeros.
  - The service now reads the joint positions at the frame it serves.
- `/infer` unpickled any file path a client sent, which allowed code execution through a crafted pickle.
  - Paths are now limited to the data folder (`RLBENCH_DATA`).
  - The loader accepts only RLBench and NumPy classes.
- Bad requests returned 500. They now return 400 (outside the data folder), 404 (missing) or 422 (unusable file).
- `/metrics` sent Prometheus text with the OpenMetrics content type.
- Inference blocked the event loop (async endpoint); it now runs in FastAPI's thread pool.
- Deprecations: `@app.on_event` (now lifespan) and torchvision `pretrained=True` (now `weights=`).
- `download_sample_data.py` downloaded nothing. It now downloads the RLBench `push_buttons` demonstrations from a pinned revision, verifies the SHA-256 and keeps only the files the service reads.

### Changed
- **A trained model.**
  - Before: the fusion head kept random weights, so each restart produced different actions.
  - `train.py` now trains it by behaviour cloning: camera frame plus joint positions, predicting the expert's next joint change.
  - 20 demonstrations train it and 5 others test it: R² 0.56, error 0.0059 rad against 0.0104 rad for no movement.
  - Manifests name their weights and training split.
- **A real canary.**
  - Before: two untrained networks were compared.
  - Now the candidate gets a seeded share of held-out requests, and the baseline answers the same requests in shadow.
  - The rollback checks errors, p95 latency, error against the expert's actions, and divergence.
  - Two candidates are committed: 1.1.0 (wider head, passes) and 1.2.0 (no camera; 5 times faster, rolled back for worse actions).
- **Docker.**
  - Before: Python 3.9 (end of life) with CUDA PyTorch.
  - Now: Python 3.12 slim, CPU-only PyTorch, ResNet-18 weights baked in, non-root user, and data and weights mounted read-only.
- Python 3.12+, with `pyproject.toml`, `uv.lock` and a pinned `requirements.txt` (CPU PyTorch on Linux).

### Added
- `rlbench_data.py`, `train.py`, `manifest_candidate.yaml`, `manifest_candidate_no_camera.yaml`, `.dockerignore`, smoke tests (`make smoke`), CI (Python 3.12 and 3.13, the printed commands, Docker), a Makefile.

### Removed (with the author's approval)
- `README_GITHUB.md` and `README_OLD.md` (older copies of the README).
- `SYSTEM_TEST_RESULTS.md` (results that couldn't be reproduced).
- `setup_github.sh` (notes from the first upload).
