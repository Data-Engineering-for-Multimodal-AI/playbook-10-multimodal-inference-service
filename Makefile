.PHONY: setup run smoke data train serve canary canary-no-camera docker-build docker-run clean

# A bare `make` runs the smoke test, as it did before the uniform targets were added.
.DEFAULT_GOAL := smoke

# Uniform targets for The Playbooks: setup, run, serve, smoke and clean. The book's commands
# (pip install -r requirements.txt, python download_sample_data.py, python train.py,
# uvicorn service:app ..., python simulate_canary.py --max-samples 5 --canary-traffic 20.0,
# docker run -p 8000:8000 multimodal-inference) still work as printed.

# Install the locked dependencies and download the RLBench demonstrations (skipped if present).
setup:
	uv sync --frozen
	uv run --frozen python download_sample_data.py

# Train the three manifests on CPU, then replay a short canary (5 samples, 20% canary traffic)
# and print the comparison. No server is left running; make serve starts the API.
run: setup
	uv run --frozen python train.py
	uv run --frozen python simulate_canary.py --max-samples 5 --canary-traffic 20.0

# Smoke tests on CPU (downloads the data and trains the models the first time)
smoke:
	uv run pytest -q

# RLBench push_buttons demonstrations (349 MB download, 78 MB kept)
data:
	uv run python download_sample_data.py

train: data
	uv run python train.py

serve: train
	uv run uvicorn service:app --host 127.0.0.1 --port 8000

canary: train
	uv run python simulate_canary.py --max-samples 400 --canary-traffic 20

canary-no-camera: train
	uv run python simulate_canary.py --candidate manifest_candidate_no_camera.yaml --max-samples 400 --canary-traffic 20

docker-build:
	docker build -t multimodal-inference .

docker-run: train
	docker run --rm -p 8000:8000 -v $(CURDIR)/rlbench-data:/app/rlbench-data:ro -v $(CURDIR)/models:/app/models:ro multimodal-inference

clean:
	rm -rf .cache models canary_results.json .pytest_cache
