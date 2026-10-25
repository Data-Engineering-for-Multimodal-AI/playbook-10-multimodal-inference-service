.PHONY: smoke data train serve canary canary-no-camera docker-build docker-run clean

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
