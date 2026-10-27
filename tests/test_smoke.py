"""Smoke tests: data, the safe loader, training, the service and the canary.

Run with `make smoke` (or `uv run pytest -q`). CPU only. Downloads the push_buttons demonstrations
(349 MB, once) and ResNet-18, trains the three manifests (a few minutes the first time; the camera
embeddings are cached in .cache/) and replays canary traffic.
"""
import io
import json
import os
import pickle
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EPISODE = "rlbench-data/push_buttons/all_variations/episodes/episode21"


def run(*args, timeout=1800):
    result = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return result.stdout + result.stderr


@pytest.fixture(scope="session")
def data():
    run("download_sample_data.py")
    import rlbench_data
    return rlbench_data.require_data()


@pytest.fixture(scope="session")
def trained(data):
    out = run("train.py")
    metrics = {}
    for version in ("1.0.0", "1.1.0", "1.2.0"):
        with open(os.path.join(ROOT, "models", f"{version}.json")) as f:
            metrics[version] = json.load(f)
    return metrics, out


def test_safe_unpickler_refuses_code():
    from rlbench_data import SafeUnpickler

    class Exploit:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    with pytest.raises(pickle.UnpicklingError):
        SafeUnpickler(io.BytesIO(pickle.dumps(Exploit()))).load()


def test_demonstrations_have_telemetry(data):
    import rlbench_data
    assert len(data) == 25
    demo = rlbench_data.load_demo(os.path.join(ROOT, EPISODE, "low_dim_obs.pkl"))
    steps = len(demo["joint_positions"])
    assert demo["joint_positions"].shape == (steps, 7)
    assert len(os.listdir(os.path.join(ROOT, EPISODE, "front_rgb"))) == steps
    assert abs(rlbench_data.expert_actions(demo["joint_positions"])).mean() > 0


def test_training_on_held_out_episodes(trained):
    metrics, out = trained
    assert "Test scores on held-out demonstrations" in out
    with_camera, wider, no_camera = (metrics[v]["test"] for v in ("1.0.0", "1.1.0", "1.2.0"))
    assert metrics["1.0.0"]["test_episodes"] == 5 and metrics["1.0.0"]["train_episodes"] == 20
    assert with_camera["r2"] > 0.3 and wider["r2"] > 0.3
    assert with_camera["mae_rad"] < with_camera["mae_rad_if_no_movement"]
    assert no_camera["mae_rad"] > with_camera["mae_rad"]       # the camera matters


def test_service(trained):
    from fastapi.testclient import TestClient
    import service

    with TestClient(service.app) as client:
        assert client.get("/health").json() == {"status": "healthy", "model_version": "1.0.0", "trained": True}
        pkl = f"{EPISODE}/low_dim_obs.pkl"
        r = client.post("/infer", json={"rgb_path": f"{EPISODE}/front_rgb/10.png", "pkl_path": pkl})
        assert r.status_code == 200 and len(r.json()["action"]) == 7 and r.json()["step"] == 10
        r = client.post("/infer", json={"rgb_path": f"{EPISODE}/front_rgb", "pkl_path": pkl})
        assert r.status_code == 200
        assert client.post("/infer", json={"rgb_path": "/etc/hosts", "pkl_path": pkl}).status_code == 400
        assert client.post("/infer", json={"rgb_path": f"{EPISODE}/../../../../../model.py",
                                           "pkl_path": pkl}).status_code == 400
        assert client.post("/infer", json={"rgb_path": f"{EPISODE}/front_rgb/999.png",
                                           "pkl_path": pkl}).status_code == 404
        assert client.post("/infer", json={"rgb_path": f"{EPISODE}/front_rgb/10.png",
                                           "pkl_path": f"{EPISODE}/front_rgb/1.png"}).status_code == 422
        metrics = client.get("/metrics")
        assert metrics.headers["content-type"].startswith("text/plain")
        assert 'inference_errors_total{status="400"} 2.0' in metrics.text


def test_canary_promotes_wider_head(trained):
    out = run("simulate_canary.py", "--max-samples", "400", "--canary-traffic", "20")
    assert "Canary deployment SUCCESSFUL" in out
    with open(os.path.join(ROOT, "canary_results.json")) as f:
        assert json.load(f)["decision"] == "promote"


def test_canary_rolls_back_no_camera(trained):
    out = run("simulate_canary.py", "--candidate", "manifest_candidate_no_camera.yaml",
              "--max-samples", "400", "--canary-traffic", "20")
    assert "ROLLBACK TRIGGERED: Worse actions" in out


def test_printed_canary_command(trained):
    out = run("simulate_canary.py", "--max-samples", "5", "--canary-traffic", "20.0")   # as printed, page 415
    assert "CANARY DEPLOYMENT SIMULATION RESULTS" in out


def test_service_starts_without_data_or_weights(tmp_path, monkeypatch):
    # The book runs `python service.py` (and the Docker image) before any data is downloaded
    import yaml
    import rlbench_data
    from model import ModelManager

    monkeypatch.setattr(rlbench_data, "DATA_ROOT", str(tmp_path / "no-data"))
    monkeypatch.setattr(rlbench_data.episode_dirs, "__defaults__", (rlbench_data.DEFAULT_TASK, str(tmp_path / "no-data")))
    with open(os.path.join(ROOT, "manifest.yaml")) as f:
        config = yaml.safe_load(f)
    config["weights"] = str(tmp_path / "missing.pt")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(config))
    manager = ModelManager(str(manifest))
    assert manager.trained is False
