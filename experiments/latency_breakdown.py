"""Try this 3: where does a request's time go? (make try-latency-breakdown)

Every /infer request runs MultimodalInferenceModel.predict (model.py): it unpickles the whole
episode's telemetry, opens the camera frame, embeds it with ResNet-18 and runs the fusion
head. This times each part on 100 held-out frames, for version 1.0.0 (camera + joints) and
1.2.0 (joints only), and then the whole predict() call. No server, no Docker.
"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import logging  # noqa: E402

logging.disable(logging.WARNING)
import rlbench_data  # noqa: E402
from model import ModelManager  # noqa: E402

N = 100


def ms(seconds):
    return f"{np.median(seconds) * 1000:8.2f}"


def main():
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    dirs = rlbench_data.parse_episodes("20-24", rlbench_data.require_data())
    frames = []
    for d in dirs:
        steps = len(os.listdir(os.path.join(d, "front_rgb")))
        frames += [(rlbench_data.frame_path(d, s), os.path.join(d, "low_dim_obs.pkl"), s)
                   for s in range(0, steps, max(1, steps // (N // len(dirs))))][:N // len(dirs)]
    print(f"Median time per request over {len(frames)} held-out frames, in ms\n")
    print(f"{'version':<26} {'read telemetry':>15} {'open frame':>11} {'ResNet-18':>10} {'fusion head':>12} "
          f"{'whole predict()':>16}")
    for manifest in ("manifest.yaml", "manifest_candidate_no_camera.yaml"):
        manager = ModelManager(manifest)
        model = manager.get_model()
        t = {k: [] for k in ("pkl", "img", "embed", "head", "all")}
        model.predict(*frames[0][:2])  # warm-up
        for rgb, pkl, step in frames:
            a = time.perf_counter()
            telemetry = rlbench_data.load_demo(pkl)
            b = time.perf_counter()
            image = Image.open(rgb)
            image.load()
            c = time.perf_counter()
            vision = model.extract_vision_features(image) if model.use_vision else torch.zeros(1, 0)
            d = time.perf_counter()
            with torch.no_grad():
                model(vision, model.extract_telemetry_features(telemetry, step))
            e = time.perf_counter()
            model.predict(rgb, pkl)
            f = time.perf_counter()
            for k, v in zip(t, (b - a, c - b, d - c, e - d, f - e)):
                t[k].append(v)
        inputs = "camera + joints" if model.use_vision else "joints only"
        print(f"{manager.get_version() + ' (' + inputs + ')':<26} {ms(t['pkl']):>15} {ms(t['img']):>11} "
              f"{ms(t['embed']) if model.use_vision else '       -':>10} {ms(t['head']):>12} {ms(t['all']):>16}")
    size = os.path.getsize(frames[0][1]) / 1024
    print(f"\nEach request reads the whole low_dim_obs.pkl ({size:.0f} KB, every frame of the episode) "
          f"to use one row of it.")


if __name__ == "__main__":
    main()
