"""Try this 2: does the service notice when the camera changes? (make try-camera-drift)

The canary compares two models on the same inputs, so it can't see the inputs themselves
drift. This darkens and blurs the held-out camera frames, as a dimmer room or a dirty lens
would, and measures two things for version 1.0.0: its error against what the expert did, and
a drift score the service could compute without any labels: how far each frame's ResNet-18
embedding is from the training frames' embeddings (mean distance to the training mean, in
units of the training frames' own spread; 1.0 is a typical training frame).
No server, no Docker.
"""

import logging
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.disable(logging.WARNING)
import rlbench_data  # noqa: E402
import train  # noqa: E402
from model import ModelManager  # noqa: E402

CONDITIONS = [
    ("as recorded", lambda im: im),
    ("70% brightness", lambda im: ImageEnhance.Brightness(im).enhance(0.7)),
    ("40% brightness", lambda im: ImageEnhance.Brightness(im).enhance(0.4)),
    ("blurred (radius 2)", lambda im: im.filter(ImageFilter.GaussianBlur(2))),
    ("blurred (radius 4)", lambda im: im.filter(ImageFilter.GaussianBlur(4))),
]


def main():
    manager = ModelManager("manifest.yaml")
    model = manager.get_model()
    dirs = rlbench_data.require_data()
    train_dirs = rlbench_data.parse_episodes("0-19", dirs)
    test_dirs = rlbench_data.parse_episodes("20-24", dirs)

    # Training-frame embeddings (cached by train.py) define "normal"
    reference = np.concatenate([train.episode_embeddings(model, d, len(rlbench_data.load_demo(
        os.path.join(d, "low_dim_obs.pkl"))["joint_positions"])) for d in train_dirs])
    center = reference.mean(0)
    spread = np.linalg.norm(reference - center, axis=1).mean()

    frames, joints, expert = [], [], []
    for d in test_dirs:
        jp = rlbench_data.load_demo(os.path.join(d, "low_dim_obs.pkl"))["joint_positions"]
        steps = range(0, len(jp) - 1, 4)
        frames += [Image.open(rlbench_data.frame_path(d, s)).convert("RGB") for s in steps]
        joints.append(jp[list(steps)])
        expert.append(rlbench_data.expert_actions(jp)[list(steps)])
    joints, expert = torch.tensor(np.concatenate(joints)), np.concatenate(expert)

    print(f"Version {manager.get_version()} on {len(frames)} held-out frames\n")
    print(f"{'camera':<20} {'error vs expert (rad)':>22} {'change':>7} {'drift score':>12}")
    base = None
    for name, change in CONDITIONS:
        emb = model.embed_images([change(f) for f in frames])
        with torch.no_grad():
            predicted = model(emb, joints).numpy()
        mae = float(np.abs(predicted - expert).mean())
        drift = float(np.linalg.norm(emb.numpy() - center, axis=1).mean() / spread)
        base = base or mae
        print(f"{name:<20} {mae:>22.5f} {f'{(mae / base - 1) * 100:+.0f}%':>7} {drift:>12.2f}")


if __name__ == "__main__":
    main()
