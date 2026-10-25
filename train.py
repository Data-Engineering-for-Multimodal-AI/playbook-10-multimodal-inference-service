#!/usr/bin/env python3
"""
Train the fusion head a manifest describes, by behaviour cloning on RLBench demonstrations.

    python train.py                          # the three shipped manifests
    python train.py manifest_candidate.yaml  # one manifest

Each example is one frame of a demonstration: the front camera image and the 7 joint positions as
input, and the joint change the expert made to the next frame as the target. Episodes are split
by number (manifest `training.train_episodes` / `test_episodes`), so the test scores come from
demonstrations the model never saw. ResNet-18 stays frozen; its embeddings are cached in .cache/.

Writes models/<version>.pt (the head and its standardization) and models/<version>.json (test scores).
"""
import argparse
import json
import logging
import os
import time

import numpy as np
import torch
import yaml
from PIL import Image

import rlbench_data
from model import build_model, weights_path

logger = logging.getLogger(__name__)
CACHE_DIR = os.path.join(rlbench_data.REPO_ROOT, ".cache", "embeddings")
MANIFESTS = ["manifest.yaml", "manifest_candidate.yaml", "manifest_candidate_no_camera.yaml"]


def episode_embeddings(model, episode_dir: str, steps: int) -> np.ndarray:
    """ResNet-18 embeddings of an episode's front camera frames (cached)."""
    task = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(episode_dir))))
    path = os.path.join(CACHE_DIR, task, os.path.basename(episode_dir) + ".npy")
    if os.path.exists(path):
        cached = np.load(path)
        if len(cached) == steps:
            return cached
    images = [Image.open(rlbench_data.frame_path(episode_dir, i)) for i in range(steps)]
    embeddings = model.embed_images(images).numpy().astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    return embeddings


def examples(model, episode_dirs: list) -> tuple:
    """(inputs, expert actions) for every frame that has a next frame."""
    inputs, targets = [], []
    for episode_dir in episode_dirs:
        joints = rlbench_data.load_demo(os.path.join(episode_dir, "low_dim_obs.pkl"))["joint_positions"]
        parts = [joints[:-1, :model.telemetry_dim]]
        if model.use_vision:
            parts.insert(0, episode_embeddings(model, episode_dir, len(joints))[:-1])
        inputs.append(np.concatenate(parts, axis=1))
        targets.append(rlbench_data.expert_actions(joints))
    return np.concatenate(inputs), np.concatenate(targets)


def scores(predicted: np.ndarray, expert: np.ndarray, train_mean: np.ndarray) -> dict:
    """Mean absolute error in radians, and R² against always predicting the training mean."""
    mse = float(((predicted - expert) ** 2).mean())
    return {
        "mae_rad": float(np.abs(predicted - expert).mean()),
        "r2": 1 - mse / float(((expert - train_mean) ** 2).mean()),
        "mae_rad_if_no_movement": float(np.abs(expert).mean()),
    }


def train_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        config = yaml.safe_load(f)
    settings = config.get("training", {})
    task = settings.get("task", rlbench_data.DEFAULT_TASK)
    dirs = rlbench_data.require_data(task)
    train_dirs = rlbench_data.parse_episodes(settings.get("train_episodes", "0-19"), dirs)
    test_dirs = rlbench_data.parse_episodes(settings.get("test_episodes", "20-24"), dirs)
    seed = settings.get("seed", 0)

    torch.manual_seed(seed)
    model = build_model(config)
    started = time.time()
    x_train, y_train = examples(model, train_dirs)
    x_test, y_test = examples(model, test_dirs)
    embed_seconds = time.time() - started

    # Standardize inputs and targets; the statistics are part of the saved model
    model.input_mean.copy_(torch.tensor(x_train.mean(0)))
    model.input_std.copy_(torch.tensor(x_train.std(0) + 1e-6))
    model.output_mean.copy_(torch.tensor(y_train.mean(0)))
    model.output_std.copy_(torch.tensor(y_train.std(0) + 1e-6))

    head = model.fusion_head
    optimizer = torch.optim.Adam(head.parameters(), lr=settings.get("learning_rate", 1e-3),
                                 weight_decay=settings.get("weight_decay", 1e-4))
    x = (torch.tensor(x_train) - model.input_mean) / model.input_std
    y = (torch.tensor(y_train) - model.output_mean) / model.output_std
    generator = torch.Generator().manual_seed(seed)
    batch_size = settings.get("batch_size", 128)
    started = time.time()
    head.train()
    for _ in range(settings.get("epochs", 300)):
        order = torch.randperm(len(x), generator=generator)
        for i in range(0, len(x), batch_size):
            batch = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = ((head(x[batch]) - y[batch]) ** 2).mean()
            loss.backward()
            optimizer.step()
    head.eval()
    train_seconds = time.time() - started

    with torch.no_grad():
        vision = torch.tensor(x_test[:, :model.vision_embedding_dim])
        telemetry = torch.tensor(x_test[:, model.vision_embedding_dim:])
        predicted = model(vision, telemetry).numpy()

    weights = weights_path(config)
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("vision_encoder.")}
    torch.save(state, weights)
    metrics = {
        "version": config["version"],
        "manifest": os.path.basename(manifest_path),
        "inputs": (["front camera"] if model.use_vision else []) + ["joint positions"],
        "task": task,
        "train_episodes": len(train_dirs), "train_examples": len(x_train),
        "test_episodes": len(test_dirs), "test_examples": len(x_test),
        "test": scores(predicted, y_test, y_train.mean(0)),
        "embedding_seconds": round(embed_seconds, 1), "training_seconds": round(train_seconds, 1),
    }
    with open(os.path.splitext(weights)[0] + ".json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Trained {config['version']} -> {weights}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="*", default=MANIFESTS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = []
    for manifest in args.manifests:
        print(f"Training {manifest}...")
        results.append(train_manifest(manifest))
    print("\nTest scores on held-out demonstrations (the joint change the expert made next):")
    print(f"{'version':9s} {'inputs':32s} {'MAE (rad)':>10s} {'R²':>7s} {'MAE if no movement':>19s}")
    for r in results:
        t = r["test"]
        print(f"{r['version']:9s} {' + '.join(r['inputs']):32s} {t['mae_rad']:>10.5f} {t['r2']:>7.3f} "
              f"{t['mae_rad_if_no_movement']:>19.5f}")
    print(f"\n{results[0]['train_examples']} training frames from {results[0]['train_episodes']} episodes, "
          f"{results[0]['test_examples']} test frames from {results[0]['test_episodes']} other episodes.")


if __name__ == "__main__":
    main()
