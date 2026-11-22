"""Check a folder of your own robot demonstrations, then score every model version on them.

    make check-data DIR=my_demos     # check every file, run nothing
    make evaluate DIR=my_demos       # check, then score 1.0.0, 1.1.0 and 1.2.0 on the demonstrations

The folder holds episodes in the RLBench layout the service reads, either directly or under
<task>/all_variations/episodes/:

    my_demos/
      episode0/
        front_rgb/0.png 1.png ... N-1.png   the front camera, one PNG per frame, numbered from 0
        low_dim_obs.pkl                     the pickled rlbench Demo: one observation per frame,
                                            each with joint_positions (7 numbers, radians)
      episode1/ ...

The checks are the things that break the service or make its scores wrong: a missing front_rgb/
or low_dim_obs.pkl, a pickle the service's safe loader refuses (anything but RLBench and NumPy
classes) or can't read, Git LFS pointers, observations without 7 finite joint positions, frame
files not named <step>.png, gaps in the frame numbers, a frame count that doesn't match the
observations, and frames that won't open, aren't RGB-convertible or are smaller than 32x32.
Every problem is reported with the file and the reason; if there is any, nothing runs.

`make evaluate` runs each trained version on every frame (camera embedding, joint positions)
and compares its prediction with the joint change your demonstration made next, like train.py
does for the held-out RLBench episodes. It writes nothing and sends nothing anywhere.
"""

import argparse
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import rlbench_data  # noqa: E402

MANIFESTS = ["manifest.yaml", "manifest_candidate.yaml", "manifest_candidate_no_camera.yaml"]
MIN_SIDE = 32


def find_episodes(folder):
    """Episode folders: any folder holding low_dim_obs.pkl or front_rgb/, sorted by name."""
    found = set()
    for dirpath, dirnames, filenames in os.walk(folder):
        if "low_dim_obs.pkl" in filenames or "front_rgb" in dirnames:
            found.add(dirpath)
            dirnames[:] = []   # don't look inside an episode
    def key(d):
        m = re.search(r"(\d+)$", d)
        return (int(m.group(1)) if m else -1, d)
    return sorted(found, key=key)


def check_episode(folder, ep):
    """Problems of one episode (relative paths), and its joint positions if they loaded."""
    rel = os.path.relpath(ep, folder)
    problems, joints = [], None
    pkl = os.path.join(ep, "low_dim_obs.pkl")
    rgb = os.path.join(ep, "front_rgb")
    if not os.path.exists(pkl):
        problems.append(f"{rel}: no low_dim_obs.pkl")
    else:
        with open(pkl, "rb") as f:
            head = f.read(40)
        if head.startswith(b"version https://git-lfs"):
            problems.append(f"{rel}/low_dim_obs.pkl: Git LFS pointer, not the file (run git lfs pull)")
        else:
            try:
                joints = rlbench_data.load_demo(pkl)["joint_positions"]
            except pickle.UnpicklingError as e:
                problems.append(f"{rel}/low_dim_obs.pkl: refused by the safe loader ({e})")
            except Exception as e:
                problems.append(f"{rel}/low_dim_obs.pkl: can't read the observations "
                                f"({e.__class__.__name__}: {e})")
            if joints is not None:
                if joints.ndim != 2 or joints.shape[1] < 7:
                    problems.append(f"{rel}/low_dim_obs.pkl: joint_positions has shape {joints.shape}, "
                                    "needs 7 per observation")
                    joints = None
                elif not np.isfinite(joints).all():
                    bad = int((~np.isfinite(joints)).any(axis=1).argmax())
                    problems.append(f"{rel}/low_dim_obs.pkl: observation {bad} has a NaN or infinite joint position")
                    joints = None
                elif len(joints) < 2:
                    problems.append(f"{rel}/low_dim_obs.pkl: {len(joints)} observation; needs at least 2 "
                                    "(the target is the change to the next frame)")
                    joints = None
    if not os.path.isdir(rgb):
        problems.append(f"{rel}: no front_rgb/ folder")
        return problems, None
    names = [n for n in os.listdir(rgb) if not n.startswith(".")]
    steps = []
    for n in sorted(names):
        try:
            steps.append(rlbench_data.frame_step(n))
        except ValueError:
            problems.append(f"{rel}/front_rgb/{n}: not named <step>.png")
    steps.sort()
    if steps and steps != list(range(len(steps))):
        missing = sorted(set(range(steps[-1] + 1)) - set(steps))[:3]
        problems.append(f"{rel}/front_rgb: frames don't run 0 to {len(steps) - 1} without gaps "
                        f"(missing {', '.join(f'{m}.png' for m in missing)})")
    if joints is not None and steps and len(steps) != len(joints):
        problems.append(f"{rel}: {len(steps)} frames but {len(joints)} observations in low_dim_obs.pkl")
    from PIL import Image
    for s in steps:
        path = os.path.join(rgb, f"{s}.png")
        with open(path, "rb") as f:
            head = f.read(40)
        if head.startswith(b"version https://git-lfs"):
            problems.append(f"{rel}/front_rgb/{s}.png: Git LFS pointer, not the image (run git lfs pull)")
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB")
                if min(im.size) < MIN_SIDE:
                    problems.append(f"{rel}/front_rgb/{s}.png: too small ({im.size[0]}x{im.size[1]}), "
                                    f"needs at least {MIN_SIDE}x{MIN_SIDE}")
        except Exception as e:
            problems.append(f"{rel}/front_rgb/{s}.png: can't open as an image ({e.__class__.__name__})")
    return problems, (joints if not problems else None)


def check(folder):
    """Returns ({episode dir: joint positions}, [problems])."""
    if not os.path.isdir(folder):
        return {}, [f"{folder}: not a folder"]
    episodes = find_episodes(folder)
    if not episodes:
        return {}, [f"{folder}: no episodes (folders with front_rgb/ and low_dim_obs.pkl)"]
    good, problems = {}, []
    for ep in episodes:
        p, joints = check_episode(folder, ep)
        problems += p
        if joints is not None:
            good[ep] = joints
    return good, problems


def evaluate(episodes):
    """Test scores of every trained version on the episodes, as train.py computes them."""
    import logging

    import torch
    from PIL import Image

    import train
    from model import ModelManager
    logging.disable(logging.WARNING)
    rows, embeddings = [], {}
    for manifest in MANIFESTS:
        manager = ModelManager(os.path.join(ROOT, manifest))
        model = manager.get_model()
        x, y = [], []
        for ep, joints in episodes.items():
            parts = [joints[:-1, :model.telemetry_dim]]
            if model.use_vision:
                if ep not in embeddings:   # computed here, not from train.py's cache (keyed by episode name)
                    frames = [Image.open(rlbench_data.frame_path(ep, i)) for i in range(len(joints))]
                    embeddings[ep] = model.embed_images(frames).numpy().astype(np.float32)
                parts.insert(0, embeddings[ep][:-1])
            x.append(np.concatenate(parts, axis=1))
            y.append(rlbench_data.expert_actions(joints))
        x, y = np.concatenate(x), np.concatenate(y)
        with torch.no_grad():
            predicted = model(torch.tensor(x[:, :model.vision_embedding_dim]),
                              torch.tensor(x[:, model.vision_embedding_dim:])).numpy()
        train_mean = model.output_mean.numpy()   # the training data's average move
        inputs = ("front camera + " if model.use_vision else "") + "joint positions"
        rows.append((manager.get_version(), inputs, train.scores(predicted, y, train_mean), len(y)))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Check your own demonstrations and score the model versions on them.")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--check-only", action="store_true", help="check the files, run nothing")
    args = parser.parse_args()
    folder = os.path.abspath(args.dir)

    episodes, problems = check(folder)
    if problems:
        print(f"{len(problems)} problem(s) in {args.dir}; nothing was run:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    frames = sum(len(j) for j in episodes.values())
    print(f"{args.dir}: {len(episodes)} episodes, {frames} frames with telemetry. All checks passed.")
    if args.check_only:
        return

    rows = evaluate(episodes)
    print(f"\nScores on your demonstrations (the joint change each one made next):")
    print(f"{'version':9s} {'inputs':32s} {'MAE (rad)':>10s} {'R²':>7s} {'MAE if no movement':>19s}")
    for version, inputs, t, n in rows:
        print(f"{version:9s} {inputs:32s} {t['mae_rad']:>10.5f} {t['r2']:>7.3f} {t['mae_rad_if_no_movement']:>19.5f}")
    print(f"\n{rows[0][3]} frames from {len(episodes)} episodes. R² below 0: worse than always predicting "
          "the training data's average move.")


if __name__ == "__main__":
    main()
