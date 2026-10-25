#!/usr/bin/env python3
"""
Download RLBench sample data for the multimodal inference service.

    python download_sample_data.py                    # push_buttons, 25 demonstrations (349 MB download)
    python download_sample_data.py --task open_drawer

The demonstrations come from the RLBench 18-task set (PerAct protocol, 128x128 cameras) published at
https://huggingface.co/datasets/shihao1895/rlbench (pinned revision below; tagged MIT there). They
were generated with RLBench (https://github.com/stepjam/RLBench, Imperial College London licence:
non-commercial use). The archive is checked against its SHA-256 before anything is unpacked, and only
what the service reads is kept: the front camera frames and the pickled telemetry of each episode
(about 75 MB for push_buttons).

    rlbench-data/push_buttons/all_variations/episodes/episode0/
        front_rgb/0.png ... 112.png
        low_dim_obs.pkl              # joint positions etc., one observation per frame
        variation_descriptions.pkl   # the task in words, e.g. "push the maroon button, then ..."
        variation_number.pkl
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

REPO = "shihao1895/rlbench"
REVISION = "b44678905fb208c89a9bd4f398c21f840991c2a7"
SPLIT = "test"   # 25 demonstrations per task; the 100-demonstration train split is 1.5-6.5 GB per task
KEEP = ("front_rgb/", "low_dim_obs.pkl", "variation_descriptions.pkl", "variation_number.pkl")
ROOT = os.path.dirname(os.path.abspath(__file__))


def expected_file(task: str) -> dict:
    """Size and SHA-256 of the task archive at the pinned revision (from the Hugging Face API)."""
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REVISION}",
        data=json.dumps({"paths": [f"{SPLIT}/{task}.tar.xz"]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        info = json.load(response)
    if not info:
        sys.exit(f"No task named {task} in {REPO}")
    return {"size": info[0]["lfs"]["size"], "sha256": info[0]["lfs"]["oid"]}


def download(url: str, path: str, size: int) -> str:
    """Download with progress; returns the SHA-256."""
    digest = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(url, timeout=60) as response, open(path + ".part", "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            print(f"\rDownloading: {done * 100 // size}% ({done / 1e6:.0f} of {size / 1e6:.0f} MB)", end="", flush=True)
    print()
    os.replace(path + ".part", path)
    return digest.hexdigest()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def extract(archive: str, data_dir: str) -> int:
    """Unpack the front camera and telemetry files; returns the number of episodes."""
    episodes = set()
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar:
            parts = member.name.split("/")
            # <task>/all_variations/episodes/episodeN/<file or folder>/...
            if len(parts) < 5 or not member.isfile() or not "/".join(parts[4:]).startswith(KEEP):
                continue
            tar.extract(member, data_dir, filter="data")
            episodes.add(parts[3])
    return len(episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="push_buttons", help="RLBench task (default: push_buttons)")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "rlbench-data"))
    parser.add_argument("--keep-archive", action="store_true", help="keep the downloaded .tar.xz in .cache/")
    args = parser.parse_args()

    print("RLBench Sample Data Downloader")
    print("=" * 40)
    episodes_dir = os.path.join(args.data_dir, args.task, "all_variations", "episodes")
    if os.path.isdir(episodes_dir) and os.listdir(episodes_dir):
        print(f"Already there: {episodes_dir} ({len(os.listdir(episodes_dir))} episodes)")
        return

    expected = expected_file(args.task)
    cache = os.path.join(ROOT, ".cache", "downloads")
    os.makedirs(cache, exist_ok=True)
    archive = os.path.join(cache, f"{args.task}_{SPLIT}.tar.xz")
    if os.path.exists(archive) and sha256(archive) == expected["sha256"]:
        print(f"Using {archive}")
    else:
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SPLIT}/{args.task}.tar.xz"
        print(f"From {url}")
        if download(url, archive, expected["size"]) != expected["sha256"]:
            os.remove(archive)
            sys.exit("SHA-256 mismatch: the download is not the pinned file. Nothing was unpacked.")
        print("SHA-256 verified.")

    print("Unpacking the front camera frames and telemetry...")
    n = extract(archive, args.data_dir)
    if not args.keep_archive:
        os.remove(archive)
    print(f"{n} demonstrations of {args.task} in {episodes_dir}")
    print("Next: python train.py, then uvicorn service:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
