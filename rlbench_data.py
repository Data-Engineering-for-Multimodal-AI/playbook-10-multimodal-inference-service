"""
Reading RLBench demonstrations without installing RLBench.

An episode folder holds the camera frames (front_rgb/0.png, 1.png, ...) and low_dim_obs.pkl, a pickled
rlbench.demo.Demo whose `_observations` list has one Observation per frame: joint_positions (7),
joint_velocities (7), gripper_open, gripper_pose and more.

Unpickling can run code, so the loader only accepts the classes these files need (the two RLBench
classes, rebuilt as plain objects, and numpy arrays) and refuses anything else.
"""
import os
import pickle
import re
from glob import glob

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.environ.get("RLBENCH_DATA", os.path.join(REPO_ROOT, "rlbench-data")))
DEFAULT_TASK = "push_buttons"

_RLBENCH_CLASSES = {("rlbench.demo", "Demo"), ("rlbench.backend.observation", "Observation")}
_NUMPY_CLASSES = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
}
_PLAIN = {name: type(name, (), {}) for _, name in _RLBENCH_CLASSES}


class SafeUnpickler(pickle.Unpickler):
    """Unpickles RLBench demos and numpy arrays only."""

    def find_class(self, module, name):
        if (module, name) in _RLBENCH_CLASSES:
            return _PLAIN[name]
        if (module, name) in _NUMPY_CLASSES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a pickle")


def load_demo(pkl_path: str) -> dict:
    """Telemetry of one episode as arrays with one row per frame."""
    with open(pkl_path, "rb") as f:
        demo = SafeUnpickler(f).load()
    observations = getattr(demo, "_observations", None)
    if not observations:
        raise ValueError(f"{pkl_path} has no observations")
    return {
        "joint_positions": np.array([o.joint_positions for o in observations], dtype=np.float32),
        "joint_velocities": np.array([o.joint_velocities for o in observations], dtype=np.float32),
        "gripper_open": np.array([o.gripper_open for o in observations], dtype=np.float32),
    }


def expert_actions(joint_positions: np.ndarray) -> np.ndarray:
    """What the expert did after each frame: the change in the 7 joint positions to the next frame."""
    return joint_positions[1:] - joint_positions[:-1]


def episode_dirs(task: str = DEFAULT_TASK, data_root: str = DATA_ROOT) -> list:
    """Episode folders of a task, in episode-number order."""
    dirs = glob(os.path.join(data_root, task, "all_variations", "episodes", "episode*"))
    return sorted((d for d in dirs if os.path.exists(os.path.join(d, "low_dim_obs.pkl"))),
                  key=lambda d: int(re.search(r"(\d+)$", d).group(1)))


def frame_path(episode_dir: str, step: int) -> str:
    return os.path.join(episode_dir, "front_rgb", f"{step}.png")


def frame_step(png_path: str) -> int:
    """front_rgb/12.png -> 12"""
    name, extension = os.path.splitext(os.path.basename(png_path))
    if extension != ".png" or not name.isdigit():
        raise ValueError(f"expected a camera frame named <step>.png, got {os.path.basename(png_path)}")
    return int(name)


def require_data(task: str = DEFAULT_TASK) -> list:
    dirs = episode_dirs(task)
    if not dirs:
        raise FileNotFoundError(f"No RLBench episodes in {DATA_ROOT}/{task}. "
                                "Run: python download_sample_data.py")
    return dirs


def parse_episodes(spec, dirs: list) -> list:
    """'0-19' or [20, 21] -> the episode folders with those numbers."""
    if isinstance(spec, str):
        lo, hi = (int(x) for x in spec.split("-"))
        numbers = set(range(lo, hi + 1))
    else:
        numbers = set(spec)
    return [d for d in dirs if int(re.search(r"(\d+)$", d).group(1)) in numbers]
