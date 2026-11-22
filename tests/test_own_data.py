"""Checks for the "Use your own data" adapter (scripts/own_episodes.py). Builds episodes in tmp_path;
no models and no downloaded data needed."""

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from own_episodes import check  # noqa: E402


class Observation:
    pass


class Demo:
    pass


# Pickle them under the RLBench names the safe loader accepts
Observation.__module__, Observation.__qualname__ = "rlbench.backend.observation", "Observation"
Demo.__module__, Demo.__qualname__ = "rlbench.demo", "Demo"
sys.modules.setdefault("rlbench", type(sys)("rlbench"))
sys.modules.setdefault("rlbench.demo", type(sys)("rlbench.demo")).Demo = Demo
sys.modules.setdefault("rlbench.backend", type(sys)("rlbench.backend"))
sys.modules.setdefault("rlbench.backend.observation", type(sys)("rlbench.backend.observation")).Observation = Observation


def write_demo(path, joints):
    demo = Demo()
    demo._observations = []
    for row in joints:
        o = Observation()
        o.joint_positions = np.asarray(row, dtype=np.float64)
        o.joint_velocities = np.zeros(7)
        o.gripper_open = 1.0
        demo._observations.append(o)
    with open(path, "wb") as f:
        pickle.dump(demo, f)


def make_episode(folder, name="episode0", frames=4, joints=None):
    ep = folder / name
    (ep / "front_rgb").mkdir(parents=True)
    for i in range(frames):
        Image.new("RGB", (64, 64), (i * 20, 100, 50)).save(ep / "front_rgb" / f"{i}.png")
    write_demo(ep / "low_dim_obs.pkl", np.linspace(0, 1, 7 * frames).reshape(frames, 7) if joints is None else joints)
    return ep


def test_good_folder_and_rlbench_layout(tmp_path):
    make_episode(tmp_path, "episode0")
    make_episode(tmp_path / "my_task" / "all_variations" / "episodes", "episode1", frames=5)
    episodes, problems = check(str(tmp_path))
    assert problems == []
    assert sorted(len(j) for j in episodes.values()) == [4, 5]


def test_missing_frame_and_count_mismatch(tmp_path):
    ep = make_episode(tmp_path)
    (ep / "front_rgb" / "2.png").unlink()
    _, problems = check(str(tmp_path))
    assert "episode0/front_rgb: frames don't run 0 to 2 without gaps (missing 2.png)" in problems
    assert "episode0: 3 frames but 4 observations in low_dim_obs.pkl" in problems


@pytest.mark.parametrize("change,expected", [
    (lambda ep: (ep / "front_rgb" / "1.png").write_bytes(b"version https://git-lfs.github.com/spec/v1\n"),
     "episode0/front_rgb/1.png: Git LFS pointer"),
    (lambda ep: (ep / "front_rgb" / "1.png").write_bytes(b"not a png"), "can't open as an image"),
    (lambda ep: Image.new("RGB", (16, 16)).save(ep / "front_rgb" / "1.png"), "too small (16x16)"),
    (lambda ep: Image.new("RGB", (64, 64)).save(ep / "front_rgb" / "first.png"), "first.png: not named <step>.png"),
    (lambda ep: (ep / "low_dim_obs.pkl").unlink(), "episode0: no low_dim_obs.pkl"),
    (lambda ep: (ep / "low_dim_obs.pkl").write_bytes(pickle.dumps({"a": __import__("collections").OrderedDict()})), "refused by the safe loader"),
    (lambda ep: write_demo(ep / "low_dim_obs.pkl", np.zeros((4, 3))), "needs 7 per observation"),
    (lambda ep: write_demo(ep / "low_dim_obs.pkl", np.full((4, 7), np.nan)), "NaN or infinite joint position"),
])
def test_problems_are_reported(tmp_path, change, expected):
    change(make_episode(tmp_path))
    _, problems = check(str(tmp_path))
    assert any(expected in p for p in problems), problems


def test_malicious_pickle_is_not_run(tmp_path):
    ep = make_episode(tmp_path)
    marker = tmp_path / "ran"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    (ep / "low_dim_obs.pkl").write_bytes(pickle.dumps(Evil()))
    _, problems = check(str(tmp_path))
    assert any("refused by the safe loader" in p for p in problems)
    assert not marker.exists()


def test_empty_folder(tmp_path):
    assert "no episodes" in check(str(tmp_path))[1][0]


def test_check_only_runs_nothing_on_problems(tmp_path):
    ep = make_episode(tmp_path / "in")
    (ep / "low_dim_obs.pkl").unlink()
    run = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "own_episodes.py"), "--dir", str(tmp_path / "in")],
                         capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 1
    assert "nothing was run" in run.stdout and "Scores" not in run.stdout
