#!/usr/bin/env python3
"""Check raw Station Lite episodes before conversion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")


def finite_vector(value, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) < length:
        raise ValueError(f"{label}: expected at least {length} values")
    values = [float(x) for x in value[:length]]
    if not all(math.isfinite(x) for x in values):
        raise ValueError(f"{label}: contains a non-finite value")
    return values


def check_episode(path: Path) -> tuple[int, float, float]:
    payload = json.loads((path / "data.json").read_text(encoding="utf-8"))
    frames = payload.get("data")
    if not isinstance(frames, list) or not frames:
        raise ValueError("data.json has no non-empty 'data' frame list")

    grippers: list[float] = []
    for frame_index, frame in enumerate(frames):
        where = f"frame {frame_index}"
        states = frame["states"]
        actions = frame["actions"]
        finite_vector(states["left_joint"]["qpos"], 6, f"{where} left joints")
        finite_vector(states["right_joint"]["qpos"], 6, f"{where} right joints")
        finite_vector(states["left_gripper"], 1, f"{where} left gripper")
        finite_vector(states["right_gripper"], 1, f"{where} right gripper")
        for arm in ("left_arm", "right_arm"):
            pose = finite_vector(actions[arm]["pose"], 7, f"{where} {arm} action")
            grippers.append(min(0.08, max(0.0, pose[6])))

        colors = frame["colors"]
        for camera in CAMERAS:
            image_path = path / colors[camera]
            if not image_path.is_file():
                raise ValueError(f"{where}: missing image {image_path.name}")
            with Image.open(image_path) as image:
                image.verify()

    return len(frames), min(grippers), max(grippers)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Folder containing zc0_touch_* episode folders")
    args = parser.parse_args()
    episodes = sorted(p for p in args.root.glob("zc0_touch_*") if p.is_dir())
    if not episodes:
        parser.error(f"No zc0_touch_* episode folders found under {args.root}")

    total_frames = 0
    errors = 0
    for episode in episodes:
        try:
            count, grip_min, grip_max = check_episode(episode)
            total_frames += count
            print(f"[OK] {episode.name}: {count} frames; action gripper range {grip_min:.3f}–{grip_max:.3f} m")
        except Exception as exc:  # report every bad episode to aid data cleanup
            errors += 1
            print(f"[FAIL] {episode.name}: {type(exc).__name__}: {exc}")

    print(f"Episodes: {len(episodes)}; valid frames: {total_frames}; failed episodes: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
