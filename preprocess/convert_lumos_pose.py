#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Lumos JSON demonstrations to LeRobot v3 with end-effector pose actions.

state  = [left_joint(6), left_feedback_m, right_joint(6), right_feedback_m]
action = [left_pose[:6], left_target_m, right_pose[:6], right_target_m]
   feedback_m = states.*_gripper[0] / (1.5*(0.046/0.117))
   target_m   = clip(actions.*_arm.pose[6], 0, 0.08)   # metres; 0=closed; use directly
   image  = cam_high / cam_left_wrist / cam_right_wrist, dtype=image
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

GRIPPER_MAX_OPENING_M = 0.08
GRIPPER_FEEDBACK_SCALE = 1.5 * (0.046 / 0.117)
CAMS = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_wrist_camera",
    "observation.images.cam_right_wrist": "right_wrist_camera",
}
STATE_NAMES = (
    [f"left_joint_{i+1}" for i in range(6)] + ["left_gripper"]
    + [f"right_joint_{i+1}" for i in range(6)] + ["right_gripper"]
)
ACTION_NAMES = (
    ["left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper"]
    + ["right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper"]
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="lumos/lumos_pose_20261010")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def frame_vecs(fr):
    st, ac = fr["states"], fr["actions"]
    lj = list(st["left_joint"]["qpos"])
    rj = list(st["right_joint"]["qpos"])
    lf = float(st["left_gripper"][0]) / GRIPPER_FEEDBACK_SCALE
    rf = float(st["right_gripper"][0]) / GRIPPER_FEEDBACK_SCALE
    state = np.asarray(lj + [lf] + rj + [rf], dtype=np.float32)

    lp = [float(x) for x in ac["left_arm"]["pose"]]
    rp = [float(x) for x in ac["right_arm"]["pose"]]
    # pose[6] is already the gripper opening in metres (0=closed); do not invert it.
    lt = max(0.0, min(GRIPPER_MAX_OPENING_M, lp[6]))
    rt = max(0.0, min(GRIPPER_MAX_OPENING_M, rp[6]))
    action = np.asarray(lp[:6] + [lt] + rp[:6] + [rt], dtype=np.float32)
    return state, action


def main():
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = sorted(d for d in os.listdir(args.src)
                 if d.startswith("zc0_touch_") and os.path.isdir(os.path.join(args.src, d)))
    if args.limit > 0:
        eps = eps[:args.limit]
    print(f"[info] episodes: {len(eps)}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_NAMES, "info": None},
        "action": {"dtype": "float32", "shape": (14,), "names": ACTION_NAMES, "info": None},
    }
    for k in CAMS:
        features[k] = {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"], "info": None}

    ds = LeRobotDataset.create(repo_id=args.repo, fps=args.fps, features=features,
                               root=os.path.abspath(args.out),
                               robot_type="station_lite", use_videos=False,
                               image_writer_threads=8)
    total = 0
    for ep in eps:
        epd = os.path.join(args.src, ep)
        jp = os.path.join(epd, "data.json")
        if not os.path.exists(jp):
            continue
        meta = json.load(open(jp, encoding="utf-8"))
        task = (meta.get("text") or {}).get("goal", "") or "teleop task"
        for fr in meta["data"]:
            st, ac = frame_vecs(fr)
            frame = {"observation.state": st, "action": ac, "task": task}
            for k, src_key in CAMS.items():
                img = np.asarray(Image.open(os.path.join(epd, fr["colors"][src_key])).convert("RGB")).copy()
                frame[k] = img
            ds.add_frame(frame)
        ds.save_episode()
        total += len(meta["data"])
        print(f"[ok] {ep}: {len(meta['data'])} frames (total {total})", flush=True)
    ds.finalize()
    print(f"[DONE] {args.out}  episodes={len(eps)} frames={total}")


if __name__ == "__main__":
    sys.exit(main())
