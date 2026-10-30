#!/usr/bin/env python3
"""Evaluate ACT action chunks against held-out demonstration actions offline.

This script reads LeRobot observations and the matching raw episode JSON files.
It does not import a robot SDK, connect to hardware, or send robot commands.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


GRIPPER_MAX_OPENING_M = 0.08
ACTION_NAMES = (
    ["left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_gripper"]
    + ["right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_gripper"]
)
HORIZON_BINS = ((0, 0, "0"), (1, 9, "1-9"), (10, 24, "10-24"), (25, 49, "25-49"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[26, 27, 28, 29])
    parser.add_argument("--stride", type=int, default=10, help="Frame spacing between chunk starts")
    parser.add_argument("--fps", type=float, default=30.0, help="Nominal FPS, used only for approximate times")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--euler-sequence",
        choices=("xyz", "XYZ"),
        default="xyz",
        help="Euler convention for rotation error; xyz is extrinsic, XYZ is intrinsic.",
    )
    return parser.parse_args()


def as_int(value: Any) -> int:
    return int(value.item()) if torch.is_tensor(value) else int(value)


def raw_action(frame: dict[str, Any]) -> np.ndarray:
    """Match the toolkit converter: each arm pose[:6] plus clipped pose[6] in metres."""
    actions = frame["actions"]
    left = [float(value) for value in actions["left_arm"]["pose"]]
    right = [float(value) for value in actions["right_arm"]["pose"]]
    if len(left) < 7 or len(right) < 7:
        raise ValueError("Expected 7 values in each arm action pose")
    left_gripper = float(np.clip(left[6], 0.0, GRIPPER_MAX_OPENING_M))
    right_gripper = float(np.clip(right[6], 0.0, GRIPPER_MAX_OPENING_M))
    return np.asarray(left[:6] + [left_gripper] + right[:6] + [right_gripper], dtype=np.float32)


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rotation_matrix(euler_xyz: np.ndarray, sequence: str) -> np.ndarray:
    x, y, z = (float(v) for v in euler_xyz)
    if sequence == "xyz":
        return rot_z(z) @ rot_y(y) @ rot_x(x)
    return rot_x(x) @ rot_y(y) @ rot_z(z)


def geodesic_degrees(pred_euler: np.ndarray, target_euler: np.ndarray, sequence: str) -> float:
    pred_rot = rotation_matrix(pred_euler, sequence)
    target_rot = rotation_matrix(target_euler, sequence)
    relative = target_rot.T @ pred_rot
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def load_raw_episodes(raw_root: Path) -> list[dict[str, Any]]:
    paths = sorted(
        path for path in raw_root.iterdir()
        if path.is_dir() and path.name.startswith("zc0_touch_")
    )
    if not paths:
        raise FileNotFoundError(f"No zc0_touch_* episode folders found under {raw_root}")

    episodes = []
    global_start = 0
    for path in paths:
        json_path = path / "data.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing {json_path}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        frames = payload.get("data", [])
        episodes.append({
            "name": path.name,
            "path": path,
            "frames": frames,
            "global_start": global_start,
            "length": len(frames),
        })
        global_start += len(frames)
    return episodes


def make_observation(row: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0)
        for key, value in row.items()
        if key.startswith("observation.")
    }


def infer_chunk(policy: Any, preprocess: Any, postprocess: Any, row: dict[str, Any]) -> np.ndarray:
    observation = make_observation(row)
    with torch.inference_mode():
        batch = preprocess(observation)
        actions = postprocess(policy.predict_action_chunk(batch))
    result = actions[0].detach().float().cpu().numpy()
    if result.ndim != 2 or result.shape[1] != 14:
        raise ValueError(f"Expected action chunk shape (T, 14), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Predicted chunk contains NaN or Inf")
    return result


def summarize(rows: list[dict[str, Any]], output_path: Path) -> None:
    fields = [
        "episode_index", "episode_name", "start_frame", "start_time_s",
        "horizon", "target_frame", "target_time_s",
        "position_mae_m", "left_gripper_abs_error_m", "right_gripper_abs_error_m",
        "left_rotation_error_deg", "right_rotation_error_deg",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nMetrics by episode and prediction horizon:")
    for episode in sorted({row["episode_index"] for row in rows}):
        for lo, hi, label in HORIZON_BINS:
            group = [
                row for row in rows
                if row["episode_index"] == episode and lo <= row["horizon"] <= hi
            ]
            if not group:
                continue
            pos = np.asarray([r["position_mae_m"] for r in group])
            lg = np.asarray([r["left_gripper_abs_error_m"] for r in group])
            rg = np.asarray([r["right_gripper_abs_error_m"] for r in group])
            lr = np.asarray([r["left_rotation_error_deg"] for r in group])
            rr = np.asarray([r["right_rotation_error_deg"] for r in group])
            print(
                f"Episode {episode:02d}, horizon {label:>5}: n={len(group):4d} | "
                f"position MAE={pos.mean():.4f} m | "
                f"gripper L/R={lg.mean():.4f}/{rg.mean():.4f} m | "
                f"rotation L/R={lr.mean():.2f}/{rr.mean():.2f} deg "
                f"(right P95={np.percentile(rr, 95):.2f} deg)"
            )


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    raw_episodes = load_raw_episodes(args.raw_root)
    selected = set(args.episodes)
    if min(selected) < 0 or max(selected) >= len(raw_episodes):
        raise IndexError(f"Episode indices must be in [0, {len(raw_episodes) - 1}]")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    policy = ACTPolicy.from_pretrained(str(args.checkpoint)).to("cuda").eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, pretrained_path=str(args.checkpoint)
    )
    chunk_size = int(policy.config.chunk_size)
    if chunk_size != 50:
        print(f"[info] Checkpoint chunk_size={chunk_size}; evaluating its full chunk")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "chunk_predictions.csv"
    detail_fields = [
        "episode_index", "episode_name", "start_frame", "start_time_s",
        "horizon", "target_frame", "target_time_s",
    ]
    for index, name in enumerate(ACTION_NAMES):
        detail_fields.extend([f"pred_{name}", f"demo_{name}", f"abs_error_{name}"])

    summary_rows = []
    windows = 0
    with detail_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()

        for episode_index in sorted(selected):
            episode = raw_episodes[episode_index]
            target_actions = np.stack([raw_action(frame) for frame in episode["frames"]])
            max_start = len(target_actions) - chunk_size
            if max_start < 0:
                print(f"[skip] Episode {episode_index} is shorter than the action chunk")
                continue

            print(
                f"[episode] {episode_index}: {episode['name']} | "
                f"frames={len(target_actions)} | global_start={episode['global_start']}"
            )
            for start_frame in range(0, max_start + 1, args.stride):
                global_index = episode["global_start"] + start_frame
                row = dataset[global_index]
                actual_episode = as_int(row["episode_index"])
                actual_frame = as_int(row["frame_index"]) if "frame_index" in row else start_frame
                if actual_episode != episode_index or actual_frame != start_frame:
                    raise ValueError(
                        "Raw episode ordering does not match LeRobot indexing: "
                        f"expected episode/frame {episode_index}/{start_frame}, "
                        f"got {actual_episode}/{actual_frame} at global row {global_index}"
                    )
                if not np.allclose(
                    row["action"].detach().float().cpu().numpy(),
                    target_actions[start_frame], atol=1e-5, rtol=1e-5,
                ):
                    raise ValueError(
                        f"Raw action does not match LeRobot action at episode {episode_index}, "
                        f"frame {start_frame}; check converter/gripper semantics"
                    )

                predicted = infer_chunk(policy, preprocess, postprocess, row)
                if predicted.shape[0] < chunk_size:
                    raise ValueError(f"Expected {chunk_size} predicted steps, got {predicted.shape[0]}")
                predicted = predicted[:chunk_size]
                demonstrated = target_actions[start_frame:start_frame + chunk_size]

                for horizon in range(chunk_size):
                    pred = predicted[horizon]
                    demo = demonstrated[horizon]
                    target_frame = start_frame + horizon
                    pos_error = np.abs(pred[[0, 1, 2, 7, 8, 9]] - demo[[0, 1, 2, 7, 8, 9]])
                    row_out = {
                        "episode_index": episode_index,
                        "episode_name": episode["name"],
                        "start_frame": start_frame,
                        "start_time_s": start_frame / args.fps,
                        "horizon": horizon,
                        "target_frame": target_frame,
                        "target_time_s": target_frame / args.fps,
                    }
                    for dim, name in enumerate(ACTION_NAMES):
                        row_out[f"pred_{name}"] = float(pred[dim])
                        row_out[f"demo_{name}"] = float(demo[dim])
                        row_out[f"abs_error_{name}"] = float(abs(pred[dim] - demo[dim]))
                    writer.writerow(row_out)

                    summary_rows.append({
                        "episode_index": episode_index,
                        "episode_name": episode["name"],
                        "start_frame": start_frame,
                        "start_time_s": start_frame / args.fps,
                        "horizon": horizon,
                        "target_frame": target_frame,
                        "target_time_s": target_frame / args.fps,
                        "position_mae_m": float(pos_error.mean()),
                        "left_gripper_abs_error_m": float(abs(pred[6] - demo[6])),
                        "right_gripper_abs_error_m": float(abs(pred[13] - demo[13])),
                        "left_rotation_error_deg": geodesic_degrees(pred[3:6], demo[3:6], args.euler_sequence),
                        "right_rotation_error_deg": geodesic_degrees(pred[10:13], demo[10:13], args.euler_sequence),
                    })
                windows += 1
                print(f"  chunk start frame {start_frame}: inferred {chunk_size} steps")

    if not summary_rows:
        raise RuntimeError("No chunks were evaluated")
    summary_path = args.output_dir / "chunk_horizon_summary.csv"
    summarize(summary_rows, summary_path)
    print(f"\nCompleted windows: {windows}")
    print(f"Detailed predictions: {detail_path}")
    print(f"Horizon summary: {summary_path}")
    print(f"Euler assumption used for rotation metrics: {args.euler_sequence}")
    print("This is offline evaluation only; no robot commands were issued.")


if __name__ == "__main__":
    main()
