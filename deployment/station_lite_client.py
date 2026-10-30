#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACT closed-loop client for Station Lite using asynchronous waypoint chunks.

Flow:
  Three camera images and both arm states -> POST /predict -> a (50, 14) pose chunk
  -> IK per waypoint using the previous solution as a seed, with retreat on failure
  -> robot.left/right_arm.update_joint_waypoint_chunk_with_gripper(...)
     The SDK call is non-blocking. It retains a trajectory prefix and replaces the
     future segment, which the SDK interpolates at 400 Hz.
  -> At the switch point (after K waypoints), observe again and request a new chunk.

Gripper: policy output is opening distance in metres (0=closed); the SDK receives
normalized position = metres / 0.085.

Safety: dry-run is the default. Robot motion requires --execute. At exit, the client
returns both arms home before releasing the motors.
"""
import argparse
import base64
import io
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np

from teleimager.image_client import ImageClient  # noqa: E402

SCHEMA = "station_lite_act_http_v1"
JOINT_LOW = np.array([-2.824, -0.366, -3.193, -1.710, -1.710, -2.856])
JOINT_UP = np.array([2.824, 3.508, 0.000, 1.710, 1.710, 2.856])
GRIPPER_MAX_M = 0.085  # Full gripper opening in metres, matching the installed SDK.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="Reachable policy server base URL")
    p.add_argument("--img-host", default="127.0.0.1")
    p.add_argument("--rate", type=float, default=30.0, help="Policy action rate in Hz")
    p.add_argument("--loops", type=int, default=40, help="Number of prediction and execution cycles")
    p.add_argument("--speed", type=float, default=0.3, help="Playback speed multiplier; larger is faster")
    p.add_argument("--execute", action="store_true", help="Enable robot motion; without this flag, run inference only")
    p.add_argument("--go-home", action="store_true", help="Move both arms home before starting")
    p.add_argument("--margin", type=float, default=0.05, help="Joint-limit margin in radians")
    p.add_argument("--exec-steps", type=int, default=15,
                   help="Joint waypoints per command M (must not exceed policy chunk length)")
    p.add_argument("--switch-after", type=int, default=5,
                   help="Observe and send a new chunk after K waypoints; requires 0 < K < M")
    p.add_argument("--switch-delay", type=float, default=0.05,
                   help="SDK chunk switch delay in seconds")
    p.add_argument("--max-ik-fail", type=int, default=20, help="Maximum consecutive IK failures")
    p.add_argument("--no-end-home", action="store_true",
                   help="Do not return home at exit; by default, return home before releasing motors")
    p.add_argument("--blend-overlap", type=int, default=0,
                   help="Number of overlap steps for blending (0 disables blending); blends the old tail with the new prediction")
    p.add_argument("--blend-mode", default="cosine", choices=["linear", "cosine", "minjerk"],
                   help="Blend weight curve")
    p.add_argument("--save-frames", default="")
    return p.parse_args()


def enc_jpeg_b64(img):
    if isinstance(img, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(img)).decode("ascii")
    from PIL import Image
    img = np.asarray(img).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def wait_frame(getter, timeout=6.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        img, _ = getter()
        if img is not None:
            return img
        time.sleep(0.05)
    raise RuntimeError("no frame from camera (timeout)")


def clip_limits(q, lo, hi):
    return np.minimum(np.maximum(np.asarray(q, dtype=float), lo), hi)


def solve_with_retreat(arm, pos, euler, seed):
    """Solve IK for the target; on failure, retreat toward the current end-effector pose.

    Returns (q, ok, alpha), where alpha=1 means the original target was reached and
    alpha<1 means the target was reduced toward the current pose.
    """
    q, ok = arm.solve_ik_euler(list(pos), list(euler), list(seed))
    if ok:
        return np.asarray(q, dtype=float).reshape(-1), True, 1.0
    try:
        cur_pos, cur_euler = arm.get_ee_pose_euler()
        cur_pos = np.asarray(cur_pos, dtype=float).reshape(3)
        cur_euler = np.asarray(cur_euler, dtype=float).reshape(3)
        tpos = np.asarray(pos, dtype=float).reshape(3)
        teul = np.asarray(euler, dtype=float).reshape(3)
        d_e = (teul - cur_euler + np.pi) % (2 * np.pi) - np.pi
        for a in (0.7, 0.4, 0.15):
            p = cur_pos + a * (tpos - cur_pos)
            e = cur_euler + a * d_e
            q2, ok2 = arm.solve_ik_euler(list(p), list(e), list(seed))
            if ok2:
                return np.asarray(q2, dtype=float).reshape(-1), True, a
    except Exception:
        pass
    return np.asarray(seed, dtype=float).reshape(-1), False, 0.0


EULER_IDX = (3, 4, 5, 10, 11, 12)
GRIPPER_IDX = (6, 13)


def _blend_alpha(i, ov, mode):
    s = (i + 1) / float(ov)
    if mode == "linear":
        return s
    if mode == "minjerk":
        return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return 0.5 * (1.0 - math.cos(math.pi * s))  # cosine


def blend_head(old_next, new_head, mode="cosine"):
    """Blend the next old reference segment with the start of the new prediction.

    The blend weight increases from 0 to 1. Gripper values use the new prediction;
    Euler angles are blended using wrapped angular differences. Returns (ov, 14).
    """
    ov = int(min(len(old_next), len(new_head)))
    out = np.array(new_head[:ov], dtype=float).copy()
    for i in range(ov):
        a = _blend_alpha(i, ov, mode)
        o = np.asarray(old_next[i], dtype=float)
        n = out[i]
        for j in range(o.shape[0]):
            if j in GRIPPER_IDX:
                continue  # Use the new value.
            if j in EULER_IDX:
                d = (n[j] - o[j] + math.pi) % (2 * math.pi) - math.pi
                out[i, j] = o[j] + a * d
            else:
                out[i, j] = (1.0 - a) * o[j] + a * n[j]
    return out


def main():
    args = parse_args()
    lo = JOINT_LOW + args.margin
    hi = JOINT_UP - args.margin
    M = max(2, int(args.exec_steps))
    K = int(args.switch_after)
    if not (0 < K < M):
        K = max(1, min(K, M - 1))
    T = M / args.rate / max(args.speed, 1e-3)
    t_switch = (K / M) * T
    print(f"server={args.server} loops={args.loops} execute={args.execute} speed={args.speed} "
          f"M={M} K={K} T={T:.3f}s t_switch={t_switch:.3f}s switch_delay={args.switch_delay} "
          f"margin={args.margin}", flush=True)

    robot = None
    if args.execute:
        from xr_teleoperate2.robot_control.robot_dual_touch import DualTouchRobot
        print("Initializing DualTouchRobot (enabling both arms)...", flush=True)
        robot = DualTouchRobot()
        time.sleep(2.0)
        if args.go_home:
            print("Calling go_home() before inference...", flush=True)
            robot.go_home()
            time.sleep(4.0)

    img_client = ImageClient(host=args.img_host, auto_subscribe=False)
    img_client.start_subscriptions()

    def read_state():
        ql = np.asarray(robot.left_arm.get_joint_positions(), dtype=float).reshape(-1)
        qr = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
        gl = float(robot.left_arm.get_gripper_distance())
        gr = float(robot.right_arm.get_gripper_distance())
        return np.concatenate([ql, [gl], qr, [gr]])

    prev_seg = None
    try:
        for it in range(args.loops):
            head = wait_frame(img_client.get_head_frame)
            lw = wait_frame(img_client.get_left_wrist_frame)
            rw = wait_frame(img_client.get_right_wrist_frame)
            imgs = {"observation.images.cam_high": head,
                    "observation.images.cam_left_wrist": lw,
                    "observation.images.cam_right_wrist": rw}
            if args.save_frames:
                os.makedirs(args.save_frames, exist_ok=True)
                d = imgs["observation.images.cam_high"]
                out = f"{args.save_frames}/loop{it:03d}_head.jpg"
                if isinstance(d, (bytes, bytearray, memoryview)):
                    open(out, "wb").write(bytes(d))
                else:
                    from PIL import Image
                    Image.fromarray(np.asarray(d)).save(out)

            state = read_state() if args.execute else np.zeros(14)
            payload = {"schema": SCHEMA, "session_id": "cl-wc", "seq": it,
                       "state": [float(x) for x in state],
                       "images_jpeg_base64": {k: enc_jpeg_b64(v) for k, v in imgs.items()}}
            t0 = time.monotonic()
            req = urllib.request.Request(args.server + "/predict",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                actions = np.asarray(json.loads(resp.read().decode())["actions"], dtype=float)
            infer_ms = (time.monotonic() - t0) * 1000
            print(f"[loop {it}] {actions.shape} infer={infer_ms:.0f}ms "
                  f"L_xyz0={np.round(actions[0,0:3],3).tolist()} Lg0={actions[0,6]:.3f} "
                  f"R_xyz0={np.round(actions[0,7:10],3).tolist()} Rg0={actions[0,13]:.3f}", flush=True)

            if not args.execute:
                continue

            # ---- Blend the old trajectory tail with the new prediction. ----
            new_seg = actions[:M].copy()
            ov = int(min(max(args.blend_overlap, 0), M - K))
            if prev_seg is not None and ov > 0:
                head = blend_head(prev_seg[K:K + ov], new_seg[:ov], args.blend_mode)
                send_seg = np.vstack([head, new_seg[ov:M]])
            else:
                send_seg = new_seg
            prev_seg = send_seg.copy()

            seed_l = np.asarray(robot.left_arm.get_joint_positions(), dtype=float).reshape(-1)
            seed_r = np.asarray(robot.right_arm.get_joint_positions(), dtype=float).reshape(-1)
            jp_l, jp_r, gl, gr = [], [], [], []
            ik_fail = 0
            for a in send_seg:
                ql, okl, al = solve_with_retreat(robot.left_arm, a[0:3], a[3:6], seed_l)
                if okl:
                    ql = clip_limits(ql, lo, hi)
                    seed_l = ql
                    if al < 1.0:
                        print(f"  [L retreat a={al}] pos={np.round(a[0:3],3).tolist()}", flush=True)
                else:
                    ql = seed_l.copy()
                    ik_fail += 1
                    print(f"  [IK FAIL L] pos={np.round(a[0:3],4).tolist()} euler={np.round(a[3:6],4).tolist()}", flush=True)
                jp_l.append([float(x) for x in ql])
                gl.append(float(np.clip(a[6] / GRIPPER_MAX_M, 0.0, 1.0)))

                qr, okr, ar = solve_with_retreat(robot.right_arm, a[7:10], a[10:13], seed_r)
                if okr:
                    qr = clip_limits(qr, lo, hi)
                    seed_r = qr
                    if ar < 1.0:
                        print(f"  [R retreat a={ar}] pos={np.round(a[7:10],3).tolist()}", flush=True)
                else:
                    qr = seed_r.copy()
                    ik_fail += 1
                    print(f"  [IK FAIL R] pos={np.round(a[7:10],4).tolist()} euler={np.round(a[10:13],4).tolist()}", flush=True)
                jp_r.append([float(x) for x in qr])
                gr.append(float(np.clip(a[13] / GRIPPER_MAX_M, 0.0, 1.0)))

            t1 = time.monotonic()
            try:
                robot.left_arm.update_joint_waypoint_chunk_with_gripper(
                    jp_l, gl, time_sec=T, switch_delay_sec=args.switch_delay)
                robot.right_arm.update_joint_waypoint_chunk_with_gripper(
                    jp_r, gr, time_sec=T, switch_delay_sec=args.switch_delay)
            except Exception as e:
                print("  chunk update err:", repr(e), flush=True)
            install_ms = (time.monotonic() - t1) * 1000
            print(f"  [chunk] points={M} T={T:.2f}s install={install_ms:.0f}ms ik_fail={ik_fail} "
                  f"blend_ov={ov}/{args.blend_mode} wait_to_switch={t_switch:.2f}s", flush=True)

            if ik_fail > args.max_ik_fail:
                print(f"  !! Aborting safely after {ik_fail} consecutive IK failures", flush=True)
                break
            sl = t_switch - (time.monotonic() - t1)
            if sl > 0:
                time.sleep(sl)
    finally:
        if robot is not None and args.execute and not args.no_end_home:
            try:
                print("Returning both arms home before releasing motors...", flush=True)
                robot.go_home()
                time.sleep(4.0)
            except Exception as e:
                print("end go_home err:", repr(e), flush=True)
        try:
            img_client.close()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.cleanup()
            except Exception:
                pass
        print("done", flush=True)


if __name__ == "__main__":
    main()
