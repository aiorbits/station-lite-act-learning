#!/usr/bin/env python3
"""HTTP inference service for the Station Lite ACT checkpoint.

This process loads a policy only. It has no robot SDK imports and never sends
commands to hardware. The control-box client owns all robot communication.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


SCHEMA = "station_lite_act_http_v1"
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
ACTION_ENCODING = "absolute_xyz_euler_xyz_gripper_opening_m"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MODEL_IMAGE_SIZE = (640, 480)  # PIL order: (width, height); training images are 640x480 RGB.


class PolicyEngine:
    def __init__(self, checkpoint: Path, device: str):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.checkpoint = checkpoint.resolve()
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(str(self.checkpoint)).to(self.device).eval()
        processor_device = {"device_processor": {"device": str(self.device)}}
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides=processor_device,
            postprocessor_overrides=processor_device,
        )
        self.chunk_size = int(self.policy.config.chunk_size)
        self.lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, payload: dict[str, Any]) -> tuple[np.ndarray, float]:
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"schema must be {SCHEMA!r}")
        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError("state must contain 14 finite values")

        encoded = payload.get("images_jpeg_base64")
        if not isinstance(encoded, dict):
            raise ValueError("images_jpeg_base64 must be an object")

        observation: dict[str, torch.Tensor] = {
            "observation.state": torch.from_numpy(state).unsqueeze(0)
        }
        for key in CAMERA_KEYS:
            value = encoded.get(key)
            if not isinstance(value, str):
                raise ValueError(f"missing image: {key}")
            try:
                image_bytes = base64.b64decode(value, validate=True)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image = image.convert("RGB")
                    source_size = image.size
                    source_ratio = source_size[0] / source_size[1]
                    model_ratio = MODEL_IMAGE_SIZE[0] / MODEL_IMAGE_SIZE[1]
                    if abs(source_ratio - model_ratio) > 1e-3:
                        raise ValueError(
                            f"aspect ratio {source_size[0]}x{source_size[1]} does not match "
                            f"training ratio {MODEL_IMAGE_SIZE[0]}x{MODEL_IMAGE_SIZE[1]}"
                        )
                    if source_size != MODEL_IMAGE_SIZE:
                        image = image.resize(MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)
                    rgb = np.asarray(image, dtype=np.uint8).copy()
            except (binascii.Error, OSError, ValueError) as exc:
                raise ValueError(f"invalid JPEG for {key}: {exc}") from exc
            # The converted LeRobot dataset stores RGB inputs as float32 in [0, 1].
            image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
            observation[key] = image_tensor.unsqueeze(0)

        started = time.perf_counter()
        with self.lock:
            batch = self.preprocess(observation)
            actions = self.postprocess(self.policy.predict_action_chunk(batch))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = actions[0].detach().float().cpu().numpy()
        if result.shape != (self.chunk_size, 14) or not np.isfinite(result).all():
            raise RuntimeError(f"invalid policy output shape/content: {result.shape}")
        return result, elapsed_ms


def make_handler(engine: PolicyEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "StationLiteACT/1.0"

        def _send(self, status: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self._send(404, {"error": "not found"})
                return
            self._send(200, {
                "model_loaded": True,
                "checkpoint": str(engine.checkpoint),
                "device": str(engine.device),
                "chunk_size": engine.chunk_size,
                "camera_keys": list(CAMERA_KEYS),
                "hardware_connected": False,
            })

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/predict":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError(f"request size must be between 1 and {MAX_REQUEST_BYTES} bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request JSON must be an object")
                actions, inference_ms = engine.predict(payload)
                self._send(200, {
                    "schema": SCHEMA,
                    "session_id": payload.get("session_id"),
                    "seq": payload.get("seq"),
                    "action_encoding": ACTION_ENCODING,
                    "chunk_size": engine.chunk_size,
                    "actions": actions.tolist(),
                    "inference_ms": round(inference_ms, 3),
                    "hardware_connected": False,
                })
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # return concise failure; full trace stays in server log
                print(f"[predict error] {type(exc).__name__}: {exc}", flush=True)
                self._send(500, {"error": f"inference failed: {type(exc).__name__}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    engine = PolicyEngine(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"[ready] checkpoint={engine.checkpoint}", flush=True)
    print(f"[ready] device={engine.device} chunk_size={engine.chunk_size}", flush=True)
    print(f"[ready] listening on {args.host}:{args.port}; robot hardware is not connected", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stop] shutting down inference service", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
