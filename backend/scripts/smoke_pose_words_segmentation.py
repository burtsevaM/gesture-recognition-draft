#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check pose_words segmentation WS payload")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8000/ws/stream")
    parser.add_argument("--frames", type=int, default=24, help="How many frames to send")
    parser.add_argument("--image", type=Path, default=None, help="Optional source image for repeated sending")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def _make_jpeg_bytes(args: argparse.Namespace) -> bytes:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for smoke script") from exc

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise FileNotFoundError(f"failed to read image: {args.image}")
    else:
        img = np.zeros((int(args.height), int(args.width), 3), dtype=np.uint8)

    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("failed to encode jpg")
    return bytes(encoded.tobytes())


async def run_smoke(args: argparse.Namespace) -> int:
    try:
        import websockets
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("websockets package is required (installed via uvicorn[standard])") from exc

    frame_bytes = _make_jpeg_bytes(args)
    seen_segments = False
    last_payload: dict | None = None

    async with websockets.connect(args.ws_url, max_size=2**24) as ws:
        for _ in range(max(1, int(args.frames))):
            await ws.send(frame_bytes)
            raw = await ws.recv()
            payload = json.loads(raw)
            last_payload = payload
            if isinstance(payload, dict) and "segments" in payload:
                seen_segments = True
                if payload.get("segments"):
                    break

    if last_payload is not None:
        print(f"last_status={last_payload.get('status')}")
        print(f"last_mode={last_payload.get('mode')}")
        print(f"has_segments={'segments' in last_payload}")

    if seen_segments:
        print("smoke_result=ok")
        return 0
    print("smoke_result=fail (segments field not observed)")
    return 1


def main() -> int:
    args = parse_args()
    return asyncio.run(run_smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
