#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np


@dataclass(slots=True)
class HealthInfo:
    ok: bool
    mode: str
    segmentation_enabled: bool
    raw: dict[str, Any]


@dataclass(slots=True)
class FrameSummary:
    ts_ms: int
    status: str
    mode: str
    latency_ms: float | None
    fps_in: float | None
    fps_pose: float | None
    fps_total: float | None
    dropped_frames_count: int | None
    sign_segments: int
    phrase_segments: int
    committed: bool


class FrameSource:
    def __init__(
        self,
        *,
        video_path: Path | None,
        width: int,
        height: int,
        jpeg_quality: int,
        loop_video: bool,
    ) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("opencv-python is required for smoke script") from exc

        self.cv2 = cv2
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = int(max(1, min(100, jpeg_quality)))
        self.loop_video = bool(loop_video)
        self.cap = None

        if self.video_path is not None:
            self.cap = cv2.VideoCapture(str(self.video_path))
            if not self.cap.isOpened():
                raise FileNotFoundError(f"failed to open video: {self.video_path}")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _mock_frame(self, elapsed_s: float) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        text = f"SMOKE {elapsed_s:06.2f}s"
        self.cv2.putText(
            frame,
            text,
            (20, 40),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        return frame

    def next_jpeg(self, *, elapsed_s: float) -> bytes:
        frame = None
        if self.cap is not None:
            ok, frm = self.cap.read()
            if not ok:
                if self.loop_video:
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frm = self.cap.read()
                if not ok:
                    raise RuntimeError("video source ended and cannot loop")
            frame = frm
        else:
            frame = self._mock_frame(elapsed_s)

        ok, enc = self.cv2.imencode(
            ".jpg",
            frame,
            [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode JPEG")
        return bytes(enc.tobytes())


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="30s smoke test for pose_words runtime via WebSocket")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base backend URL")
    parser.add_argument("--ws-url", default="", help="Optional explicit WS url")
    parser.add_argument("--duration-sec", type=float, default=30.0, help="Run duration in seconds")
    parser.add_argument("--fps", type=float, default=12.0, help="Target frame send rate")
    parser.add_argument("--video", type=Path, default=None, help="Optional video file for input source")
    parser.add_argument("--loop-video", type=parse_bool, default=True, help="Loop video when source reaches end")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--health-timeout-sec", type=float, default=5.0)
    parser.add_argument("--require-pose-words", type=parse_bool, default=False)
    parser.add_argument("--require-segmentation", type=parse_bool, default=False)
    parser.add_argument("--log-path", type=Path, default=Path("backend/artifacts/smoke_pose_words.jsonl"))
    return parser


def infer_ws_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid base url: {base_url}")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws/stream"


def load_health(base_url: str, timeout_sec: float) -> HealthInfo:
    url = f"{base_url.rstrip('/')}/health"
    with urlopen(url, timeout=float(timeout_sec)) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise RuntimeError("invalid /health response")

    cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
    mode = str(cfg.get("recognition_mode", ""))
    segmentation_enabled = bool(cfg.get("segmentation_enabled", False))
    return HealthInfo(ok=bool(data.get("ok", False)), mode=mode, segmentation_enabled=segmentation_enabled, raw=data)


def parse_frame_summary(payload: dict[str, Any]) -> FrameSummary:
    perf = payload.get("perf") if isinstance(payload.get("perf"), dict) else {}
    segments = payload.get("segments") if isinstance(payload.get("segments"), dict) else {}

    sign_segments = segments.get("sign") if isinstance(segments.get("sign"), list) else []
    phrase_segments = segments.get("phrase") if isinstance(segments.get("phrase"), list) else []

    latency = perf.get("latency_ms")
    if latency is None:
        debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}
        latency = debug.get("latency_ms")

    return FrameSummary(
        ts_ms=int(payload.get("timestamp_ms", int(time.time() * 1000))),
        status=str(payload.get("status", "")),
        mode=str(payload.get("mode", "")),
        latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
        fps_in=float(perf.get("fps_in")) if isinstance(perf.get("fps_in"), (int, float)) else None,
        fps_pose=float(perf.get("fps_pose")) if isinstance(perf.get("fps_pose"), (int, float)) else None,
        fps_total=float(perf.get("fps_total")) if isinstance(perf.get("fps_total"), (int, float)) else None,
        dropped_frames_count=int(perf.get("dropped_frames_count"))
        if isinstance(perf.get("dropped_frames_count"), (int, float))
        else None,
        sign_segments=len(sign_segments),
        phrase_segments=len(phrase_segments),
        committed=bool(payload.get("text_state", {}).get("committed", False))
        if isinstance(payload.get("text_state"), dict)
        else False,
    )


async def run_smoke(args: argparse.Namespace) -> int:
    try:
        import websockets
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("websockets package is required for smoke script") from exc

    health = load_health(args.base_url, timeout_sec=float(args.health_timeout_sec))
    ws_url = args.ws_url or infer_ws_url(args.base_url)

    print("[smoke] health ok:", health.ok)
    print("[smoke] mode:", health.mode)
    print("[smoke] segmentation.enabled:", health.segmentation_enabled)
    print("[smoke] ws_url:", ws_url)

    if bool(args.require_pose_words) and health.mode != "pose_words":
        print("[smoke] fail: recognition_mode is not pose_words")
        return 2
    if bool(args.require_segmentation) and not health.segmentation_enabled:
        print("[smoke] fail: segmentation.enabled is false")
        return 2

    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = (Path.cwd() / log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    source = FrameSource(
        video_path=(args.video.resolve() if args.video else None),
        width=int(args.width),
        height=int(args.height),
        jpeg_quality=int(args.jpeg_quality),
        loop_video=bool(args.loop_video),
    )

    target_interval = 1.0 / max(0.1, float(args.fps))
    duration = max(1.0, float(args.duration_sec))

    latencies: list[float] = []
    fps_total_values: list[float] = []
    statuses = Counter()
    segment_events: set[tuple[int, int, int]] = set()
    sign_segments_total = 0
    phrase_segments_total = 0
    dropped_frames_max = 0
    received = 0
    sent = 0

    started = time.perf_counter()
    next_tick = started

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            async with websockets.connect(ws_url, max_size=2**24) as ws:
                while True:
                    now = time.perf_counter()
                    elapsed = now - started
                    if elapsed >= duration:
                        break
                    if now < next_tick:
                        await asyncio.sleep(next_tick - now)

                    frame_bytes = source.next_jpeg(elapsed_s=elapsed)
                    await ws.send(frame_bytes)
                    sent += 1

                    raw = await ws.recv()
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue

                    summary = parse_frame_summary(payload)
                    received += 1
                    statuses[summary.status] += 1

                    if summary.latency_ms is not None:
                        latencies.append(summary.latency_ms)
                    if summary.fps_total is not None:
                        fps_total_values.append(summary.fps_total)

                    sign_segments_total += int(summary.sign_segments)
                    phrase_segments_total += int(summary.phrase_segments)
                    if summary.dropped_frames_count is not None:
                        dropped_frames_max = max(dropped_frames_max, int(summary.dropped_frames_count))

                    seg_event = payload.get("segment_event") if isinstance(payload.get("segment_event"), dict) else None
                    if seg_event is not None:
                        start = int(seg_event.get("start", -1))
                        end = int(seg_event.get("end", -1))
                        ts_ms = int(seg_event.get("timestamp_ms", summary.ts_ms))
                        segment_events.add((start, end, ts_ms))

                    log_row = {
                        "ts_ms": summary.ts_ms,
                        "status": summary.status,
                        "mode": summary.mode,
                        "latency_ms": summary.latency_ms,
                        "fps_in": summary.fps_in,
                        "fps_pose": summary.fps_pose,
                        "fps_total": summary.fps_total,
                        "sign_segments": summary.sign_segments,
                        "phrase_segments": summary.phrase_segments,
                        "committed": summary.committed,
                    }
                    log_file.write(json.dumps(log_row, ensure_ascii=False) + "\n")

                    next_tick += target_interval
        finally:
            source.close()

    elapsed_total = max(1e-6, time.perf_counter() - started)
    fps_stream = received / elapsed_total

    latency_avg = float(statistics.fmean(latencies)) if latencies else None
    latency_p95 = float(np.percentile(np.asarray(latencies, dtype=np.float32), 95)) if latencies else None
    fps_total_avg = float(statistics.fmean(fps_total_values)) if fps_total_values else None

    print("[smoke] elapsed_sec:", round(elapsed_total, 2))
    print("[smoke] frames_sent:", sent)
    print("[smoke] frames_received:", received)
    print("[smoke] stream_fps:", round(fps_stream, 2))
    print("[smoke] latency_avg_ms:", None if latency_avg is None else round(latency_avg, 2))
    print("[smoke] latency_p95_ms:", None if latency_p95 is None else round(latency_p95, 2))
    print("[smoke] fps_total_avg:", None if fps_total_avg is None else round(fps_total_avg, 2))
    print("[smoke] status_counts:", dict(statuses))
    print("[smoke] unique_segment_events:", len(segment_events))
    print("[smoke] sign_segments_total:", sign_segments_total)
    print("[smoke] phrase_segments_total:", phrase_segments_total)
    print("[smoke] dropped_frames_max:", dropped_frames_max)
    print("[smoke] log_path:", log_path)

    if received == 0:
        print("[smoke] fail: no WS frames received")
        return 1
    print("[smoke] result: ok")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run_smoke(args))


if __name__ == "__main__":
    raise SystemExit(main())
