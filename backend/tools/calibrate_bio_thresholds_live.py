#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.config import load_config
from app.pose import PoseExtractor, compose_features
from app.segmentation import (
    BioSegmenterOnnxModel,
    average_segment_length_frames,
    average_segment_length_seconds,
    boundary_jitter,
    decode_segments,
    estimate_fp_per_minute,
    segments_per_minute,
    stability_score,
)


@dataclass(slots=True)
class GridConfig:
    th_b: float
    th_o: float
    min_len: int
    merge_gap: int


@dataclass(slots=True)
class EvalResult:
    cfg: GridConfig
    segment_count: int
    segments_per_min: float
    fp_per_min: float
    avg_len_frames: float
    avg_len_seconds: float
    jitter: float
    stability: float
    score: float


def parse_float_grid(raw: str) -> list[float]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("empty float grid")
    return values


def parse_int_grid(raw: str) -> list[int]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("empty int grid")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate BIO thresholds on live camera/video")
    parser.add_argument("--config", default=str(BACKEND / "config.yaml"))
    parser.add_argument("--source", choices=["camera", "video", "ws"], default="camera")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--ws-url", default="", help="WebSocket frame source (binary JPEG or json{frame_jpeg_b64})")
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=0.0, help="Override fps; 0 -> auto")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=0, help="0 -> from config")
    parser.add_argument("--cool-off-frames", type=int, default=0, help="0 -> from config")
    parser.add_argument("--use-shoulder-norm", action="store_true")
    parser.add_argument("--use-hands-3d-norm", action="store_true")
    parser.add_argument("--grid-th-b", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--grid-th-o", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--grid-min-len", default="4,6,8")
    parser.add_argument("--grid-merge-gap", default="0,1,2")
    parser.add_argument("--quiet-mode", action="store_true", help="Treat all detected segments as false positives")
    parser.add_argument("--motion-threshold", type=float, default=0.02)
    parser.add_argument("--target-segments-per-min", type=float, default=8.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thresholds-out", default="")
    return parser.parse_args()


def _iter_capture_frames(cap, *, max_frames: int, stride: int) -> Iterable[np.ndarray]:
    taken = 0
    frame_i = 0
    while taken < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_i % max(1, stride) != 0:
            frame_i += 1
            continue
        frame_i += 1
        taken += 1
        yield frame


def _decode_ws_frame(message: bytes | str):
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required") from exc

    if isinstance(message, bytes):
        arr = np.frombuffer(message, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    try:
        payload = json.loads(message)
    except Exception:
        return None
    b64 = payload.get("frame_jpeg_b64") or payload.get("jpeg_b64") or payload.get("frame_b64")
    if not isinstance(b64, str) or not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def _collect_ws_frames(ws_url: str, *, max_frames: int) -> list[np.ndarray]:
    try:
        import websockets
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("websockets package is required for --source ws") from exc

    frames: list[np.ndarray] = []
    async with websockets.connect(ws_url, max_size=2**24) as ws:
        while len(frames) < max_frames:
            message = await ws.recv()
            frame = _decode_ws_frame(message)
            if frame is None:
                continue
            frames.append(frame)
    return frames


def collect_features(args: argparse.Namespace, cfg) -> tuple[np.ndarray, float]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required") from exc

    model_path = Path(cfg.segmentation_model_path)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    seg_model = BioSegmenterOnnxModel(model_path=model_path, ort_num_threads=cfg.segmentation_ort_num_threads)
    feature_dim = seg_model.input_feature_dim
    if feature_dim is None:
        raise ValueError("BIO ONNX input feature dim is dynamic/unknown; calibration requires fixed feature dim")

    frames: list[np.ndarray] = []
    fps = float(args.fps)
    if args.source == "video":
        if not args.video_path:
            raise ValueError("--video-path is required for --source video")
        cap = cv2.VideoCapture(str(Path(args.video_path).expanduser().resolve()))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {args.video_path}")
        try:
            if fps <= 0:
                src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if src_fps > 1e-6:
                    fps = src_fps
            frames = list(_iter_capture_frames(cap, max_frames=int(args.max_frames), stride=int(args.frame_stride)))
        finally:
            cap.release()
    elif args.source == "camera":
        cap = cv2.VideoCapture(int(args.camera_index))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera index={args.camera_index}")
        try:
            if fps <= 0:
                src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if src_fps > 1e-6:
                    fps = src_fps
            frames = list(_iter_capture_frames(cap, max_frames=int(args.max_frames), stride=int(args.frame_stride)))
        finally:
            cap.release()
    else:
        if not args.ws_url:
            raise ValueError("--ws-url is required for --source ws")
        frames = asyncio.run(_collect_ws_frames(args.ws_url, max_frames=int(args.max_frames)))

    if not frames:
        raise RuntimeError("no frames collected")
    if fps <= 0:
        fps = 25.0

    use_shoulder = bool(args.use_shoulder_norm) or bool(getattr(cfg, "use_shoulder_norm", True))
    use_hands = bool(args.use_hands_3d_norm) or bool(getattr(cfg, "use_hands_3d_norm", False))
    extractor = PoseExtractor(include_face=False)

    features: list[np.ndarray] = []
    for idx, frame_bgr in enumerate(frames, start=1):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose = extractor.process(frame_rgb)
        if pose is None:
            feat = np.zeros((feature_dim,), dtype=np.float32)
        else:
            feat, _ = compose_features(
                pose,
                apply_shoulder_norm=use_shoulder,
                hide_legs_before_body=True,
                canonical_hands_3d=use_hands,
            )
            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            if feat.shape[0] != feature_dim:
                if feat.shape[0] < feature_dim:
                    padded = np.zeros((feature_dim,), dtype=np.float32)
                    padded[: feat.shape[0]] = feat
                    feat = padded
                else:
                    feat = feat[:feature_dim]
        if not np.all(np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        features.append(feat.astype(np.float32, copy=False))
        if idx % 100 == 0:
            print(f"[collect] processed {idx}/{len(frames)} frames")

    extractor.close()
    matrix = np.stack(features, axis=0).astype(np.float32)
    print(f"[collect] done: frames={matrix.shape[0]}, feature_dim={matrix.shape[1]}, fps={fps:.3f}")
    return matrix, float(fps)


def run_bio_over_sequence(
    *,
    features: np.ndarray,
    model: BioSegmenterOnnxModel,
    window: int,
    step: int,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    n_frames = int(features.shape[0])
    sign_sum = np.zeros((n_frames, 3), dtype=np.float32)
    counts = np.zeros((n_frames,), dtype=np.int32)
    snapshots: list[np.ndarray] = []
    latencies_ms: list[float] = []

    for end in range(n_frames):
        if end + 1 < window:
            continue
        if ((end + 1 - window) % step) != 0:
            continue
        clip = features[end - window + 1 : end + 1]
        sign_probs, _, latency_ms = model.infer(clip)
        latencies_ms.append(float(latency_ms))
        idx = np.arange(end - window + 1, end + 1)
        sign_sum[idx] += sign_probs
        counts[idx] += 1

        prefix_len = end + 1
        prefix = np.zeros((prefix_len, 3), dtype=np.float32)
        prefix[:, 2] = 1.0
        valid = counts[:prefix_len] > 0
        if np.any(valid):
            prefix[valid] = sign_sum[:prefix_len][valid] / counts[:prefix_len][valid][:, None]
        snapshots.append(prefix)

    averaged = np.zeros((n_frames, 3), dtype=np.float32)
    averaged[:, 2] = 1.0
    valid = counts > 0
    if np.any(valid):
        averaged[valid] = sign_sum[valid] / counts[valid][:, None]

    mean_latency = float(np.mean(np.asarray(latencies_ms, dtype=np.float32))) if latencies_ms else 0.0
    return averaged, snapshots, mean_latency


def evaluate_grid(
    *,
    probs: np.ndarray,
    snapshots: list[np.ndarray],
    motion: np.ndarray,
    fps: float,
    total_frames: int,
    grid_cfgs: list[GridConfig],
    max_len: int,
    quiet_mode: bool,
    motion_threshold: float,
    target_segments_per_min: float,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for cfg in grid_cfgs:
        segments = decode_segments(
            probs,
            th_B=cfg.th_b,
            th_O=cfg.th_o,
            min_len=cfg.min_len,
            max_len=max_len,
            merge_gap=cfg.merge_gap,
        )
        snapshots_segments = [
            decode_segments(
                snap,
                th_B=cfg.th_b,
                th_O=cfg.th_o,
                min_len=cfg.min_len,
                max_len=max_len,
                merge_gap=cfg.merge_gap,
            )
            for snap in snapshots
        ]

        spm = segments_per_minute(segment_count=len(segments), total_frames=total_frames, fps=fps)
        fp_min = estimate_fp_per_minute(
            segments,
            total_frames=total_frames,
            fps=fps,
            quiet_mode=quiet_mode,
            motion_per_frame=motion,
            motion_threshold=motion_threshold,
        )
        avg_frames = average_segment_length_frames(segments)
        avg_seconds = average_segment_length_seconds(segments, fps=fps)
        jitter = boundary_jitter(snapshots_segments)
        stability = stability_score(snapshots_segments)
        score = (
            (fp_min * 2.0)
            + (jitter * 0.3)
            + (abs(spm - float(target_segments_per_min)) * 0.2)
            - (stability * 0.2)
        )

        results.append(
            EvalResult(
                cfg=cfg,
                segment_count=len(segments),
                segments_per_min=spm,
                fp_per_min=fp_min,
                avg_len_frames=avg_frames,
                avg_len_seconds=avg_seconds,
                jitter=jitter,
                stability=stability,
                score=float(score),
            )
        )

    return sorted(results, key=lambda x: (x.score, x.fp_per_min, x.jitter))


def print_top_results(results: list[EvalResult], top_k: int) -> None:
    print("")
    print("Top configurations:")
    print(
        " rank | th_B | th_O | min_len | merge_gap | seg/min | fp/min | avg_len(fr) | avg_len(s) | jitter | stability | score"
    )
    print("-" * 126)
    for idx, item in enumerate(results[: max(1, int(top_k))], start=1):
        print(
            f" {idx:>4} | {item.cfg.th_b:>4.2f} | {item.cfg.th_o:>4.2f} |"
            f" {item.cfg.min_len:>7d} | {item.cfg.merge_gap:>9d} |"
            f" {item.segments_per_min:>7.2f} | {item.fp_per_min:>6.2f} |"
            f" {item.avg_len_frames:>11.2f} | {item.avg_len_seconds:>10.3f} |"
            f" {item.jitter:>6.2f} | {item.stability:>9.3f} | {item.score:>5.3f}"
        )


def backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak.{stamp}")
    shutil.copy2(path, backup)
    return backup


def save_best_thresholds(
    *,
    out_path: Path,
    best: EvalResult,
    max_len: int,
    cool_off_frames: int,
    source: str,
    total_frames: int,
    fps: float,
) -> Path | None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_file(out_path)

    payload: dict = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    if not isinstance(payload, dict):
        payload = {}

    payload.setdefault("bio_mapping", {"B": 0, "I": 1, "O": 2})
    payload["sign"] = {
        "th_b": float(best.cfg.th_b),
        "th_o": float(best.cfg.th_o),
        "min_len": int(best.cfg.min_len),
        "merge_gap": int(best.cfg.merge_gap),
        "max_len": int(max_len),
        "cool_off_frames": int(cool_off_frames),
    }
    payload.setdefault("phrase", {"th_b": 0.5, "th_o": 0.5})
    payload["calibration"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "total_frames": int(total_frames),
        "fps": float(fps),
        "segment_count": int(best.segment_count),
        "segments_per_min": float(best.segments_per_min),
        "fp_per_min": float(best.fp_per_min),
        "avg_len_frames": float(best.avg_len_frames),
        "avg_len_seconds": float(best.avg_len_seconds),
        "jitter": float(best.jitter),
        "stability": float(best.stability),
        "score": float(best.score),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return backup


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    model_path = Path(cfg.segmentation_model_path)
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    model = BioSegmenterOnnxModel(model_path=model_path, ort_num_threads=cfg.segmentation_ort_num_threads)

    features, fps = collect_features(args, cfg)
    motion = np.zeros((features.shape[0],), dtype=np.float32)
    if features.shape[0] > 1:
        motion[1:] = np.linalg.norm(features[1:] - features[:-1], axis=1).astype(np.float32)

    window = int(args.window) if int(args.window) > 0 else int(cfg.segmentation_window)
    step = int(args.step) if int(args.step) > 0 else int(cfg.segmentation_step)
    max_len = int(args.max_len) if int(args.max_len) > 0 else int(cfg.segmentation_max_len)
    cool_off_frames = int(args.cool_off_frames) if int(args.cool_off_frames) > 0 else int(cfg.segmentation_cool_off_frames)
    print(f"[calibrate] inference params: window={window}, step={step}, max_len={max_len}, cool_off_frames={cool_off_frames}")

    started = time.perf_counter()
    probs, snapshots, mean_latency = run_bio_over_sequence(
        features=features,
        model=model,
        window=window,
        step=step,
    )
    print(f"[calibrate] BIO pass done in {(time.perf_counter() - started):.2f}s, mean_onnx_latency={mean_latency:.2f}ms")

    grid = [
        GridConfig(th_b=th_b, th_o=th_o, min_len=min_len, merge_gap=merge_gap)
        for th_b in parse_float_grid(args.grid_th_b)
        for th_o in parse_float_grid(args.grid_th_o)
        for min_len in parse_int_grid(args.grid_min_len)
        for merge_gap in parse_int_grid(args.grid_merge_gap)
    ]
    print(f"[calibrate] evaluating {len(grid)} configurations...")
    results = evaluate_grid(
        probs=probs,
        snapshots=snapshots,
        motion=motion,
        fps=float(fps),
        total_frames=int(features.shape[0]),
        grid_cfgs=grid,
        max_len=max_len,
        quiet_mode=bool(args.quiet_mode),
        motion_threshold=float(args.motion_threshold),
        target_segments_per_min=float(args.target_segments_per_min),
    )
    if not results:
        print("[calibrate] no results")
        return 1

    print_top_results(results, top_k=int(args.top_k))
    best = results[0]
    print("")
    print("[best]")
    print(
        f" th_B={best.cfg.th_b:.3f}, th_O={best.cfg.th_o:.3f}, min_len={best.cfg.min_len}, merge_gap={best.cfg.merge_gap},"
        f" seg/min={best.segments_per_min:.2f}, fp/min={best.fp_per_min:.2f}, stability={best.stability:.3f}"
    )

    out_path = Path(args.thresholds_out) if args.thresholds_out else Path(cfg.segmentation_thresholds_path)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    backup = save_best_thresholds(
        out_path=out_path,
        best=best,
        max_len=max_len,
        cool_off_frames=cool_off_frames,
        source=str(args.source),
        total_frames=int(features.shape[0]),
        fps=float(fps),
    )
    if backup is not None:
        print(f"[calibrate] backup saved: {backup}")
    print(f"[calibrate] updated thresholds: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
