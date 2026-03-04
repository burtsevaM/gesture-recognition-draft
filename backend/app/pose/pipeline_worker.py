from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..perf import FrameProfiler, PerfAggregator
from .datatypes import PoseFrame, PoseLandmarksGroup
from .extractor import PoseExtractor
from .normalization import compose_features, hand_normalize_3d, shoulder_normalize


@dataclass(slots=True)
class PoseWorkerInput:
    frame_id: int
    now_ms: int
    frame_bgr: np.ndarray
    decode_jpeg_ms: float | None = None


@dataclass(slots=True)
class PoseWorkerResult:
    frame_id: int
    now_ms: int
    pose_frame: PoseFrame | None
    norm_frame: PoseFrame | None
    feature_vec: np.ndarray | None
    hand_present: bool
    error: str = ""
    timings_ms: dict[str, float] | None = None
    dropped_frames_count: int = 0


class PosePipelineWorker:
    """Background worker: BGR frame -> MediaPipe pose -> normalization -> features.

    Input queue uses latest-only policy for realtime: on overflow oldest frame is dropped.
    """

    def __init__(
        self,
        *,
        extractor: PoseExtractor,
        use_shoulder_norm: bool,
        use_hands_3d_norm: bool,
        queue_size: int = 3,
        output_size: int = 4,
        perf_enabled: bool = False,
        perf_aggregator: PerfAggregator | None = None,
        transform_fn: Callable[[PoseFrame], tuple[PoseFrame, np.ndarray]] | None = None,
    ) -> None:
        self.extractor = extractor
        self.use_shoulder_norm = bool(use_shoulder_norm)
        self.use_hands_3d_norm = bool(use_hands_3d_norm)
        self._in_q: queue.Queue[PoseWorkerInput] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._out_q: queue.Queue[PoseWorkerResult] = queue.Queue(maxsize=max(1, int(output_size)))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._next_frame_id = 1
        self._dropped_frames_count = 0
        self._processed_frames = 0
        self._perf_enabled = bool(perf_enabled)
        self._perf = perf_aggregator or PerfAggregator()
        self._transform_fn = transform_fn

    @staticmethod
    def _copy_group(group: PoseLandmarksGroup | None) -> PoseLandmarksGroup | None:
        if group is None:
            return None
        conf = None if group.confidence is None else group.confidence.copy()
        return PoseLandmarksGroup(points=group.points.copy(), confidence=conf)

    @classmethod
    def _copy_pose_frame(cls, frame: PoseFrame) -> PoseFrame:
        return PoseFrame(
            timestamp=float(frame.timestamp),
            body=cls._copy_group(frame.body),
            left_hand=cls._copy_group(frame.left_hand),
            right_hand=cls._copy_group(frame.right_hand),
            face=cls._copy_group(frame.face),
            meta=dict(frame.meta),
        )

    def _normalize_pose_frame(self, pose_frame: PoseFrame) -> PoseFrame:
        if self._transform_fn is not None:
            frame, _ = self._transform_fn(pose_frame)
            return frame

        if self.use_shoulder_norm:
            normalized, _ = shoulder_normalize([pose_frame], safe_mode=True)
            frame = normalized[0] if normalized else self._copy_pose_frame(pose_frame)
        else:
            frame = self._copy_pose_frame(pose_frame)

        if self.use_hands_3d_norm:
            if frame.left_hand is not None:
                frame.left_hand.points = hand_normalize_3d(frame.left_hand.points)
            if frame.right_hand is not None:
                frame.right_hand.points = hand_normalize_3d(frame.right_hand.points)
        return frame

    def _extract_features(self, norm_frame: PoseFrame) -> np.ndarray:
        if self._transform_fn is not None:
            _, feat = self._transform_fn(norm_frame)
            return np.asarray(feat, dtype=np.float32).reshape(-1)

        feature_vec, _ = compose_features(
            norm_frame,
            apply_shoulder_norm=False,
            hide_legs_before_body=True,
            canonical_hands_3d=False,
        )
        return np.asarray(feature_vec, dtype=np.float32).reshape(-1)

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="pose-pipeline-worker", daemon=True)
            self._thread.start()

    def stop(self, *, timeout_sec: float = 1.0) -> None:
        self._stop.set()
        # Unblock queue get.
        try:
            self._in_q.put_nowait(
                PoseWorkerInput(frame_id=-1, now_ms=int(time.monotonic() * 1000), frame_bgr=np.zeros((1, 1, 3), dtype=np.uint8))
            )
        except Exception:
            pass
        with self._lock:
            if self._thread is not None:
                self._thread.join(timeout=max(0.0, float(timeout_sec)))
            self._thread = None

    @property
    def dropped_frames_count(self) -> int:
        return int(self._dropped_frames_count)

    @property
    def processed_frames(self) -> int:
        return int(self._processed_frames)

    def perf_snapshot(self) -> dict[str, float | int]:
        return self._perf.snapshot_flat()

    def submit_frame(self, frame_bgr: np.ndarray, now_ms: int, *, decode_jpeg_ms: float | None = None) -> int:
        with self._lock:
            frame_id = self._next_frame_id
            self._next_frame_id += 1

        item = PoseWorkerInput(
            frame_id=int(frame_id),
            now_ms=int(now_ms),
            frame_bgr=np.asarray(frame_bgr, dtype=np.uint8),
            decode_jpeg_ms=float(decode_jpeg_ms) if decode_jpeg_ms is not None else None,
        )

        while True:
            try:
                self._in_q.put_nowait(item)
                break
            except queue.Full:
                try:
                    self._in_q.get_nowait()
                    self._dropped_frames_count += 1
                except queue.Empty:
                    continue
        return int(frame_id)

    def get_latest_result(self) -> PoseWorkerResult | None:
        latest: PoseWorkerResult | None = None
        while True:
            try:
                latest = self._out_q.get_nowait()
            except queue.Empty:
                break
        return latest

    def _emit_result(self, result: PoseWorkerResult) -> None:
        while True:
            try:
                self._out_q.put_nowait(result)
                break
            except queue.Full:
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    continue

    def _run(self) -> None:
        try:
            import cv2
        except Exception as exc:  # pragma: no cover - runtime dependency
            err = PoseWorkerResult(
                frame_id=-1,
                now_ms=int(time.monotonic() * 1000),
                pose_frame=None,
                norm_frame=None,
                feature_vec=None,
                hand_present=False,
                error=f"opencv import error: {exc}",
                timings_ms={},
                dropped_frames_count=self._dropped_frames_count,
            )
            self._emit_result(err)
            return

        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if item.frame_id < 0:
                continue

            frame_prof = FrameProfiler(self._perf, enabled=self._perf_enabled)
            frame_prof.record("decode_jpeg_ms", item.decode_jpeg_ms)
            pose_frame: PoseFrame | None = None
            norm_frame: PoseFrame | None = None
            feature_vec: np.ndarray | None = None
            hand_present = False
            error = ""

            try:
                with frame_prof.stage("mediapipe_ms"):
                    frame_rgb = cv2.cvtColor(item.frame_bgr, cv2.COLOR_BGR2RGB)
                    pose_frame = self.extractor.process(frame_rgb)

                if pose_frame is not None:
                    with frame_prof.stage("normalize_ms"):
                        norm_frame = self._normalize_pose_frame(pose_frame)
                    hand_present = bool(norm_frame.left_hand is not None or norm_frame.right_hand is not None)

                    with frame_prof.stage("feature_ms"):
                        feature_vec = self._extract_features(norm_frame)
            except Exception as exc:
                error = str(exc)

            timings = frame_prof.finish()
            self._processed_frames += 1
            self._emit_result(
                PoseWorkerResult(
                    frame_id=int(item.frame_id),
                    now_ms=int(item.now_ms),
                    pose_frame=pose_frame,
                    norm_frame=norm_frame,
                    feature_vec=feature_vec,
                    hand_present=bool(hand_present),
                    error=error,
                    timings_ms=timings,
                    dropped_frames_count=int(self._dropped_frames_count),
                )
            )


__all__ = ["PosePipelineWorker", "PoseWorkerInput", "PoseWorkerResult"]
