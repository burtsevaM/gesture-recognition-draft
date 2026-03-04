from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.main import SessionProcessor
from app.pose.datatypes import PoseFrame, PoseLandmarksGroup


class _StubExtractor:
    def process(self, rgb_frame: np.ndarray) -> PoseFrame:
        body = np.zeros((33, 3), dtype=np.float32)
        body[11] = [0.4, 0.5, 0.0]
        body[12] = [0.6, 0.5, 0.0]
        return PoseFrame(
            timestamp=1.0,
            body=PoseLandmarksGroup(points=body, confidence=np.ones((33,), dtype=np.float32)),
            left_hand=None,
            right_hand=None,
            meta={},
        )


class _StubRuntime:
    def __init__(self, *, perf_enabled: bool) -> None:
        self.errors: dict[str, str] = {}
        self._extractor = _StubExtractor()
        self.config = SimpleNamespace(
            recognition_mode="pose_words",
            hold_ms=700,
            cooldown_ms=500,
            precommit_ratio=0.8,
            uncertain_streak_frames=4,
            switch_min_frames=3,
            use_shoulder_norm=True,
            use_hands_3d_norm=False,
            perf_enabled=bool(perf_enabled),
            perf_window_size=64,
            perf_ema_alpha=0.2,
            pose_worker_enabled=False,
            pose_worker_queue_size=2,
            pose_worker_output_size=2,
            segmentation_enabled=False,
            segmentation_window=256,
            segmentation_step=8,
            segmentation_min_len=6,
            segmentation_max_len=150,
            segmentation_merge_gap=2,
            segmentation_cool_off_frames=4,
            segmentation_sign_th_b=0.5,
            segmentation_sign_th_o=0.5,
            word_hold_frames=6,
        )

    def get_pose_extractor(self) -> _StubExtractor:
        return self._extractor



def test_perf_payload_is_optional_disabled() -> None:
    runtime = _StubRuntime(perf_enabled=False)
    session = SessionProcessor(runtime)  # type: ignore[arg-type]
    payload = session.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), now_ms=1000, decode_jpeg_ms=2.5)
    assert payload["mode"] == "pose_words"
    assert "perf" not in payload



def test_perf_payload_present_when_enabled() -> None:
    runtime = _StubRuntime(perf_enabled=True)
    session = SessionProcessor(runtime)  # type: ignore[arg-type]
    payload = session.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), now_ms=1000, decode_jpeg_ms=3.5)
    assert payload["mode"] == "pose_words"
    assert "perf" in payload
    perf = payload["perf"]
    assert "latency_ms" in perf
    assert "decode_jpeg_ms_ema" in perf
    assert "total_ms_ema" in perf
