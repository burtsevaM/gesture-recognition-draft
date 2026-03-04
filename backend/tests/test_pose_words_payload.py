from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.main import SessionProcessor
from app.pose.datatypes import PoseFrame, PoseLandmarksGroup


class _StubExtractor:
    def __init__(self, frame: PoseFrame | None) -> None:
        self._frame = frame

    def process(self, rgb_frame: np.ndarray) -> PoseFrame | None:
        return self._frame


class _StubRuntime:
    def __init__(self, pose_frame: PoseFrame | None) -> None:
        self.config = SimpleNamespace(
            recognition_mode="pose_words",
            hold_ms=700,
            cooldown_ms=500,
            precommit_ratio=0.8,
            uncertain_streak_frames=4,
            switch_min_frames=3,
            use_shoulder_norm=True,
            use_hands_3d_norm=True,
        )
        self.errors: dict[str, str] = {}
        self._extractor = _StubExtractor(pose_frame)

    def get_pose_extractor(self) -> _StubExtractor:
        return self._extractor


def _make_pose_frame() -> PoseFrame:
    body = np.zeros((33, 3), dtype=np.float32)
    body[11] = [0.4, 0.5, 0.0]
    body[12] = [0.6, 0.5, 0.0]

    left = np.zeros((21, 3), dtype=np.float32)
    left[0] = [0.42, 0.64, 0.0]
    left[5] = [0.47, 0.60, 0.0]
    left[9] = [0.44, 0.57, 0.0]
    left[17] = [0.39, 0.60, 0.0]

    right = np.zeros((21, 3), dtype=np.float32)
    right[0] = [0.58, 0.64, 0.0]
    right[5] = [0.63, 0.60, 0.0]
    right[9] = [0.60, 0.57, 0.0]
    right[17] = [0.55, 0.60, 0.0]

    return PoseFrame(
        timestamp=1.0,
        body=PoseLandmarksGroup(points=body, confidence=np.ones((33,), dtype=np.float32)),
        left_hand=PoseLandmarksGroup(points=left, confidence=np.ones((21,), dtype=np.float32)),
        right_hand=PoseLandmarksGroup(points=right, confidence=np.ones((21,), dtype=np.float32)),
        meta={},
    )


def test_pose_words_payload_contains_skeleton() -> None:
    runtime = _StubRuntime(_make_pose_frame())
    session = SessionProcessor(runtime)  # type: ignore[arg-type]

    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    payload = session.process_frame(frame_bgr, now_ms=1234)

    assert payload["mode"] == "pose_words"
    assert payload["status"] == "POSE"
    assert "skeleton" in payload
    assert payload["skeleton"]["raw"]["body"] is not None
    assert payload["skeleton"]["raw"]["lh"] is not None
    assert payload["skeleton"]["raw"]["rh"] is not None
    assert payload["skeleton"]["norm"]["body"] is not None
    assert payload["skeleton"]["norm"]["lh"] is not None
    assert payload["skeleton"]["norm"]["rh"] is not None

