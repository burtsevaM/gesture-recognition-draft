from __future__ import annotations

import numpy as np
import pytest

from app.pose.datatypes import PoseFrame, PoseLandmarksGroup
from app.pose.extractor import PoseExtractor


def test_validate_catches_bad_landmark_shape() -> None:
    frame = PoseFrame(
        timestamp=0.0,
        body=PoseLandmarksGroup(points=np.zeros((33, 3), dtype=np.float32)),
    )
    # Имитируем поломанные данные после инициализации.
    frame.body.points = np.zeros((33, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \[N, 3\]"):
        frame.validate()


def test_validate_catches_bad_confidence_shape() -> None:
    frame = PoseFrame(
        timestamp=0.0,
        left_hand=PoseLandmarksGroup(points=np.zeros((21, 3), dtype=np.float32)),
    )
    frame.left_hand.confidence = np.ones((20,), dtype=np.float32)
    with pytest.raises(ValueError, match="confidence"):
        frame.validate()


def test_pose_extractor_rejects_invalid_input() -> None:
    pytest.importorskip("mediapipe")

    with PoseExtractor(include_face=False) as extractor:
        with pytest.raises(ValueError, match="numpy.ndarray"):
            extractor.process("not-array")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="dtype uint8"):
            extractor.process(np.zeros((32, 32, 3), dtype=np.float32))

        with pytest.raises(ValueError, match="shape"):
            extractor.process(np.zeros((32, 32), dtype=np.uint8))

