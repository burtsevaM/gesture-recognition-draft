from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayF32 = np.ndarray


def _as_float32_points(points: np.ndarray, *, expected_points: int | None = None) -> ArrayF32:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"landmarks must have shape [N, 3], got {arr.shape}")
    if expected_points is not None and arr.shape[0] != expected_points:
        raise ValueError(f"expected {expected_points} landmarks, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN/Inf values")
    return arr


def _as_confidence(confidence: np.ndarray | None, *, expected_points: int) -> ArrayF32 | None:
    if confidence is None:
        return None
    arr = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if arr.shape[0] != expected_points:
        raise ValueError(f"confidence must have shape [N], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("confidence contains NaN/Inf values")
    return arr


@dataclass(slots=True)
class PoseLandmarksGroup:
    points: ArrayF32
    confidence: ArrayF32 | None = None

    def __post_init__(self) -> None:
        self.points = _as_float32_points(self.points)
        self.confidence = _as_confidence(self.confidence, expected_points=self.points.shape[0])

    def validate(self, *, expected_points: int | None = None) -> None:
        self.points = _as_float32_points(self.points, expected_points=expected_points)
        self.confidence = _as_confidence(self.confidence, expected_points=self.points.shape[0])

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "points": self.points.astype(np.float32).tolist(),
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence.astype(np.float32).tolist()
        return payload


@dataclass(slots=True)
class PoseFrame:
    timestamp: float
    body: PoseLandmarksGroup | None = None
    left_hand: PoseLandmarksGroup | None = None
    right_hand: PoseLandmarksGroup | None = None
    face: PoseLandmarksGroup | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not np.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be a finite float")
        if self.body is not None:
            self.body.validate(expected_points=33)
        if self.left_hand is not None:
            self.left_hand.validate(expected_points=21)
        if self.right_hand is not None:
            self.right_hand.validate(expected_points=21)
        if self.face is not None:
            # MediaPipe Holistic face landmarks count is 468.
            self.face.validate(expected_points=468)

    def to_jsonable(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "timestamp": float(self.timestamp),
            "meta": dict(self.meta),
        }
        payload["body"] = self.body.to_jsonable() if self.body is not None else None
        payload["left_hand"] = self.left_hand.to_jsonable() if self.left_hand is not None else None
        payload["right_hand"] = self.right_hand.to_jsonable() if self.right_hand is not None else None
        payload["face"] = self.face.to_jsonable() if self.face is not None else None
        return payload


def validate_pose_frame(frame: PoseFrame) -> None:
    frame.validate()

