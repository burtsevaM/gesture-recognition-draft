from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .datatypes import PoseFrame, PoseLandmarksGroup


@dataclass(slots=True)
class PoseExtractorConfig:
    include_face: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class PoseExtractor:
    """MediaPipe Holistic wrapper for pose-first processing."""

    def __init__(
        self,
        *,
        include_face: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.config = PoseExtractorConfig(
            include_face=bool(include_face),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

        try:
            import mediapipe as mp
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("mediapipe is required for PoseExtractor") from exc

        self._mp = mp
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        if getattr(self, "_holistic", None) is not None:
            self._holistic.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    @staticmethod
    def _validate_frame(rgb_frame: np.ndarray) -> np.ndarray:
        if not isinstance(rgb_frame, np.ndarray):
            raise ValueError("rgb_frame must be numpy.ndarray")
        if rgb_frame.dtype != np.uint8:
            raise ValueError(f"rgb_frame must have dtype uint8, got {rgb_frame.dtype}")
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"rgb_frame must have shape [H, W, 3], got {rgb_frame.shape}")
        if rgb_frame.shape[0] < 1 or rgb_frame.shape[1] < 1:
            raise ValueError("rgb_frame must have non-zero width and height")
        return np.ascontiguousarray(rgb_frame)

    @staticmethod
    def _landmarks_to_group(landmarks: Any, *, with_visibility: bool = False) -> PoseLandmarksGroup | None:
        if landmarks is None:
            return None
        lm_list = getattr(landmarks, "landmark", None)
        if not lm_list:
            return None

        points = np.asarray([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
        confidence: np.ndarray | None = None

        if with_visibility:
            values: list[float] = []
            all_present = True
            for lm in lm_list:
                if hasattr(lm, "visibility"):
                    values.append(float(lm.visibility))
                else:
                    all_present = False
                    break
            if all_present:
                confidence = np.asarray(values, dtype=np.float32)

        return PoseLandmarksGroup(points=points, confidence=confidence)

    def process(self, rgb_frame: np.ndarray) -> PoseFrame | None:
        frame = self._validate_frame(rgb_frame)
        h, w = frame.shape[:2]

        results = self._holistic.process(frame)

        body = self._landmarks_to_group(results.pose_landmarks, with_visibility=True)
        left_hand = self._landmarks_to_group(results.left_hand_landmarks)
        right_hand = self._landmarks_to_group(results.right_hand_landmarks)
        face = None
        if self.config.include_face:
            face = self._landmarks_to_group(results.face_landmarks)

        if body is None and left_hand is None and right_hand is None and face is None:
            return None

        pose_frame = PoseFrame(
            timestamp=float(time.time()),
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face,
            meta={
                "image_size": [int(h), int(w)],
                "include_face": bool(self.config.include_face),
                "source": "mediapipe_holistic",
            },
        )
        pose_frame.validate()
        return pose_frame

