from __future__ import annotations

import numpy as np

from app.pose.datatypes import PoseFrame, PoseLandmarksGroup
from app.pose.normalization import POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER, shoulder_normalize


def _make_body_sequence(
    *,
    frames_count: int,
    shoulder_distance: float,
    translation: np.ndarray,
) -> list[PoseFrame]:
    seq: list[PoseFrame] = []
    l = np.array([-shoulder_distance * 0.5, 0.0, 0.0], dtype=np.float32)
    r = np.array([shoulder_distance * 0.5, 0.0, 0.0], dtype=np.float32)

    for t in range(frames_count):
        body = np.zeros((33, 3), dtype=np.float32)
        drift = np.array([0.03 * t, -0.02 * t, 0.01 * t], dtype=np.float32)
        body[POSE_LEFT_SHOULDER] = l + translation + drift
        body[POSE_RIGHT_SHOULDER] = r + translation + drift

        visibility = np.ones((33,), dtype=np.float32)
        seq.append(
            PoseFrame(
                timestamp=float(t),
                body=PoseLandmarksGroup(points=body, confidence=visibility),
            )
        )
    return seq


def test_shoulder_normalize_centers_and_scales_sequence() -> None:
    seq = _make_body_sequence(
        frames_count=12,
        shoulder_distance=2.8,
        translation=np.array([4.2, -3.5, 1.7], dtype=np.float32),
    )

    normalized, info = shoulder_normalize(seq, safe_mode=True)

    assert info.normalized is True
    assert info.valid_frames == 12

    mids = []
    dists = []
    for frame in normalized:
        assert frame.body is not None
        l = frame.body.points[POSE_LEFT_SHOULDER]
        r = frame.body.points[POSE_RIGHT_SHOULDER]
        mids.append((l + r) * 0.5)
        dists.append(np.linalg.norm(l - r))

    mids_arr = np.asarray(mids, dtype=np.float32)
    dists_arr = np.asarray(dists, dtype=np.float32)

    assert np.allclose(np.mean(mids_arr, axis=0), np.zeros(3, dtype=np.float32), atol=1e-4)
    assert np.isclose(float(np.mean(dists_arr)), 1.0, atol=1e-4)

