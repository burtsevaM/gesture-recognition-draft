from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .datatypes import PoseFrame, PoseLandmarksGroup

# MediaPipe Pose landmark indices
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24
POSE_LEFT_KNEE = 25
POSE_RIGHT_KNEE = 26
POSE_LEFT_ANKLE = 27
POSE_RIGHT_ANKLE = 28
POSE_LEFT_HEEL = 29
POSE_RIGHT_HEEL = 30
POSE_LEFT_FOOT_INDEX = 31
POSE_RIGHT_FOOT_INDEX = 32

HAND_WRIST = 0
HAND_INDEX_MCP = 5
HAND_MIDDLE_MCP = 9
HAND_PINKY_MCP = 17

POSE_LEG_INDICES: tuple[int, ...] = (
    POSE_LEFT_HIP,
    POSE_RIGHT_HIP,
    POSE_LEFT_KNEE,
    POSE_RIGHT_KNEE,
    POSE_LEFT_ANKLE,
    POSE_RIGHT_ANKLE,
    POSE_LEFT_HEEL,
    POSE_RIGHT_HEEL,
    POSE_LEFT_FOOT_INDEX,
    POSE_RIGHT_FOOT_INDEX,
)

DEFAULT_UPPER_BODY_INDICES: tuple[int, ...] = (
    0,
    9,
    10,
    POSE_LEFT_SHOULDER,
    POSE_RIGHT_SHOULDER,
    13,
    14,
    15,
    16,
    POSE_LEFT_HIP,
    POSE_RIGHT_HIP,
)


@dataclass(slots=True)
class ShoulderNormInfo:
    normalized: bool
    center: np.ndarray
    scale: float
    valid_frames: int
    reason: str = ""


def _copy_group(group: PoseLandmarksGroup | None) -> PoseLandmarksGroup | None:
    if group is None:
        return None
    conf = None if group.confidence is None else group.confidence.copy()
    return PoseLandmarksGroup(points=group.points.copy(), confidence=conf)


def _copy_frame(frame: PoseFrame) -> PoseFrame:
    return PoseFrame(
        timestamp=float(frame.timestamp),
        body=_copy_group(frame.body),
        left_hand=_copy_group(frame.left_hand),
        right_hand=_copy_group(frame.right_hand),
        face=_copy_group(frame.face),
        meta=dict(frame.meta),
    )


def _iter_shoulder_stats(
    frames: Sequence[PoseFrame],
    *,
    min_shoulder_confidence: float,
) -> Iterable[tuple[np.ndarray, float]]:
    for frame in frames:
        body = frame.body
        if body is None:
            continue
        if body.points.shape[0] <= max(POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER):
            continue
        l = body.points[POSE_LEFT_SHOULDER]
        r = body.points[POSE_RIGHT_SHOULDER]
        if not (np.all(np.isfinite(l)) and np.all(np.isfinite(r))):
            continue
        if body.confidence is not None:
            lc = float(body.confidence[POSE_LEFT_SHOULDER])
            rc = float(body.confidence[POSE_RIGHT_SHOULDER])
            if lc < min_shoulder_confidence or rc < min_shoulder_confidence:
                continue
        midpoint = (l + r) * 0.5
        dist = float(np.linalg.norm(l - r))
        if dist <= 0.0:
            continue
        yield midpoint.astype(np.float32), dist


def shoulder_normalize(
    sequence: Sequence[PoseFrame] | np.ndarray,
    *,
    window: int | None = None,
    safe_mode: bool = True,
    min_shoulder_confidence: float = 0.0,
    eps: float = 1e-6,
) -> tuple[Sequence[PoseFrame] | np.ndarray, ShoulderNormInfo]:
    """Normalize points by shoulder midpoint and shoulder distance.

    Safe behavior by default: if reliable shoulders are unavailable, returns input copy and
    marks `normalized=False` in debug info.
    """

    if isinstance(sequence, np.ndarray):
        points = np.asarray(sequence, dtype=np.float32)
        squeeze_back = False
        if points.ndim == 2 and points.shape[1] == 3:
            points = points[None, ...]
            squeeze_back = True
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError("numpy input must have shape [T, N, 3] or [N, 3]")

        output = points.copy()
        if output.shape[1] <= max(POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER):
            info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "body_has_no_shoulders")
            if safe_mode:
                return (output[0] if squeeze_back else output), info
            raise ValueError("cannot normalize: body landmarks do not include shoulder indices")

        stats_points = output
        if window is not None:
            w = max(1, int(window))
            stats_points = stats_points[-w:]

        l = stats_points[:, POSE_LEFT_SHOULDER, :]
        r = stats_points[:, POSE_RIGHT_SHOULDER, :]
        valid = np.isfinite(l).all(axis=1) & np.isfinite(r).all(axis=1)
        if not np.any(valid):
            info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "no_valid_shoulders")
            if safe_mode:
                return (output[0] if squeeze_back else output), info
            raise ValueError("cannot normalize: no valid shoulders")

        mid = (l[valid] + r[valid]) * 0.5
        dists = np.linalg.norm(l[valid] - r[valid], axis=1)
        mean_dist = float(np.mean(dists)) if dists.size else 0.0
        if mean_dist <= eps:
            info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, int(valid.sum()), "shoulder_distance_too_small")
            if safe_mode:
                return (output[0] if squeeze_back else output), info
            raise ValueError("cannot normalize: shoulder distance too small")

        center = np.mean(mid, axis=0).astype(np.float32)
        scale = float(1.0 / mean_dist)
        output = (output - center.reshape(1, 1, 3)) * scale
        info = ShoulderNormInfo(True, center, scale, int(valid.sum()), "")
        return (output[0] if squeeze_back else output), info

    frames = [_copy_frame(frame) for frame in sequence]
    if not frames:
        info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "empty_sequence")
        return frames, info

    frames_for_stats = frames
    if window is not None:
        w = max(1, int(window))
        frames_for_stats = frames[-w:]

    stats = list(_iter_shoulder_stats(frames_for_stats, min_shoulder_confidence=min_shoulder_confidence))
    if not stats:
        info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "no_valid_shoulders")
        if safe_mode:
            return frames, info
        raise ValueError("cannot normalize: no reliable shoulders in sequence")

    mids = np.stack([x[0] for x in stats], axis=0)
    dist = np.asarray([x[1] for x in stats], dtype=np.float32)
    mean_dist = float(np.mean(dist))
    if mean_dist <= eps:
        info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, len(stats), "shoulder_distance_too_small")
        if safe_mode:
            return frames, info
        raise ValueError("cannot normalize: shoulder distance too small")

    center = np.mean(mids, axis=0).astype(np.float32)
    scale = float(1.0 / mean_dist)

    for frame in frames:
        for group_name in ("body", "left_hand", "right_hand", "face"):
            group = getattr(frame, group_name)
            if group is None:
                continue
            group.points = ((group.points - center.reshape(1, 3)) * scale).astype(np.float32)

    info = ShoulderNormInfo(True, center, scale, len(stats), "")
    return frames, info


def hide_legs(sequence: Sequence[PoseFrame] | np.ndarray) -> Sequence[PoseFrame] | np.ndarray:
    if isinstance(sequence, np.ndarray):
        points = np.asarray(sequence, dtype=np.float32)
        squeeze_back = False
        if points.ndim == 2 and points.shape[1] == 3:
            points = points[None, ...]
            squeeze_back = True
        if points.ndim != 3 or points.shape[2] != 3:
            raise ValueError("numpy input must have shape [T, N, 3] or [N, 3]")

        out = points.copy()
        valid_indices = [idx for idx in POSE_LEG_INDICES if idx < out.shape[1]]
        if valid_indices:
            out[:, valid_indices, :] = 0.0
        return out[0] if squeeze_back else out

    frames = [_copy_frame(frame) for frame in sequence]
    for frame in frames:
        body = frame.body
        if body is None:
            continue
        for idx in POSE_LEG_INDICES:
            if idx >= body.points.shape[0]:
                continue
            body.points[idx, :] = 0.0
            if body.confidence is not None:
                body.confidence[idx] = 0.0
    return frames


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= eps or nb <= eps:
        return np.eye(3, dtype=np.float32)
    a = a / na
    b = b / nb

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s <= eps:
        if c > 0.0:
            return np.eye(3, dtype=np.float32)
        # 180-degree rotation around a stable orthogonal axis.
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(axis, a))) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis = axis - float(np.dot(axis, a)) * a
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= eps:
            return np.eye(3, dtype=np.float32)
        axis = axis / axis_norm
        k = _skew(axis)
        return (np.eye(3, dtype=np.float32) + 2.0 * (k @ k)).astype(np.float32)

    k = _skew(v)
    r = np.eye(3, dtype=np.float32) + k + (k @ k) * ((1.0 - c) / (s * s))
    return r.astype(np.float32)


def _rotation_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def hand_normalize_3d(hand_21x3: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    hand = np.asarray(hand_21x3, dtype=np.float32)
    if hand.shape != (21, 3):
        raise ValueError(f"hand landmarks must have shape [21, 3], got {hand.shape}")
    if not np.all(np.isfinite(hand)):
        hand = np.nan_to_num(hand, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    wrist = hand[HAND_WRIST].copy()
    centered = hand - wrist.reshape(1, 3)

    v_index = centered[HAND_INDEX_MCP]
    v_pinky = centered[HAND_PINKY_MCP]
    normal = np.cross(v_index, v_pinky)
    n_norm = float(np.linalg.norm(normal))

    if n_norm > eps:
        r1 = _rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0], dtype=np.float32), eps=eps)
        centered = (r1 @ centered.T).T.astype(np.float32)

    v_middle = centered[HAND_MIDDLE_MCP]
    v_middle_xy = np.array([v_middle[0], v_middle[1]], dtype=np.float32)
    if float(np.linalg.norm(v_middle_xy)) > eps:
        theta = float(np.arctan2(v_middle[0], v_middle[1]))
        r2 = _rotation_z(theta)
        centered = (r2 @ centered.T).T.astype(np.float32)

    axis_len = float(np.linalg.norm(centered[HAND_MIDDLE_MCP]))
    if axis_len > eps:
        centered = centered * float(1.0 / axis_len)

    if not np.all(np.isfinite(centered)):
        centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    centered[HAND_WRIST] = 0.0
    return centered.astype(np.float32)


def _extract_body_subset(
    frame: PoseFrame,
    *,
    upper_body_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(upper_body_indices)
    body = np.zeros((count, 3), dtype=np.float32)
    mask = np.zeros((count,), dtype=np.float32)
    conf = np.zeros((count,), dtype=np.float32)

    if frame.body is None:
        return body, mask, conf

    points = frame.body.points
    visibility = frame.body.confidence

    for i, idx in enumerate(upper_body_indices):
        if idx < 0 or idx >= points.shape[0]:
            continue
        p = points[idx]
        if not np.all(np.isfinite(p)):
            continue
        body[i] = p.astype(np.float32)
        mask[i] = 1.0
        if visibility is None:
            conf[i] = 1.0
        else:
            conf[i] = float(visibility[idx])
    return body, mask, conf


def _extract_hand_group(group: PoseLandmarksGroup | None, *, canonical_3d: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.zeros((21, 3), dtype=np.float32)
    mask = np.zeros((21,), dtype=np.float32)
    conf = np.zeros((21,), dtype=np.float32)

    if group is None:
        return points, mask, conf

    hand_points = group.points.astype(np.float32)
    if hand_points.shape[0] >= 21:
        hand_points = hand_points[:21]
    elif hand_points.shape[0] < 21:
        padded = np.zeros((21, 3), dtype=np.float32)
        padded[: hand_points.shape[0]] = hand_points
        hand_points = padded

    if canonical_3d:
        hand_points = hand_normalize_3d(hand_points)

    points[:] = hand_points
    valid_count = int(min(21, group.points.shape[0]))
    mask[:valid_count] = 1.0

    if group.confidence is None:
        conf[:valid_count] = 1.0
    else:
        conf[:valid_count] = group.confidence[:valid_count].astype(np.float32)

    return points, mask, conf


def compose_features(
    frame: PoseFrame,
    *,
    upper_body_indices: Sequence[int] = DEFAULT_UPPER_BODY_INDICES,
    apply_shoulder_norm: bool = True,
    hide_legs_before_body: bool = True,
    canonical_hands_3d: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames: list[PoseFrame] = [_copy_frame(frame)]
    norm_info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "not_requested")

    if apply_shoulder_norm:
        normalized, norm_info = shoulder_normalize(frames, safe_mode=True)
        frames = list(normalized)

    if hide_legs_before_body:
        frames = list(hide_legs(frames))

    body_pts, body_mask, body_conf = _extract_body_subset(frames[0], upper_body_indices=upper_body_indices)
    left_pts, left_mask, left_conf = _extract_hand_group(frames[0].left_hand, canonical_3d=canonical_hands_3d)
    right_pts, right_mask, right_conf = _extract_hand_group(frames[0].right_hand, canonical_3d=canonical_hands_3d)

    feature = np.concatenate(
        [
            body_pts.reshape(-1),
            left_pts.reshape(-1),
            right_pts.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    point_mask = np.concatenate([body_mask, left_mask, right_mask], axis=0).astype(np.float32)
    point_conf = np.concatenate([body_conf, left_conf, right_conf], axis=0).astype(np.float32)

    aux = {
        "point_mask": point_mask,
        "point_confidence": point_conf,
        "normalization": {
            "normalized": norm_info.normalized,
            "center": norm_info.center.astype(np.float32),
            "scale": float(norm_info.scale),
            "valid_frames": int(norm_info.valid_frames),
            "reason": norm_info.reason,
        },
    }
    return feature, aux


def compose_features_sequence(
    sequence: Sequence[PoseFrame],
    *,
    upper_body_indices: Sequence[int] = DEFAULT_UPPER_BODY_INDICES,
    apply_shoulder_norm: bool = True,
    shoulder_window: int | None = None,
    hide_legs_before_body: bool = True,
    canonical_hands_3d: bool = True,
    include_velocity: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames = [_copy_frame(frame) for frame in sequence]
    if not frames:
        return np.zeros((0, 0), dtype=np.float32), {
            "point_mask": np.zeros((0, 0), dtype=np.float32),
            "point_confidence": np.zeros((0, 0), dtype=np.float32),
            "normalization": {
                "normalized": False,
                "center": np.zeros(3, dtype=np.float32),
                "scale": 1.0,
                "valid_frames": 0,
                "reason": "empty_sequence",
            },
        }

    norm_info = ShoulderNormInfo(False, np.zeros(3, dtype=np.float32), 1.0, 0, "not_requested")
    if apply_shoulder_norm:
        normalized, norm_info = shoulder_normalize(frames, safe_mode=True, window=shoulder_window)
        frames = list(normalized)

    if hide_legs_before_body:
        frames = list(hide_legs(frames))

    feats: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    confs: list[np.ndarray] = []

    for frame in frames:
        feat, aux = compose_features(
            frame,
            upper_body_indices=upper_body_indices,
            apply_shoulder_norm=False,
            hide_legs_before_body=False,
            canonical_hands_3d=canonical_hands_3d,
        )
        feats.append(feat)
        masks.append(np.asarray(aux["point_mask"], dtype=np.float32))
        confs.append(np.asarray(aux["point_confidence"], dtype=np.float32))

    features = np.stack(feats, axis=0).astype(np.float32)
    if include_velocity:
        velocity = np.zeros_like(features)
        if features.shape[0] > 1:
            velocity[1:] = features[1:] - features[:-1]
        features = np.concatenate([features, velocity], axis=1)

    return features, {
        "point_mask": np.stack(masks, axis=0).astype(np.float32),
        "point_confidence": np.stack(confs, axis=0).astype(np.float32),
        "normalization": {
            "normalized": norm_info.normalized,
            "center": norm_info.center.astype(np.float32),
            "scale": float(norm_info.scale),
            "valid_frames": int(norm_info.valid_frames),
            "reason": norm_info.reason,
        },
    }

