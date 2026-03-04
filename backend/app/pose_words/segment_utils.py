from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def clamp_indices(start: int, end: int, buf_len: int) -> tuple[int, int]:
    """Clamp [start, end] to available buffer range [0, buf_len-1]."""

    if buf_len <= 0:
        logger.warning("clamp_indices: empty buffer, returning (0, -1)")
        return 0, -1

    s = int(start)
    e = int(end)
    if s > e:
        logger.warning("clamp_indices: swapped invalid range start=%s end=%s", s, e)
        s, e = e, s

    cs = max(0, min(s, buf_len - 1))
    ce = max(0, min(e, buf_len - 1))
    if (cs, ce) != (s, e):
        logger.warning(
            "clamp_indices: range clamped from [%s, %s] to [%s, %s] (buf_len=%s)",
            s,
            e,
            cs,
            ce,
            buf_len,
        )
    return cs, ce


def _as_buffer_array(features_buffer: Any) -> tuple[np.ndarray, int]:
    """Convert feature buffer to (features[N,F], global_start_idx)."""

    global_start = 0
    arr: np.ndarray

    if isinstance(features_buffer, tuple) and len(features_buffer) == 2:
        arr = np.asarray(features_buffer[0], dtype=np.float32)
        global_start = int(features_buffer[1])
    elif isinstance(features_buffer, dict):
        arr = np.asarray(features_buffer.get("features"), dtype=np.float32)
        global_start = int(features_buffer.get("start_idx", 0))
    else:
        arr = np.asarray(features_buffer, dtype=np.float32)

    if arr.ndim != 2:
        raise ValueError(f"features_buffer must have shape [N, F], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr.astype(np.float32, copy=False), global_start


def extract_segment(features_buffer: Any, start_idx: int, end_idx: int) -> np.ndarray:
    """Extract [start_idx, end_idx] from feature buffer.

    Supported buffer formats:
    - np.ndarray [N, F] with local indices
    - tuple(np.ndarray [N, F], global_start_idx)
    - {"features": np.ndarray [N, F], "start_idx": int}
    """

    feats, global_start = _as_buffer_array(features_buffer)
    if feats.shape[0] == 0:
        return np.zeros((0, feats.shape[1] if feats.ndim == 2 else 0), dtype=np.float32)

    local_start = int(start_idx) - global_start
    local_end = int(end_idx) - global_start
    clamped_start, clamped_end = clamp_indices(local_start, local_end, int(feats.shape[0]))
    if clamped_end < clamped_start:
        return np.zeros((0, feats.shape[1]), dtype=np.float32)

    segment = feats[clamped_start : clamped_end + 1]
    if segment.size == 0:
        return np.zeros((0, feats.shape[1]), dtype=np.float32)
    return np.asarray(segment, dtype=np.float32).copy()


def resample_to_fixed_T(seg: np.ndarray, T: int = 32, method: str = "linear") -> np.ndarray:
    """Resample segment [Tseg,F] to fixed [T,F] robustly."""

    target = max(1, int(T))
    arr = np.asarray(seg, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"seg must have shape [Tseg, F], got {arr.shape}")

    if arr.shape[0] == 0:
        return np.zeros((target, arr.shape[1]), dtype=np.float32)

    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    tseg, feat_dim = int(arr.shape[0]), int(arr.shape[1])
    if tseg == target:
        return arr.astype(np.float32, copy=True)

    if tseg < target:
        out = np.zeros((target, feat_dim), dtype=np.float32)
        out[:tseg] = arr
        out[tseg:] = arr[-1]
        return out

    mode = str(method).strip().lower()
    if mode == "index":
        idx = np.linspace(0, tseg - 1, num=target, dtype=np.float32)
        idx = np.clip(np.round(idx).astype(np.int32), 0, tseg - 1)
        return arr[idx].astype(np.float32, copy=False)

    # linear interpolation by default
    source_x = np.linspace(0, tseg - 1, num=tseg, dtype=np.float32)
    target_x = np.linspace(0, tseg - 1, num=target, dtype=np.float32)
    out = np.zeros((target, feat_dim), dtype=np.float32)
    for col in range(feat_dim):
        out[:, col] = np.interp(target_x, source_x, arr[:, col])
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out
