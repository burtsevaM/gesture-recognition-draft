from __future__ import annotations

from typing import Sequence

import numpy as np


def segments_per_minute(*, segment_count: int, total_frames: int, fps: float) -> float:
    if total_frames <= 0 or fps <= 0:
        return 0.0
    minutes = (float(total_frames) / float(fps)) / 60.0
    if minutes <= 0:
        return 0.0
    return float(segment_count) / minutes


def average_segment_length_frames(segments: Sequence[tuple[int, int, float]] | Sequence[tuple[int, int]]) -> float:
    if not segments:
        return 0.0
    lengths = []
    for seg in segments:
        s = int(seg[0])
        e = int(seg[1])
        if e < s:
            continue
        lengths.append(e - s + 1)
    if not lengths:
        return 0.0
    return float(np.mean(np.asarray(lengths, dtype=np.float32)))


def average_segment_length_seconds(
    segments: Sequence[tuple[int, int, float]] | Sequence[tuple[int, int]],
    *,
    fps: float,
) -> float:
    if fps <= 0:
        return 0.0
    return average_segment_length_frames(segments) / float(fps)


def estimate_fp_per_minute(
    segments: Sequence[tuple[int, int, float]] | Sequence[tuple[int, int]],
    *,
    total_frames: int,
    fps: float,
    quiet_mode: bool,
    motion_per_frame: np.ndarray | None = None,
    motion_threshold: float = 0.02,
) -> float:
    if total_frames <= 0 or fps <= 0:
        return 0.0

    if quiet_mode:
        fp_count = len(segments)
        return segments_per_minute(segment_count=fp_count, total_frames=total_frames, fps=fps)

    if motion_per_frame is None or motion_per_frame.size == 0:
        return 0.0

    motion = np.asarray(motion_per_frame, dtype=np.float32).reshape(-1)
    fp_count = 0
    for seg in segments:
        s = max(0, int(seg[0]))
        e = min(int(total_frames) - 1, int(seg[1]))
        if e < s:
            continue
        mean_motion = float(np.mean(motion[s : e + 1]))
        if mean_motion < float(motion_threshold):
            fp_count += 1
    return segments_per_minute(segment_count=fp_count, total_frames=total_frames, fps=fps)


def boundary_jitter(
    snapshots: Sequence[Sequence[tuple[int, int, float]] | Sequence[tuple[int, int]]],
) -> float:
    """Jitter as average symmetric boundary delta between consecutive snapshots."""

    if len(snapshots) < 2:
        return 0.0

    diffs: list[float] = []
    prev_bounds: set[int] | None = None
    for snapshot in snapshots:
        bounds: set[int] = set()
        for seg in snapshot:
            s = int(seg[0])
            e = int(seg[1])
            if e < s:
                continue
            bounds.add(s)
            bounds.add(e)
        if prev_bounds is not None:
            diff = prev_bounds.symmetric_difference(bounds)
            diffs.append(float(len(diff)))
        prev_bounds = bounds

    if not diffs:
        return 0.0
    return float(np.mean(np.asarray(diffs, dtype=np.float32)))


def stability_score(
    snapshots: Sequence[Sequence[tuple[int, int, float]] | Sequence[tuple[int, int]]],
) -> float:
    jitter = boundary_jitter(snapshots)
    return float(1.0 / (1.0 + max(0.0, jitter)))
