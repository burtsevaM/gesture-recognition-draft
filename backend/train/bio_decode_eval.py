from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

BIO_B = 0
BIO_I = 1
BIO_O = 2
BIO_MAPPING = {"B": BIO_B, "I": BIO_I, "O": BIO_O}


@dataclass(frozen=True, slots=True)
class DecodeThresholds:
    th_b: float = 0.5
    th_o: float = 0.5
    min_segment_len: int = 1


def _validate_probs(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"probs must have shape [T, 3], got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("probs must contain at least one frame")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def decode_bio_probs(
    probs: np.ndarray,
    *,
    th_b: float = 0.5,
    th_o: float = 0.5,
    min_segment_len: int = 1,
) -> list[tuple[int, int]]:
    """Decode BIO probabilities into (start, end) segments.

    Rule:
    - start: frame where P(B) > th_b
    - end: frame before P(O) > th_o or before a new start
    """

    p = _validate_probs(probs)
    segments: list[tuple[int, int]] = []

    active = False
    start = 0
    t_len = int(p.shape[0])
    min_len = max(1, int(min_segment_len))

    for t in range(t_len):
        p_b = float(p[t, BIO_B])
        p_o = float(p[t, BIO_O])

        if not active:
            if p_b > th_b:
                active = True
                start = t
            continue

        # Active segment: new start closes previous one.
        if p_b > th_b:
            end = t - 1
            if end >= start and (end - start + 1) >= min_len:
                segments.append((int(start), int(end)))
            start = t
            active = True
            continue

        if p_o > th_o:
            end = t - 1
            if end >= start and (end - start + 1) >= min_len:
                segments.append((int(start), int(end)))
            active = False

    if active:
        end = t_len - 1
        if end >= start and (end - start + 1) >= min_len:
            segments.append((int(start), int(end)))

    return segments


def labels_to_segments(labels: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return []
    segments: list[tuple[int, int]] = []

    active = False
    start = 0

    for i, tag in enumerate(arr.tolist()):
        if not active:
            if tag in (BIO_B, BIO_I):
                active = True
                start = i
            continue

        if tag == BIO_B:
            end = i - 1
            if end >= start:
                segments.append((int(start), int(end)))
            start = i
            active = True
            continue

        if tag == BIO_O:
            end = i - 1
            if end >= start:
                segments.append((int(start), int(end)))
            active = False

    if active:
        end = int(arr.size - 1)
        if end >= start:
            segments.append((int(start), int(end)))

    return segments


def boundaries_from_segments(segments: Sequence[tuple[int, int]]) -> list[int]:
    return [int(start) for start, _ in segments]


def boundary_prf(
    predicted: Sequence[int] | np.ndarray,
    target: Sequence[int] | np.ndarray,
    *,
    tolerance: int = 0,
) -> dict[str, float | int]:
    pred = [int(x) for x in np.asarray(predicted, dtype=np.int64).reshape(-1).tolist()]
    tgt = [int(x) for x in np.asarray(target, dtype=np.int64).reshape(-1).tolist()]

    if not pred and not tgt:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}

    used = [False] * len(tgt)
    tp = 0
    tol = max(0, int(tolerance))

    for p in pred:
        best_idx = -1
        best_dist = 10**9
        for i, t in enumerate(tgt):
            if used[i]:
                continue
            dist = abs(p - t)
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            used[best_idx] = True
            tp += 1

    fp = len(pred) - tp
    fn = len(tgt) - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _segment_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    start = max(int(a[0]), int(b[0]))
    end = min(int(a[1]), int(b[1]))
    if end < start:
        return 0.0
    inter = float(end - start + 1)
    len_a = float(a[1] - a[0] + 1)
    len_b = float(b[1] - b[0] + 1)
    union = len_a + len_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def segment_prf(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float | int]:
    pred = [(int(a), int(b)) for a, b in predicted]
    tgt = [(int(a), int(b)) for a, b in target]

    if not pred and not tgt:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou": 1.0,
        }

    matched_tgt = [False] * len(tgt)
    tp = 0
    iou_sum = 0.0

    for seg in pred:
        best_i = -1
        best_iou = 0.0
        for i, ref in enumerate(tgt):
            if matched_tgt[i]:
                continue
            iou = _segment_iou(seg, ref)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i >= 0:
            matched_tgt[best_i] = True
            tp += 1
            iou_sum += float(best_iou)

    fp = len(pred) - tp
    fn = len(tgt) - tp
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(iou_sum / tp) if tp > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def evaluate_prob_sequences(
    probs_list: Sequence[np.ndarray],
    labels_list: Sequence[np.ndarray],
    *,
    th_b: float,
    th_o: float,
    boundary_tolerance: int = 0,
    segment_iou_threshold: float = 0.5,
    min_segment_len: int = 1,
) -> dict[str, float | int]:
    if len(probs_list) != len(labels_list):
        raise ValueError("probs_list and labels_list must have equal length")

    total_boundary_tp = 0
    total_boundary_fp = 0
    total_boundary_fn = 0
    total_segment_tp = 0
    total_segment_fp = 0
    total_segment_fn = 0
    total_iou_sum = 0.0

    for probs, labels in zip(probs_list, labels_list):
        pred_segments = decode_bio_probs(probs, th_b=th_b, th_o=th_o, min_segment_len=min_segment_len)
        true_segments = labels_to_segments(labels)

        boundary = boundary_prf(
            boundaries_from_segments(pred_segments),
            boundaries_from_segments(true_segments),
            tolerance=boundary_tolerance,
        )
        total_boundary_tp += int(boundary["tp"])
        total_boundary_fp += int(boundary["fp"])
        total_boundary_fn += int(boundary["fn"])

        seg = segment_prf(pred_segments, true_segments, iou_threshold=segment_iou_threshold)
        total_segment_tp += int(seg["tp"])
        total_segment_fp += int(seg["fp"])
        total_segment_fn += int(seg["fn"])
        total_iou_sum += float(seg["mean_iou"]) * int(seg["tp"])

    boundary_precision = (
        float(total_boundary_tp / (total_boundary_tp + total_boundary_fp))
        if (total_boundary_tp + total_boundary_fp) > 0
        else 0.0
    )
    boundary_recall = (
        float(total_boundary_tp / (total_boundary_tp + total_boundary_fn))
        if (total_boundary_tp + total_boundary_fn) > 0
        else 0.0
    )
    boundary_f1 = (
        float((2.0 * boundary_precision * boundary_recall) / (boundary_precision + boundary_recall))
        if (boundary_precision + boundary_recall) > 0
        else 0.0
    )

    segment_precision = (
        float(total_segment_tp / (total_segment_tp + total_segment_fp))
        if (total_segment_tp + total_segment_fp) > 0
        else 0.0
    )
    segment_recall = (
        float(total_segment_tp / (total_segment_tp + total_segment_fn))
        if (total_segment_tp + total_segment_fn) > 0
        else 0.0
    )
    segment_f1 = (
        float((2.0 * segment_precision * segment_recall) / (segment_precision + segment_recall))
        if (segment_precision + segment_recall) > 0
        else 0.0
    )
    segment_mean_iou = float(total_iou_sum / total_segment_tp) if total_segment_tp > 0 else 0.0

    return {
        "th_b": float(th_b),
        "th_o": float(th_o),
        "boundary_tp": int(total_boundary_tp),
        "boundary_fp": int(total_boundary_fp),
        "boundary_fn": int(total_boundary_fn),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "segment_tp": int(total_segment_tp),
        "segment_fp": int(total_segment_fp),
        "segment_fn": int(total_segment_fn),
        "segment_precision": segment_precision,
        "segment_recall": segment_recall,
        "segment_f1": segment_f1,
        "segment_mean_iou": segment_mean_iou,
    }

