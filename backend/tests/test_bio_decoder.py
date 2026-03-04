from __future__ import annotations

import numpy as np

from train.bio_decode_eval import (
    BIO_B,
    BIO_I,
    BIO_O,
    boundaries_from_segments,
    boundary_prf,
    decode_bio_probs,
    evaluate_prob_sequences,
    labels_to_segments,
)


def _base_probs(t: int) -> np.ndarray:
    probs = np.zeros((t, 3), dtype=np.float32)
    probs[:, BIO_O] = 0.9
    probs[:, BIO_B] = 0.05
    probs[:, BIO_I] = 0.05
    return probs


def test_decode_bio_probs_standard_case() -> None:
    probs = _base_probs(10)
    probs[2] = [0.92, 0.06, 0.02]  # start #1
    probs[3] = [0.10, 0.82, 0.08]
    probs[4] = [0.08, 0.84, 0.08]
    probs[5] = [0.01, 0.02, 0.97]  # close #1
    probs[7] = [0.88, 0.10, 0.02]  # start #2
    probs[8] = [0.08, 0.86, 0.06]
    probs[9] = [0.08, 0.86, 0.06]  # no O -> close at end

    segments = decode_bio_probs(probs, th_b=0.5, th_o=0.5)
    assert segments == [(2, 4), (7, 9)]


def test_decode_bio_probs_consecutive_b_starts_new_segment() -> None:
    probs = _base_probs(7)
    probs[1] = [0.91, 0.07, 0.02]  # start segment 1
    probs[2] = [0.89, 0.08, 0.03]  # new B -> close segment 1 at t=1, start segment 2
    probs[3] = [0.08, 0.84, 0.08]
    probs[4] = [0.02, 0.03, 0.95]  # close segment 2 at t=3

    segments = decode_bio_probs(probs, th_b=0.5, th_o=0.5)
    assert segments == [(1, 1), (2, 3)]


def test_decode_bio_probs_without_o_closes_on_last_frame() -> None:
    probs = _base_probs(8)
    probs[3] = [0.85, 0.10, 0.05]
    probs[4] = [0.10, 0.82, 0.08]
    probs[5] = [0.10, 0.82, 0.08]
    probs[6] = [0.10, 0.82, 0.08]
    probs[7] = [0.10, 0.82, 0.08]

    segments = decode_bio_probs(probs, th_b=0.5, th_o=0.5)
    assert segments == [(3, 7)]


def test_labels_to_segments_handles_i_without_b() -> None:
    labels = np.asarray([BIO_O, BIO_I, BIO_I, BIO_O, BIO_B, BIO_I], dtype=np.int64)
    segments = labels_to_segments(labels)
    assert segments == [(1, 2), (4, 5)]
    assert boundaries_from_segments(segments) == [1, 4]


def test_boundary_and_segment_metrics_smoke() -> None:
    labels = np.asarray([BIO_O, BIO_B, BIO_I, BIO_O, BIO_B, BIO_I, BIO_O], dtype=np.int64)
    true_segments = labels_to_segments(labels)
    assert true_segments == [(1, 2), (4, 5)]

    probs = _base_probs(len(labels))
    probs[1] = [0.95, 0.03, 0.02]
    probs[2] = [0.05, 0.90, 0.05]
    probs[3] = [0.01, 0.01, 0.98]
    probs[4] = [0.94, 0.03, 0.03]
    probs[5] = [0.03, 0.93, 0.04]
    probs[6] = [0.01, 0.02, 0.97]

    pred_segments = decode_bio_probs(probs, th_b=0.5, th_o=0.5)
    boundary = boundary_prf(boundaries_from_segments(pred_segments), boundaries_from_segments(true_segments), tolerance=0)
    assert boundary["f1"] == 1.0

    metrics = evaluate_prob_sequences([probs], [labels], th_b=0.5, th_o=0.5, boundary_tolerance=0, segment_iou_threshold=0.5)
    assert metrics["boundary_f1"] == 1.0
    assert metrics["segment_f1"] == 1.0

