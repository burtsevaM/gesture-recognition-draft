from __future__ import annotations

import numpy as np

from app.pose_words.segment_utils import clamp_indices, extract_segment
from app.segmentation.streaming import StreamingBioSegmenter


class _DummyBioModel:
    def infer(self, features_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        t_len = int(features_tf.shape[0])
        probs = np.zeros((t_len, 3), dtype=np.float32)
        probs[:, 2] = 1.0
        return probs, probs.copy(), 0.1


def test_extract_segment_with_global_indices_and_clamp() -> None:
    feats = np.stack([np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) * 10], axis=1)
    seg = extract_segment({"features": feats, "start_idx": 100}, 103, 106)
    assert seg.shape == (4, 2)
    np.testing.assert_allclose(seg[:, 0], np.asarray([3, 4, 5, 6], dtype=np.float32))

    seg2 = extract_segment({"features": feats, "start_idx": 100}, 95, 104)
    assert seg2.shape == (5, 2)
    np.testing.assert_allclose(seg2[:, 0], np.asarray([0, 1, 2, 3, 4], dtype=np.float32))

    assert clamp_indices(9, 2, 10) == (2, 9)


def test_streaming_ring_wrap_around_span_extraction() -> None:
    segmenter = StreamingBioSegmenter(
        model=_DummyBioModel(),  # type: ignore[arg-type]
        window=8,
        step=99,
        min_len=2,
        max_len=20,
        merge_gap=0,
        cool_off_frames=0,
        max_buffer=5,
    )

    for i in range(25):
        feature = np.asarray([float(i), float(i) + 0.5], dtype=np.float32)
        segmenter.update(feature)

    # Internal max_buffer is hard-capped to at least window*2, so here it keeps [9..24].
    span = segmenter.get_feature_span(20, 22)
    assert span is not None
    assert span.shape == (3, 2)
    np.testing.assert_allclose(span[:, 0], np.asarray([20.0, 21.0, 22.0], dtype=np.float32))

    assert segmenter.get_feature_span(3, 5) is None
