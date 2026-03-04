from __future__ import annotations

import numpy as np

from app.pose_words.segment_utils import resample_to_fixed_T


def test_resample_short_segment_to_fixed_32_with_padding() -> None:
    seg = np.asarray(
        [
            [0.0, 10.0, 100.0],
            [1.0, 11.0, 101.0],
            [2.0, 12.0, 102.0],
            [3.0, 13.0, 103.0],
            [4.0, 14.0, 104.0],
        ],
        dtype=np.float32,
    )
    out = resample_to_fixed_T(seg, T=32, method="linear")
    assert out.shape == (32, 3)
    np.testing.assert_allclose(out[:5], seg, atol=1e-6)
    expected_tail = np.repeat(seg[-1][None, :], repeats=27, axis=0)
    np.testing.assert_allclose(out[5:], expected_tail, atol=1e-6)


def test_resample_long_segment_to_fixed_32_linear() -> None:
    t = np.arange(120, dtype=np.float32)
    seg = np.stack([t, t * 2.0, t * 0.1], axis=1).astype(np.float32)
    out = resample_to_fixed_T(seg, T=32, method="linear")
    assert out.shape == (32, 3)
    np.testing.assert_allclose(out[0], seg[0], atol=1e-4)
    np.testing.assert_allclose(out[-1], seg[-1], atol=1e-4)
    assert np.all(np.diff(out[:, 0]) >= -1e-5)


def test_resample_nan_guard() -> None:
    seg = np.asarray([[0.0, np.nan], [1.0, 2.0]], dtype=np.float32)
    out = resample_to_fixed_T(seg, T=8, method="linear")
    assert out.shape == (8, 2)
    assert np.all(np.isfinite(out))
