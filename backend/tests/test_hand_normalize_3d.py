from __future__ import annotations

import numpy as np

from app.pose.normalization import hand_normalize_3d


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float32,
    )


def _synthetic_hand() -> np.ndarray:
    # Простая устойчивая 3D-конфигурация кисти (не коллинеарная).
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = [0.0, 0.0, 0.0]    # wrist
    hand[5] = [0.35, 0.20, 0.04]  # index_mcp
    hand[9] = [0.0, 0.65, 0.03]   # middle_mcp
    hand[13] = [-0.28, 0.52, 0.01]
    hand[17] = [-0.45, 0.18, -0.02]  # pinky_mcp

    # Заполняем остальные точки небольшими смещениями.
    for idx in range(21):
        if np.allclose(hand[idx], 0.0) and idx != 0:
            hand[idx] = np.array([
                0.02 * ((idx % 4) - 1.5),
                0.12 + 0.03 * idx,
                0.01 * ((idx % 3) - 1.0),
            ], dtype=np.float32)
    return hand


def test_hand_normalize_3d_invariant_to_rigid_transform_and_scale() -> None:
    base = _synthetic_hand()

    rot = _axis_angle_rotation(np.array([0.7, -0.3, 0.6], dtype=np.float32), angle=1.1)
    scale = 2.75
    shift = np.array([1.3, -2.1, 0.9], dtype=np.float32)

    transformed = ((rot @ base.T).T * scale) + shift

    canon_base = hand_normalize_3d(base)
    canon_trans = hand_normalize_3d(transformed)

    diff = np.max(np.abs(canon_base - canon_trans))
    assert diff < 2e-4

    # Дополнительные sanity checks
    assert np.allclose(canon_base[0], np.zeros(3, dtype=np.float32), atol=1e-6)
    middle_len = float(np.linalg.norm(canon_base[9]))
    assert np.isclose(middle_len, 1.0, atol=1e-5)

