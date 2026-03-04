from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from app.pose_words import PoseWordOnnxModel

pytest.importorskip("onnx")


class _TinyPoseWordModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.fc(pooled)



def _export_tiny_model(path: Path, *, clip_frames: int, input_dim: int, num_classes: int) -> None:
    model = _TinyPoseWordModel(input_dim=input_dim, num_classes=num_classes)
    model.eval()
    dummy = torch.randn(1, clip_frames, input_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )



def test_pose_word_onnx_infer_probs_smoke(tmp_path: Path) -> None:
    onnx_path = tmp_path / "pose_word.onnx"
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("_no_event\nHELLO\nTHANKS\n", encoding="utf-8")

    _export_tiny_model(onnx_path, clip_frames=32, input_dim=8, num_classes=3)

    model = PoseWordOnnxModel(model_path=onnx_path, labels_path=labels_path, ort_num_threads=1)
    features = np.random.randn(32, 8).astype(np.float32)
    probs, latency_ms = model.infer_probs(features)

    assert probs.shape == (3,)
    assert np.all(np.isfinite(probs))
    assert pytest.approx(float(np.sum(probs)), rel=1e-4, abs=1e-4) == 1.0
    assert latency_ms >= 0.0
    assert model.find_no_event_index("_no_event") == 0



def test_pose_word_onnx_raises_on_bad_feature_dim(tmp_path: Path) -> None:
    onnx_path = tmp_path / "pose_word.onnx"
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("A\nB\n", encoding="utf-8")

    _export_tiny_model(onnx_path, clip_frames=16, input_dim=6, num_classes=2)
    model = PoseWordOnnxModel(model_path=onnx_path, labels_path=labels_path, ort_num_threads=1)

    bad_features = np.random.randn(16, 5).astype(np.float32)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        model.infer_probs(bad_features)
