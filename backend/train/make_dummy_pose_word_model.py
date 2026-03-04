#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATH = ROOT_DIR / "backend" / "artifacts" / "pose_word_model.onnx"
DEFAULT_LABELS_PATH = ROOT_DIR / "backend" / "artifacts" / "pose_word_labels.txt"
DEFAULT_CONFIG_PATH = ROOT_DIR / "backend" / "artifacts" / "pose_word_config.json"

DEFAULT_LABELS = [
    "_no_event",
    "UNKNOWN",
    "привет",
    "пока",
    "спасибо",
    "да",
    "нет",
]


class DummyPoseWordClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, T, F] -> average over time -> [B, F]
        pooled = features.mean(dim=1)
        return self.net(pooled)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT_DIR / p).resolve()


def _ensure_onnx_dependency() -> None:
    if importlib.util.find_spec("onnx") is not None:
        return
    raise RuntimeError(
        "Python package 'onnx' is required for export. Install dependencies via: "
        "pip install -r backend/requirements.txt"
    )


def _load_or_create_labels(path: Path) -> list[str]:
    if path.exists():
        labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels
    labels = list(DEFAULT_LABELS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create dummy pose-word ONNX model for local bootstrap.")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--config-out", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=159)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _ensure_onnx_dependency()

    onnx_path = _resolve(args.onnx_out)
    labels_path = _resolve(args.labels_path)
    config_path = _resolve(args.config_out)

    clip_frames = max(4, int(args.clip_frames))
    feature_dim = max(8, int(args.feature_dim))

    labels = _load_or_create_labels(labels_path)
    model = DummyPoseWordClassifier(input_dim=feature_dim, num_classes=len(labels)).eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, clip_frames, feature_dim, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes=None,
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    payload = {
        "generated_by": "backend/train/make_dummy_pose_word_model.py",
        "input": {
            "name": "features",
            "shape": [1, clip_frames, feature_dim],
            "dtype": "float32",
        },
        "output": {
            "name": "logits",
            "shape": [1, len(labels)],
            "dtype": "float32",
        },
        "model": {
            "type": "dummy_pose_word_classifier",
            "clip_frames": clip_frames,
            "input_dim": feature_dim,
            "num_classes": len(labels),
        },
        "labels_total": len(labels),
        "feature_schema": "compose_features: body(11x3)+left_hand(21x3)+right_hand(21x3)=159",
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[make_dummy_pose_word_model] done")
    print(f"  onnx={onnx_path}")
    print(f"  labels={labels_path}")
    print(f"  config={config_path}")
    print(f"  clip_frames={clip_frames} feature_dim={feature_dim} classes={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
