#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ONNX_PATH = ROOT_DIR / "backend" / "artifacts" / "bio_segmenter.onnx"
DEFAULT_THRESHOLDS_PATH = ROOT_DIR / "backend" / "artifacts" / "bio_thresholds.json"
DEFAULT_CONFIG_PATH = ROOT_DIR / "backend" / "artifacts" / "bio_config.json"


class DummyBioSegmenter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sign_head = nn.Linear(hidden_dim, 3)
        self.phrase_head = nn.Linear(hidden_dim, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # features: [B, T, F] -> frame-wise heads [B, T, 3]
        x = self.backbone(features)
        sign_probs = torch.softmax(self.sign_head(x), dim=-1)
        phrase_probs = torch.softmax(self.phrase_head(x), dim=-1)
        return sign_probs, phrase_probs


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT_DIR / p).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path.resolve())


def _ensure_onnx_dependency() -> None:
    if importlib.util.find_spec("onnx") is not None:
        return
    raise RuntimeError(
        "Python package 'onnx' is required for export. Install dependencies via: "
        "pip install -r backend/requirements.txt"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create dummy BIO ONNX model for local bootstrap.")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--thresholds-out", type=Path, default=DEFAULT_THRESHOLDS_PATH)
    parser.add_argument("--config-out", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=159)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _ensure_onnx_dependency()

    onnx_path = _resolve(args.onnx_out)
    thresholds_path = _resolve(args.thresholds_out)
    config_path = _resolve(args.config_out)

    window_size = max(8, int(args.window_size))
    feature_dim = max(8, int(args.feature_dim))
    hidden_dim = max(8, int(args.hidden_dim))

    model = DummyBioSegmenter(input_dim=feature_dim, hidden_dim=hidden_dim).eval()

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, window_size, feature_dim, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["features"],
            output_names=["sign_probs", "phrase_probs"],
            dynamic_axes={"features": {1: "time"}, "sign_probs": {1: "time"}, "phrase_probs": {1: "time"}},
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    thresholds_payload = {
        "generated_by": "backend/train/make_dummy_bio_segmenter.py",
        "artifact_kind": "dummy",
        "dataset_kind": "synthetic_fixture",
        "trained": False,
        "source_pipeline": "bootstrap_pose_words_artifacts",
        "bio_mapping": {"B": 0, "I": 1, "O": 2},
        "sign": {"th_b": 0.6, "th_o": 0.6},
        "phrase": {"th_b": 0.6, "th_o": 0.6},
        "min_len": 6,
        "merge_gap": 2,
    }
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(thresholds_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config_payload = {
        "generated_by": "backend/train/make_dummy_bio_segmenter.py",
        "artifact_kind": "dummy",
        "dataset_kind": "synthetic_fixture",
        "trained": False,
        "source_pipeline": "bootstrap_pose_words_artifacts",
        "input_dim": feature_dim,
        "window_size": window_size,
        "dynamic_time": True,
        "model": {
            "type": "dummy_bio_segmenter",
            "hidden_dim": hidden_dim,
        },
        "onnx_path": _display_path(onnx_path),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[make_dummy_bio_segmenter] done")
    print(f"  onnx={onnx_path}")
    print(f"  thresholds={thresholds_path}")
    print(f"  config={config_path}")
    print(f"  window_size={window_size} feature_dim={feature_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
