from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from .bio_decode_eval import BIO_MAPPING
    from .train_pose_bio_segmenter import PoseBioSegmenter
except ImportError:  # pragma: no cover - script mode fallback
    from bio_decode_eval import BIO_MAPPING
    from train_pose_bio_segmenter import PoseBioSegmenter


class BioOnnxWrapper(nn.Module):
    def __init__(self, model: PoseBioSegmenter) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sign_logits, phrase_logits = self.model(features, None)
        sign_probs = torch.softmax(sign_logits, dim=-1)
        phrase_probs = torch.softmax(phrase_logits, dim=-1)
        return sign_probs, phrase_probs


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained BIO segmenter to ONNX")
    parser.add_argument("--checkpoint", default="backend/artifacts/bio_training/best_model.pt")
    parser.add_argument("--onnx-out", default="backend/artifacts/bio_segmenter.onnx")
    parser.add_argument("--thresholds-in", default="backend/artifacts/bio_thresholds.json")
    parser.add_argument("--thresholds-out", default="backend/artifacts/bio_thresholds.json")
    parser.add_argument("--bio-config-out", default="backend/artifacts/bio_config.json")
    parser.add_argument("--window-size", type=int, default=256, help="Dummy time length used for export")
    parser.add_argument("--dynamic-time", type=parse_bool, default=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    feature_dim = int(checkpoint.get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError("checkpoint missing feature_dim")

    model_cfg = checkpoint.get("model_config", {})
    model = PoseBioSegmenter(
        input_dim=feature_dim,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        bidirectional=bool(model_cfg.get("bidirectional", True)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    wrapper = BioOnnxWrapper(model).eval()
    window_size = max(1, int(args.window_size))
    dummy = torch.randn(1, window_size, feature_dim, dtype=torch.float32)

    onnx_out = Path(args.onnx_out).resolve()
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if bool(args.dynamic_time):
        dynamic_axes = {
            "features": {1: "time"},
            "sign_probs": {1: "time"},
            "phrase_probs": {1: "time"},
        }

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_out),
            input_names=["features"],
            output_names=["sign_probs", "phrase_probs"],
            dynamic_axes=dynamic_axes,
            opset_version=int(args.opset),
        )

    thresholds_in = Path(args.thresholds_in).resolve()
    thresholds_out = Path(args.thresholds_out).resolve()
    if thresholds_in.exists():
        thresholds_out.parent.mkdir(parents=True, exist_ok=True)
        if thresholds_in != thresholds_out:
            shutil.copyfile(thresholds_in, thresholds_out)
    else:
        fallback = {
            "bio_mapping": BIO_MAPPING,
            "sign": {"th_b": 0.5, "th_o": 0.5},
            "phrase": {"th_b": 0.5, "th_o": 0.5},
            "note": "thresholds file was absent during export",
        }
        save_json(thresholds_out, fallback)

    train_cfg = checkpoint.get("train_config", {})
    bio_config = {
        "input_dim": feature_dim,
        "window_size": window_size,
        "dynamic_time": bool(args.dynamic_time),
        "model_config": {
            "hidden_size": int(model_cfg.get("hidden_size", 256)),
            "num_layers": int(model_cfg.get("num_layers", 2)),
            "dropout": float(model_cfg.get("dropout", 0.2)),
            "bidirectional": bool(model_cfg.get("bidirectional", True)),
        },
        "bio_mapping": BIO_MAPPING,
        "norm_flags": {
            "use_shoulder_norm": bool(train_cfg.get("use_shoulder_norm", True)),
            "use_hands_3d_norm": bool(train_cfg.get("use_hands_3d_norm", False)),
        },
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(onnx_out),
    }
    save_json(Path(args.bio_config_out).resolve(), bio_config)

    print("[export_bio_segmenter_onnx] done")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  onnx={onnx_out}")
    print(f"  thresholds={thresholds_out}")
    print(f"  bio_config={Path(args.bio_config_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
