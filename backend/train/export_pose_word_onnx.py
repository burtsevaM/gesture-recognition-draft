#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from train.pose_word_arch import PoseWordClassifier  # noqa: E402



def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export trained pose-word classifier checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("backend/artifacts/pose_word_training/pose_word_best.pt"),
        help="Path to checkpoint from train_pose_word_model.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("backend/artifacts/pose_word_model.onnx"),
        help="Target ONNX path",
    )
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=Path("backend/artifacts/pose_word_labels.txt"),
        help="Output labels file",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path("backend/artifacts/pose_word_config.json"),
        help="Output model config json",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", type=parse_bool, default=False)
    parser.add_argument("--verify", type=parse_bool, default=True, help="Run ONNXRuntime sanity check after export")
    return parser



def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()



def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must be a dict")
    return payload



def _build_model(ckpt: dict[str, Any]) -> tuple[PoseWordClassifier, dict[str, Any], list[str]]:
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint missing model_state")

    labels = ckpt.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError("checkpoint missing labels")
    labels = [str(x) for x in labels]

    model_cfg = ckpt.get("model_config")
    if not isinstance(model_cfg, dict):
        raise ValueError("checkpoint missing model_config")

    input_dim = int(model_cfg.get("input_dim", 0))
    num_classes = int(model_cfg.get("num_classes", len(labels)))
    clip_frames = int(model_cfg.get("clip_frames", 32))
    conv_channels = int(model_cfg.get("conv_channels", 192))
    gru_hidden = int(model_cfg.get("gru_hidden", 192))
    gru_layers = int(model_cfg.get("gru_layers", 1))
    dropout = float(model_cfg.get("dropout", 0.2))

    if input_dim <= 0:
        raise ValueError(f"invalid input_dim in checkpoint: {input_dim}")
    if num_classes != len(labels):
        raise ValueError(f"num_classes mismatch: {num_classes} vs labels={len(labels)}")

    model = PoseWordClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        conv_channels=conv_channels,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
        dropout=dropout,
    )
    model.load_state_dict(model_state)
    model.eval()

    normalized_cfg = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "clip_frames": clip_frames,
        "conv_channels": conv_channels,
        "gru_hidden": gru_hidden,
        "gru_layers": gru_layers,
        "dropout": dropout,
    }
    return model, normalized_cfg, labels



def _export_onnx(
    model: PoseWordClassifier,
    *,
    output_path: Path,
    clip_frames: int,
    input_dim: int,
    opset: int,
    dynamic_batch: bool,
) -> None:
    dummy = torch.randn(1, int(clip_frames), int(input_dim), dtype=torch.float32)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "features": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(opset),
        do_constant_folding=True,
    )



def _verify_export(model: PoseWordClassifier, onnx_path: Path, *, clip_frames: int, input_dim: int) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("onnxruntime is required for --verify true") from exc

    x = np.random.randn(1, int(clip_frames), int(input_dim)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(x)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_out = session.run([output_name], {input_name: x})[0]

    if onnx_out.shape != torch_out.shape:
        raise RuntimeError(f"shape mismatch torch={torch_out.shape} onnx={onnx_out.shape}")

    max_abs = float(np.max(np.abs(torch_out - onnx_out)))
    mean_abs = float(np.mean(np.abs(torch_out - onnx_out)))
    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }



def _extract_norm_flags(ckpt: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "use_shoulder_norm": False,
        "use_hands_3d_norm": False,
    }
    dataset_flags = ckpt.get("dataset_flags")
    if isinstance(dataset_flags, dict):
        for key in flags.keys():
            if key in dataset_flags:
                flags[key] = bool(dataset_flags.get(key))
    return flags



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    checkpoint_path = _resolve_path(Path(args.checkpoint))
    output_path = _resolve_path(Path(args.output))
    labels_path = _resolve_path(Path(args.labels_out))
    config_path = _resolve_path(Path(args.config_out))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(checkpoint_path)
    model, model_cfg, labels = _build_model(ckpt)

    _export_onnx(
        model,
        output_path=output_path,
        clip_frames=int(model_cfg["clip_frames"]),
        input_dim=int(model_cfg["input_dim"]),
        opset=int(args.opset),
        dynamic_batch=bool(args.dynamic_batch),
    )

    verify_stats: dict[str, float] | None = None
    if bool(args.verify):
        verify_stats = _verify_export(
            model,
            output_path,
            clip_frames=int(model_cfg["clip_frames"]),
            input_dim=int(model_cfg["input_dim"]),
        )

    labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    config_payload = {
        "checkpoint": str(checkpoint_path),
        "onnx_path": str(output_path),
        "labels_path": str(labels_path),
        "input": {
            "name": "features",
            "shape": [1, int(model_cfg["clip_frames"]), int(model_cfg["input_dim"])],
            "dtype": "float32",
        },
        "output": {
            "name": "logits",
            "shape": [1, int(model_cfg["num_classes"])],
            "dtype": "float32",
        },
        "model": {
            "type": "1dcnn_bigru",
            **model_cfg,
        },
        "labels_total": len(labels),
        "norm_flags": _extract_norm_flags(ckpt),
        "dynamic_batch": bool(args.dynamic_batch),
        "verify": verify_stats,
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[export_pose_word_onnx] done")
    print(f"  checkpoint={checkpoint_path}")
    print(f"  onnx={output_path}")
    print(f"  labels={labels_path}")
    print(f"  config={config_path}")
    if verify_stats is not None:
        print(f"  verify.max_abs_diff={verify_stats['max_abs_diff']:.6f}")
        print(f"  verify.mean_abs_diff={verify_stats['mean_abs_diff']:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
