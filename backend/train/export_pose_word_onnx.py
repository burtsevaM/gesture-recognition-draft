from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scaffolding: export pose word classifier checkpoint to ONNX.",
    )
    parser.add_argument("--checkpoint", type=Path, required=False, help="Path to trained checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("backend/artifacts/pose_word_model.onnx"), help="Target ONNX path.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("[export_pose_word_onnx] scaffolding only.")
    print("[export_pose_word_onnx] TODO: load real model checkpoint and export to ONNX.")
    print(f"[export_pose_word_onnx] output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

