from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scaffolding: training entrypoint for pose-based isolated word classifier.",
    )
    parser.add_argument("--train-manifest", type=Path, required=False, help="Path to train manifest/jsonl.")
    parser.add_argument("--val-manifest", type=Path, required=False, help="Path to val manifest/jsonl.")
    parser.add_argument("--labels", type=Path, required=False, help="Path to labels.txt.")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/artifacts/pose_words"), help="Output directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[train_pose_word_model] scaffolding only.")
    print("[train_pose_word_model] TODO: implement dataset loader, model, train loop and checkpointing.")
    print(f"[train_pose_word_model] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

