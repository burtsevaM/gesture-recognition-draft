from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scaffolding: training entrypoint for pose-based BIO segmenter (B/I/O).",
    )
    parser.add_argument("--train-manifest", type=Path, required=False, help="Path to train manifest/jsonl.")
    parser.add_argument("--val-manifest", type=Path, required=False, help="Path to val manifest/jsonl.")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/artifacts/pose_bio"), help="Output directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--class-weights", action="store_true", help="Enable class weights for B/I/O imbalance.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[train_pose_bio_segmenter] scaffolding only.")
    print("[train_pose_bio_segmenter] TODO: implement BIO labels pipeline, model, train loop and metrics.")
    print(f"[train_pose_bio_segmenter] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

