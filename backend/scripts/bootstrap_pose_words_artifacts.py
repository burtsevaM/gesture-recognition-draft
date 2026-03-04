#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
POSE_GENERATOR = ROOT_DIR / "backend" / "train" / "make_dummy_pose_word_model.py"
BIO_GENERATOR = ROOT_DIR / "backend" / "train" / "make_dummy_bio_segmenter.py"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap required pose_words artifacts locally.")
    parser.add_argument("--force", action="store_true", help="Re-generate artifacts even if files already exist")
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=159)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("backend/artifacts"),
        help="Artifacts directory for generated ONNX/config files",
    )
    return parser


def _missing_paths(mapping: dict[str, Path]) -> dict[str, Path]:
    return {name: path for name, path in mapping.items() if not path.exists()}


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _print_status(title: str, mapping: dict[str, Path]) -> None:
    print(f"[bootstrap] {title}")
    for name, path in mapping.items():
        exists = "ok" if path.exists() else "missing"
        print(f"  - {name}: {exists} ({path})")


def _artifact_map(artifacts_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    pose_artifacts = {
        "pose_word_model.onnx": artifacts_dir / "pose_word_model.onnx",
        "pose_word_labels.txt": artifacts_dir / "pose_word_labels.txt",
        "pose_word_config.json": artifacts_dir / "pose_word_config.json",
    }
    bio_artifacts = {
        "bio_segmenter.onnx": artifacts_dir / "bio_segmenter.onnx",
        "bio_thresholds.json": artifacts_dir / "bio_thresholds.json",
        "bio_config.json": artifacts_dir / "bio_config.json",
    }
    return pose_artifacts, bio_artifacts


def main() -> int:
    args = _build_parser().parse_args()

    force = bool(args.force)
    artifacts_dir = args.artifacts_dir
    if not artifacts_dir.is_absolute():
        artifacts_dir = (ROOT_DIR / artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pose_artifacts, bio_artifacts = _artifact_map(artifacts_dir)
    pose_missing = _missing_paths(pose_artifacts)
    bio_missing = _missing_paths(bio_artifacts)

    if force or pose_missing:
        print("[bootstrap] generating pose_word artifacts...")
        _run(
            [
                sys.executable,
                str(POSE_GENERATOR),
                "--onnx-out",
                str(pose_artifacts["pose_word_model.onnx"]),
                "--labels-path",
                str(pose_artifacts["pose_word_labels.txt"]),
                "--config-out",
                str(pose_artifacts["pose_word_config.json"]),
                "--clip-frames",
                str(max(4, int(args.clip_frames))),
                "--feature-dim",
                str(max(8, int(args.feature_dim))),
            ]
        )
    else:
        print("[bootstrap] pose_word artifacts already exist; skipping.")

    if force or bio_missing:
        print("[bootstrap] generating BIO artifacts...")
        _run(
            [
                sys.executable,
                str(BIO_GENERATOR),
                "--onnx-out",
                str(bio_artifacts["bio_segmenter.onnx"]),
                "--thresholds-out",
                str(bio_artifacts["bio_thresholds.json"]),
                "--config-out",
                str(bio_artifacts["bio_config.json"]),
                "--window-size",
                str(max(8, int(args.window_size))),
                "--feature-dim",
                str(max(8, int(args.feature_dim))),
            ]
        )
    else:
        print("[bootstrap] BIO artifacts already exist; skipping.")

    pose_missing_after = _missing_paths(pose_artifacts)
    bio_missing_after = _missing_paths(bio_artifacts)
    all_missing = {**pose_missing_after, **bio_missing_after}
    if all_missing:
        _print_status("final status", {**pose_artifacts, **bio_artifacts})
        missing_names = ", ".join(sorted(all_missing.keys()))
        raise RuntimeError(f"bootstrap finished with missing artifacts: {missing_names}")

    _print_status("final status", {**pose_artifacts, **bio_artifacts})
    print("[bootstrap] pose_words artifacts are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
