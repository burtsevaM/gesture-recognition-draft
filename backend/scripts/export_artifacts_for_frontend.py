#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC_DIR = REPO_ROOT / "backend" / "artifacts"
DEFAULT_DST_DIR = REPO_ROOT / "frontend" / "assets" / "models"


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    required: bool
    role: str


ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(name="pose_word_model.onnx", required=False, role="pose_word_model"),
    ArtifactSpec(name="bio_segmenter.onnx", required=False, role="bio_segmenter"),
    ArtifactSpec(name="pose_word_labels.txt", required=False, role="pose_word_labels"),
    ArtifactSpec(name="labels.txt", required=False, role="words_labels"),
    ArtifactSpec(name="pose_word_config.json", required=False, role="pose_word_config"),
    ArtifactSpec(name="bio_config.json", required=False, role="bio_config"),
    ArtifactSpec(name="bio_thresholds.json", required=False, role="bio_thresholds"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def copy_artifact(src_dir: Path, dst_dir: Path, spec: ArtifactSpec) -> dict[str, Any] | None:
    src = src_dir / spec.name
    if not src.exists():
        if spec.required:
            raise FileNotFoundError(f"required artifact is missing: {src}")
        return None

    dst = dst_dir / spec.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    stat = dst.stat()
    return {
        "name": spec.name,
        "role": spec.role,
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(dst),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
    }


def build_manifest(src_dir: Path, dst_dir: Path, files: list[dict[str, Any]]) -> dict[str, Any]:
    pose_cfg = load_json_if_exists(dst_dir / "pose_word_config.json")
    bio_cfg = load_json_if_exists(dst_dir / "bio_config.json")

    pose_frames = pose_cfg.get("clip_frames", pose_cfg.get("window_frames"))
    pose_feature_dim = pose_cfg.get("feature_dim", pose_cfg.get("input_feature_dim"))
    bio_window = bio_cfg.get("window", bio_cfg.get("window_frames"))
    bio_feature_dim = bio_cfg.get("feature_dim", bio_cfg.get("input_feature_dim"))

    return {
        "version": "offline-scaffold-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_dir": str(src_dir),
        "target_dir": str(dst_dir),
        "models": {
            "pose_word_model": {
                "file": "pose_word_model.onnx",
                "T": pose_frames,
                "F": pose_feature_dim,
            },
            "bio_segmenter": {
                "file": "bio_segmenter.onnx",
                "T": bio_window,
                "F": bio_feature_dim,
            },
        },
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy ONNX artifacts from backend/artifacts to frontend/assets/models and build manifest.json."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=DEFAULT_SRC_DIR,
        help=f"Source artifacts directory (default: {DEFAULT_SRC_DIR})",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=DEFAULT_DST_DIR,
        help=f"Destination directory for frontend models (default: {DEFAULT_DST_DIR})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any artifact from export list is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src_dir = args.src_dir.resolve()
    dst_dir = args.dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    exported: list[dict[str, Any]] = []
    missing: list[str] = []

    for spec in ARTIFACTS:
        effective_spec = ArtifactSpec(spec.name, args.strict or spec.required, spec.role)
        try:
            item = copy_artifact(src_dir, dst_dir, effective_spec)
        except FileNotFoundError:
            missing.append(spec.name)
            continue
        if item is None:
            missing.append(spec.name)
            continue
        exported.append(item)

    manifest = build_manifest(src_dir, dst_dir, exported)
    manifest_path = dst_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[export] destination: {dst_dir}")
    print(f"[export] copied: {len(exported)} file(s)")
    for item in exported:
        print(f"  - {item['name']} ({item['size_bytes']} bytes)")
    if missing:
        print(f"[export] missing: {', '.join(missing)}")
    print(f"[export] manifest: {manifest_path}")

    if args.strict and missing:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
