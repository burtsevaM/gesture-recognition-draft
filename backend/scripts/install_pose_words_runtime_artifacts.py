#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT_DIR / "backend" / "artifacts" / "validation" / "pose_words"
DEFAULT_ARTIFACTS_DIR = ROOT_DIR / "backend" / "artifacts" / "runtime" / "active" / "pose_words"
MANIFEST_NAME = "pose_words_active_manifest.json"


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path.resolve())


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install non-dummy pose_words artifacts as active backend runtime artifacts.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("backend/artifacts/validation/pose_words"),
        help="Directory with validated pose_words artifacts to promote into active runtime paths.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("backend/artifacts/runtime/active/pose_words"),
        help="Active runtime artifacts directory used by backend/config.yaml.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing active artifacts after creating a backup.",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=Path("backend/artifacts/runtime/backups/pose_words"),
        help="Root directory for backups when replacing active artifacts.",
    )
    return parser


def resolve_dir(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def artifact_map(base_dir: Path) -> dict[str, Path]:
    return {
        "pose_word_model.onnx": base_dir / "pose_word_model.onnx",
        "pose_word_labels.txt": base_dir / "pose_word_labels.txt",
        "pose_word_config.json": base_dir / "pose_word_config.json",
        "bio_segmenter.onnx": base_dir / "bio_segmenter.onnx",
        "bio_thresholds.json": base_dir / "bio_thresholds.json",
        "bio_config.json": base_dir / "bio_config.json",
    }


def manifest_path(base_dir: Path) -> Path:
    return base_dir / MANIFEST_NAME


def ensure_source_ready(source_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = artifact_map(source_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise FileNotFoundError(f"source artifacts are incomplete: {missing_str}")

    pose_cfg = load_json(paths["pose_word_config.json"])
    bio_cfg = load_json(paths["bio_config.json"])
    bio_thresholds = load_json(paths["bio_thresholds.json"])

    pose_kind = str(pose_cfg.get("artifact_kind") or "").strip().lower()
    bio_kind = str(bio_cfg.get("artifact_kind") or bio_thresholds.get("artifact_kind") or "").strip().lower()
    pose_trained = bool(pose_cfg.get("trained", False))
    bio_trained = bool(bio_cfg.get("trained", bio_thresholds.get("trained", False)))

    if pose_kind in {"", "dummy"}:
        raise ValueError(
            "pose_word_config.json does not describe a non-dummy artifact; "
            f"artifact_kind={pose_kind or '<missing>'}"
        )
    if bio_kind in {"", "dummy"}:
        raise ValueError(
            "bio_config.json/bio_thresholds.json do not describe non-dummy artifacts; "
            f"artifact_kind={bio_kind or '<missing>'}"
        )
    if not pose_trained:
        raise ValueError("pose_word_config.json must have trained=true for active non-dummy installation")
    if not bio_trained:
        raise ValueError("BIO config/thresholds must have trained=true for active non-dummy installation")

    payload = {
        "pose_word": {
            "artifact_kind": pose_kind,
            "dataset_kind": str(pose_cfg.get("dataset_kind") or "unknown"),
            "trained": pose_trained,
            "source_pipeline": str(pose_cfg.get("source_pipeline") or ""),
            "generated_by": str(pose_cfg.get("generated_by") or ""),
        },
        "bio": {
            "artifact_kind": bio_kind,
            "dataset_kind": str(bio_cfg.get("dataset_kind") or bio_thresholds.get("dataset_kind") or "unknown"),
            "trained": bio_trained,
            "source_pipeline": str(bio_cfg.get("source_pipeline") or bio_thresholds.get("source_pipeline") or ""),
            "generated_by": str(bio_cfg.get("generated_by") or bio_thresholds.get("generated_by") or ""),
        },
    }
    return paths, payload


def profile_name(metadata: dict[str, Any]) -> str:
    pose_kind = str(((metadata.get("pose_word") or {}).get("artifact_kind")) or "").strip().lower()
    bio_kind = str(((metadata.get("bio") or {}).get("artifact_kind")) or "").strip().lower()
    if pose_kind == "dummy" or bio_kind == "dummy":
        return "dummy_fallback"
    if pose_kind == "validation" and bio_kind == "validation":
        return "validation_active"
    if pose_kind == "runtime" and bio_kind == "runtime":
        return "runtime_active"
    return "mixed_active"


def backup_existing(target_paths: dict[str, Path], existing_manifest: Path, backup_root: Path) -> Path | None:
    existing = {name: path for name, path in target_paths.items() if path.exists()}
    if existing_manifest.exists():
        existing[MANIFEST_NAME] = existing_manifest
    if not existing:
        return None

    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name, path in existing.items():
        shutil.copy2(path, backup_dir / name)
    return backup_dir


def write_manifest(
    *,
    target_dir: Path,
    source_dir: Path,
    source_paths: dict[str, Path],
    target_paths: dict[str, Path],
    metadata: dict[str, Any],
    backup_dir: Path | None,
) -> Path:
    pose_meta = dict(metadata["pose_word"])
    bio_meta = dict(metadata["bio"])
    manifest = {
        "profile": profile_name(metadata),
        "installed_at_unix": float(time.time()),
        "installed_by": "backend/scripts/install_pose_words_runtime_artifacts.py",
        "source_dir": repo_rel(source_dir),
        "target_dir": repo_rel(target_dir),
        "backup_dir": repo_rel(backup_dir) if backup_dir is not None else "",
        "pose_word": {
            **pose_meta,
            "active_model_path": repo_rel(target_paths["pose_word_model.onnx"]),
            "active_labels_path": repo_rel(target_paths["pose_word_labels.txt"]),
            "active_config_path": repo_rel(target_paths["pose_word_config.json"]),
            "installed_from_model_path": repo_rel(source_paths["pose_word_model.onnx"]),
            "installed_from_labels_path": repo_rel(source_paths["pose_word_labels.txt"]),
            "installed_from_config_path": repo_rel(source_paths["pose_word_config.json"]),
            "sha256_16": sha256_short(target_paths["pose_word_model.onnx"]),
        },
        "bio": {
            **bio_meta,
            "active_model_path": repo_rel(target_paths["bio_segmenter.onnx"]),
            "active_thresholds_path": repo_rel(target_paths["bio_thresholds.json"]),
            "active_config_path": repo_rel(target_paths["bio_config.json"]),
            "installed_from_model_path": repo_rel(source_paths["bio_segmenter.onnx"]),
            "installed_from_thresholds_path": repo_rel(source_paths["bio_thresholds.json"]),
            "installed_from_config_path": repo_rel(source_paths["bio_config.json"]),
            "sha256_16": sha256_short(target_paths["bio_segmenter.onnx"]),
        },
    }
    path = manifest_path(target_dir)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def copy_artifacts(source_paths: dict[str, Path], target_paths: dict[str, Path]) -> None:
    for name, source_path in source_paths.items():
        target_path = target_paths[name]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def main() -> int:
    args = build_parser().parse_args()
    source_dir = resolve_dir(args.source_dir)
    artifacts_dir = resolve_dir(args.artifacts_dir)
    backup_root = resolve_dir(args.backup_root)

    source_paths, metadata = ensure_source_ready(source_dir)
    target_paths = artifact_map(artifacts_dir)
    existing_manifest = manifest_path(artifacts_dir)

    same_dir = source_dir == artifacts_dir
    has_existing_targets = any(path.exists() for path in target_paths.values())
    if has_existing_targets and not args.force and not same_dir:
        raise RuntimeError(
            "active pose_words artifacts already exist; rerun with --force to back them up and replace them"
        )

    backup_dir: Path | None = None
    if not same_dir:
        backup_dir = backup_existing(target_paths, existing_manifest, backup_root)
        copy_artifacts(source_paths, target_paths)
    else:
        backup_dir = backup_existing(target_paths, existing_manifest, backup_root) if args.force else None

    manifest = write_manifest(
        target_dir=artifacts_dir,
        source_dir=source_dir,
        source_paths=source_paths,
        target_paths=target_paths if not same_dir else source_paths,
        metadata=metadata,
        backup_dir=backup_dir,
    )

    print("[install_pose_words_runtime_artifacts] done")
    print(f"  source_dir={repo_rel(source_dir)}")
    print(f"  artifacts_dir={repo_rel(artifacts_dir)}")
    if backup_dir is not None:
        print(f"  backup_dir={repo_rel(backup_dir)}")
    print(f"  active_profile={profile_name(metadata)}")
    print(f"  manifest={repo_rel(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
