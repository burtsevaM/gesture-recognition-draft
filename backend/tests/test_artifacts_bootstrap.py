from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, runtime
from app.pose_words.model_onnx_pose import PoseWordOnnxModel
from app.segmentation.model_onnx import BioSegmenterOnnxModel


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = REPO_ROOT / "backend" / "scripts" / "install_pose_words_runtime_artifacts.py"


def _run_bootstrap(artifacts_dir: Path, *, force: bool = True) -> None:
    cmd = [
        sys.executable,
        "-m",
        "backend.scripts.bootstrap_pose_words_artifacts",
        "--artifacts-dir",
        str(artifacts_dir),
        "--feature-dim",
        "159",
        "--clip-frames",
        "32",
        "--window-size",
        "64",
    ]
    if force:
        cmd.append("--force")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _expected_paths(artifacts_dir: Path) -> dict[str, Path]:
    return {
        "pose_word_model.onnx": artifacts_dir / "pose_word_model.onnx",
        "pose_word_labels.txt": artifacts_dir / "pose_word_labels.txt",
        "pose_word_config.json": artifacts_dir / "pose_word_config.json",
        "bio_segmenter.onnx": artifacts_dir / "bio_segmenter.onnx",
        "bio_thresholds.json": artifacts_dir / "bio_thresholds.json",
        "bio_config.json": artifacts_dir / "bio_config.json",
    }


def _run_install(source_dir: Path, artifacts_dir: Path, *, force: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(INSTALL_SCRIPT),
        "--source-dir",
        str(source_dir),
        "--artifacts-dir",
        str(artifacts_dir),
    ]
    if force:
        cmd.append("--force")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), check=check, capture_output=True, text=True)


def _rewrite_as_validation_metadata(artifacts_dir: Path) -> None:
    pose_cfg_path = artifacts_dir / "pose_word_config.json"
    bio_cfg_path = artifacts_dir / "bio_config.json"
    bio_thresholds_path = artifacts_dir / "bio_thresholds.json"

    pose_cfg = json.loads(pose_cfg_path.read_text(encoding="utf-8"))
    pose_cfg.update(
        {
            "artifact_kind": "validation",
            "dataset_kind": "synthetic_fixture",
            "trained": True,
            "source_pipeline": "run_pose_words_validation",
            "generated_by": "backend/train/export_pose_word_onnx.py",
        }
    )
    pose_cfg_path.write_text(json.dumps(pose_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    bio_cfg = json.loads(bio_cfg_path.read_text(encoding="utf-8"))
    bio_cfg.update(
        {
            "artifact_kind": "validation",
            "dataset_kind": "synthetic_fixture",
            "trained": True,
            "source_pipeline": "run_pose_words_validation",
            "generated_by": "backend/train/export_bio_segmenter_onnx.py",
        }
    )
    bio_cfg_path.write_text(json.dumps(bio_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    bio_thresholds = json.loads(bio_thresholds_path.read_text(encoding="utf-8"))
    bio_thresholds.update(
        {
            "artifact_kind": "validation",
            "dataset_kind": "synthetic_fixture",
            "trained": True,
            "source_pipeline": "run_pose_words_validation",
            "generated_by": "backend/train/train_pose_bio_segmenter.py",
        }
    )
    bio_thresholds_path.write_text(json.dumps(bio_thresholds, ensure_ascii=False, indent=2), encoding="utf-8")


def test_bootstrap_creates_artifacts_and_models_are_loadable(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _run_bootstrap(artifacts_dir, force=True)

    paths = _expected_paths(artifacts_dir)
    for path in paths.values():
        assert path.exists(), f"missing bootstrap artifact: {path}"

    pose_model = PoseWordOnnxModel(
        model_path=paths["pose_word_model.onnx"],
        labels_path=paths["pose_word_labels.txt"],
        config_path=paths["pose_word_config.json"],
        ort_num_threads=1,
    )
    clip = np.random.randn(32, 159).astype(np.float32)
    probs, _ = pose_model.infer_probs(clip)
    assert probs.ndim == 1
    assert probs.shape[0] > 0

    bio_model = BioSegmenterOnnxModel(
        model_path=paths["bio_segmenter.onnx"],
        config_path=paths["bio_config.json"],
        ort_num_threads=1,
    )
    sign_probs, phrase_probs, _ = bio_model.infer(np.random.randn(20, 159).astype(np.float32))
    assert sign_probs.shape == (20, 3)
    assert phrase_probs.shape == (20, 3)


def test_pose_words_health_reflects_missing_and_ready_states(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    paths = _expected_paths(artifacts_dir)

    old_cfg = copy.deepcopy(runtime.config)
    old_errors = dict(runtime.errors)
    old_pose_model = runtime._pose_word_model
    old_bio_model = runtime._bio_segmenter_model
    old_pose_extractor = runtime._pose_extractor

    try:
        runtime.config.recognition_mode = "pose_words"
        runtime.config.segmentation_enabled = True
        runtime.config.pose_word_model_path = str(paths["pose_word_model.onnx"])
        runtime.config.pose_word_labels_path = str(paths["pose_word_labels.txt"])
        runtime.config.pose_word_config_path = str(paths["pose_word_config.json"])
        runtime.config.segmentation_model_path = str(paths["bio_segmenter.onnx"])
        runtime.config.segmentation_config_path = str(paths["bio_config.json"])
        runtime.config.segmentation_thresholds_path = str(paths["bio_thresholds.json"])

        runtime._pose_word_model = None
        runtime._bio_segmenter_model = None
        runtime._pose_extractor = object()  # bypass MediaPipe init during health check
        runtime.errors.clear()
        runtime._logged_missing_paths.clear()

        client = TestClient(app)
        health_missing = client.get("/health").json()
        assert health_missing["ok"] is True
        assert health_missing["pose_words_ready"] is False
        assert "pose_word_model.onnx" in health_missing["missing_artifacts"]
        assert "bio_segmenter.onnx" in health_missing["missing_artifacts"]

        _run_bootstrap(artifacts_dir, force=True)
        runtime._pose_word_model = None
        runtime._bio_segmenter_model = None
        runtime.errors.clear()
        runtime._logged_missing_paths.clear()

        health_ready = client.get("/health").json()
        assert health_ready["ok"] is True
        assert health_ready["pose_words_ready"] is True
        assert health_ready["missing_artifacts"] == []
        assert health_ready["active_artifact_profile"] == "dummy_fallback"
        assert health_ready["pose_words_artifact_kind"] == "dummy"
        assert health_ready["bio_artifact_kind"] == "dummy"
        assert health_ready["pose_words_non_dummy_active"] is False
        assert health_ready["pose_words_validated_runtime_ready"] is False
        assert health_ready["active_artifacts_manifest_present"] is True
    finally:
        runtime.config = old_cfg
        runtime.errors.clear()
        runtime.errors.update(old_errors)
        runtime._pose_word_model = old_pose_model
        runtime._bio_segmenter_model = old_bio_model
        runtime._pose_extractor = old_pose_extractor
        runtime._logged_missing_paths.clear()


def test_install_runtime_artifacts_rejects_dummy_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "dummy_source"
    active_dir = tmp_path / "active_runtime"
    _run_bootstrap(source_dir, force=True)

    proc = _run_install(source_dir, active_dir, check=False)
    assert proc.returncode != 0
    assert "non-dummy" in (proc.stderr or "")


def test_health_reports_validation_active_runtime_after_install(tmp_path: Path) -> None:
    source_dir = tmp_path / "validation_source"
    active_dir = tmp_path / "active_runtime"
    _run_bootstrap(source_dir, force=True)
    _rewrite_as_validation_metadata(source_dir)
    _run_install(source_dir, active_dir, check=True)

    paths = _expected_paths(active_dir)
    manifest_path = active_dir / "pose_words_active_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["profile"] == "validation_active"

    old_cfg = copy.deepcopy(runtime.config)
    old_errors = dict(runtime.errors)
    old_pose_model = runtime._pose_word_model
    old_bio_model = runtime._bio_segmenter_model
    old_pose_extractor = runtime._pose_extractor

    try:
        runtime.config.recognition_mode = "pose_words"
        runtime.config.segmentation_enabled = True
        runtime.config.pose_word_model_path = str(paths["pose_word_model.onnx"])
        runtime.config.pose_word_labels_path = str(paths["pose_word_labels.txt"])
        runtime.config.pose_word_config_path = str(paths["pose_word_config.json"])
        runtime.config.segmentation_model_path = str(paths["bio_segmenter.onnx"])
        runtime.config.segmentation_config_path = str(paths["bio_config.json"])
        runtime.config.segmentation_thresholds_path = str(paths["bio_thresholds.json"])

        runtime._pose_word_model = None
        runtime._bio_segmenter_model = None
        runtime._pose_extractor = object()
        runtime.errors.clear()
        runtime._logged_missing_paths.clear()

        client = TestClient(app)
        health = client.get("/health").json()
        assert health["ok"] is True
        assert health["pose_words_ready"] is True
        assert health["pose_words_validated_runtime_ready"] is True
        assert health["pose_words_non_dummy_active"] is True
        assert health["active_artifact_profile"] == "validation_active"
        assert health["pose_words_artifact_kind"] == "validation"
        assert health["bio_artifact_kind"] == "validation"
        assert health["missing_artifacts"] == []
        assert health["active_artifacts_manifest_present"] is True
        assert health["active_artifacts_manifest_path"].endswith("pose_words_active_manifest.json")
    finally:
        runtime.config = old_cfg
        runtime.errors.clear()
        runtime.errors.update(old_errors)
        runtime._pose_word_model = old_pose_model
        runtime._bio_segmenter_model = old_bio_model
        runtime._pose_extractor = old_pose_extractor
        runtime._logged_missing_paths.clear()
