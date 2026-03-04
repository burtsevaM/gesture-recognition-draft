from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.main import app, runtime
from app.pose_words.model_onnx_pose import PoseWordOnnxModel
from app.segmentation.model_onnx import BioSegmenterOnnxModel


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    finally:
        runtime.config = old_cfg
        runtime.errors.clear()
        runtime.errors.update(old_errors)
        runtime._pose_word_model = old_pose_model
        runtime._bio_segmenter_model = old_bio_model
        runtime._pose_extractor = old_pose_extractor
        runtime._logged_missing_paths.clear()
