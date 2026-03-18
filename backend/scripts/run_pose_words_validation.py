#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
CONFIG_PATH = BACKEND_DIR / "config.yaml"
DATA_ROOT = BACKEND_DIR / "data" / "pose_words_validation" / "generated"
ARTIFACTS_ROOT = BACKEND_DIR / "artifacts" / "validation" / "pose_words"
POSE_CLIPS_DIR = DATA_ROOT / "clips"
POSE_INDEX_PATH = DATA_ROOT / "pose_word_index.jsonl"
BIO_SOURCE_INDEX_PATH = DATA_ROOT / "bio_source_index.jsonl"
LABELS_PATH = DATA_ROOT / "labels.txt"
LEGACY_LABELS_PATH = DATA_ROOT / "pose_word_labels.txt"
DATASET_STATS_PATH = DATA_ROOT / "dataset_stats.json"
BIO_DATA_DIR = DATA_ROOT / "bio_continuous"
REPORT_PATH = ARTIFACTS_ROOT / "technical_validation_report.json"
SMOKE_LOG_PATH = ARTIFACTS_ROOT / "smoke_pose_words_validation.jsonl"
SERVER_LOG_PATH = ARTIFACTS_ROOT / "backend_validation.log"


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path.resolve())


def sanitize_text(value: str) -> str:
    root_text = str(ROOT_DIR.resolve())
    return str(value).replace(root_text, ".")


def display_token(token: str) -> str:
    try:
        root_text = str(ROOT_DIR)
        if token.startswith(root_text):
            return token.replace(root_text + os.sep, "", 1)
        path = Path(token)
        if path.is_absolute() and path.exists():
            return repo_rel(path)
    except Exception:
        return token
    return token


def display_cmd(cmd: list[str]) -> str:
    return " ".join(display_token(str(part)) for part in cmd)


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible technical validation for pose_words.")
    parser.add_argument("--python-bin", default="", help="Python interpreter to use for subprocess steps.")
    parser.add_argument("--device", default="cpu", help="Training device for validation run.")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--smoke-duration-sec", type=float, default=5.0)
    parser.add_argument("--smoke-fps", type=float, default=8.0)
    parser.add_argument("--clean", type=parse_bool, default=True, help="Clean generated validation dirs before run.")
    return parser


def choose_python(args: argparse.Namespace) -> str:
    custom = str(args.python_bin).strip()
    if custom:
        return custom
    candidate = ROOT_DIR / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def run_step(
    cmd: list[str],
    *,
    cwd: Path = ROOT_DIR,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    printable = display_cmd(cmd)
    print(f"[validation] run: {printable}")
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {printable}\n"
            f"stdout:\n{sanitize_text(proc.stdout)}\n"
            f"stderr:\n{sanitize_text(proc.stderr)}"
        )
    return {
        "cmd": printable,
        "cwd": repo_rel(cwd),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(round(elapsed, 4)),
        "stdout": sanitize_text(proc.stdout),
        "stderr": sanitize_text(proc.stderr),
    }


def sha256_short(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def prepare_dirs(clean: bool) -> None:
    if clean and DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    if clean and ARTIFACTS_ROOT.exists():
        shutil.rmtree(ARTIFACTS_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    POSE_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)


def build_clip_features(label: str, sample_idx: int, *, feature_dim: int, length: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    features = np.zeros((length, feature_dim), dtype=np.float32)
    phase = float(sample_idx + 1)
    seed_offset = 0 if label == "_no_event" else 100 if label == "привет" else 200
    rng = np.random.default_rng(1000 + sample_idx + seed_offset)

    if label == "_no_event":
        features += 0.015 * np.sin((2.0 * np.pi * t * (1.0 + 0.1 * phase)))[:, None]
        features[:, 120:] += 0.01
    elif label == "привет":
        features[:, :40] = 0.90 + 0.08 * np.sin((2.0 * np.pi * t) + phase / 10.0)[:, None]
        features[:, 40:80] = 0.35 + 0.05 * np.cos((3.0 * np.pi * t) + phase / 8.0)[:, None]
        features[:, 120:140] = t[:, None]
        features[:, 140:] = 0.2
    elif label == "пока":
        features[:, 80:120] = -0.85 + 0.10 * np.cos((3.5 * np.pi * t) + phase / 7.0)[:, None]
        features[:, 20:40] = 0.28 + 0.02 * np.sin((2.5 * np.pi * t) + phase / 5.0)[:, None]
        features[:, 140:] = (1.0 - t)[:, None]
        features[:, :10] = -0.15
    else:
        raise ValueError(f"unsupported label: {label}")

    features += rng.normal(loc=0.0, scale=0.008, size=features.shape).astype(np.float32)
    return features.astype(np.float32)


def prepare_validation_data() -> dict[str, Any]:
    labels = ["_no_event", "привет", "пока"]
    feature_dim = 159
    rows_classifier: list[dict[str, Any]] = []
    rows_bio_source: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"labels": labels, "feature_dim": feature_dim, "splits": {}}

    split_counts = {"train": 4, "val": 2, "test": 2}
    split_to_signers = {
        "train": ["signer_a", "signer_b", "signer_c", "signer_d"],
        "val": ["signer_eval_a", "signer_eval_b"],
        "test": ["signer_holdout_a", "signer_holdout_b"],
    }

    for split_name, count in split_counts.items():
        summary["splits"][split_name] = {}
        for label_idx, label in enumerate(labels):
            summary["splits"][split_name][label] = 0
            for i in range(count):
                sample_id = f"{label}_{split_name}_{i:02d}"
                length = 24 + ((i + label_idx) % 6)
                features = build_clip_features(label, sample_idx=(label_idx * 10 + i), feature_dim=feature_dim, length=length)
                clip_path = POSE_CLIPS_DIR / f"{sample_id}.npz"
                meta = {
                    "sample_id": sample_id,
                    "label": label,
                    "split": split_name,
                    "signer_id": split_to_signers[split_name][i % len(split_to_signers[split_name])],
                    "fps": 25,
                    "dataset_kind": "synthetic_fixture",
                    "source_pipeline": "run_pose_words_validation",
                }
                np.savez_compressed(
                    clip_path,
                    features=features,
                    meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
                )
                row = {
                    "clip_id": sample_id,
                    "path": str(Path("clips") / clip_path.name),
                    "label": label,
                    "split": split_name,
                    "signer_id": meta["signer_id"],
                    "fps": 25,
                }
                rows_classifier.append(row)
                if label != "_no_event":
                    rows_bio_source.append(dict(row))
                summary["splits"][split_name][label] += 1

    labels_text = "\n".join(labels) + "\n"
    LABELS_PATH.write_text(labels_text, encoding="utf-8")
    LEGACY_LABELS_PATH.write_text(labels_text, encoding="utf-8")
    DATASET_STATS_PATH.write_text(
        json.dumps(
            {
                "use_shoulder_norm": False,
                "use_hands_3d_norm": False,
                "dataset_kind": "synthetic_fixture",
                "source_pipeline": "run_pose_words_validation",
                "description": "Deterministic tiny validation fixtures for pose_words technical validation.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with POSE_INDEX_PATH.open("w", encoding="utf-8") as f:
        for row in rows_classifier:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with BIO_SOURCE_INDEX_PATH.open("w", encoding="utf-8") as f:
        for row in rows_bio_source:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "labels_path": repo_rel(LABELS_PATH),
        "pose_word_index": repo_rel(POSE_INDEX_PATH),
        "bio_source_index": repo_rel(BIO_SOURCE_INDEX_PATH),
        "dataset_stats": repo_rel(DATASET_STATS_PATH),
        "clips_dir": repo_rel(POSE_CLIPS_DIR),
        "summary": summary,
    }


def artifact_summary(path: Path) -> dict[str, Any]:
    return {
        "path": repo_rel(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256_16": sha256_short(path) if path.exists() else "",
    }


def run_runtime_replay(artifacts_dir: Path) -> dict[str, Any]:
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))

    from app.pose_words import PoseWordOnnxModel, resample_to_fixed_T
    from app.segmentation import BioSegmenterOnnxModel, StreamingBioSegmenter, load_bio_thresholds
    from app.words.decoder import WordDecisionDecoder, WordThresholds

    pose_model = PoseWordOnnxModel(
        model_path=artifacts_dir / "pose_word_model.onnx",
        labels_path=artifacts_dir / "pose_word_labels.txt",
        config_path=artifacts_dir / "pose_word_config.json",
        ort_num_threads=1,
    )
    bio_model = BioSegmenterOnnxModel(
        model_path=artifacts_dir / "bio_segmenter.onnx",
        config_path=artifacts_dir / "bio_config.json",
        ort_num_threads=1,
    )
    thresholds = load_bio_thresholds(artifacts_dir / "bio_thresholds.json")
    segmenter = StreamingBioSegmenter(
        model=bio_model,
        window=32,
        step=1,
        min_len=1,
        max_len=80,
        merge_gap=0,
        cool_off_frames=0,
        sign_th_b=thresholds.sign_th_b,
        sign_th_o=thresholds.sign_th_o,
        phrase_th_b=thresholds.phrase_th_b,
        phrase_th_o=thresholds.phrase_th_o,
        max_buffer=128,
    )
    decoder = WordDecisionDecoder(
        ema_alpha=1.0,
        thresholds=WordThresholds(
            th_no_event=0.55,
            th_unknown=0.35,
            th_margin=0.01,
        ),
        hold_frames=1,
        cooldown_frames=0,
        dedup_same_word=False,
    )
    no_event_index = pose_model.find_no_event_index("_no_event")

    with np.load(POSE_CLIPS_DIR / "_no_event_train_00.npz", allow_pickle=True) as data:
        bg_clip = np.asarray(data["features"], dtype=np.float32)
    bg_frame = bg_clip[min(3, len(bg_clip) - 1)]

    per_sample: list[dict[str, Any]] = []
    total_segments = 0
    committed_words: list[str] = []
    expected_total_words = 0
    exact_match_samples = 0

    for sample_path in sorted((BIO_DATA_DIR / "test").glob("*.npz")):
        with np.load(sample_path, allow_pickle=True) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            raw_meta = data["meta"]
        if isinstance(raw_meta, np.ndarray):
            raw_meta = raw_meta.reshape(-1)[0]
        meta = json.loads(str(raw_meta))
        expected_words = [str(item.get("label", "")) for item in meta.get("words", [])]
        expected_total_words += len(expected_words)

        emitted_for_sample: list[str] = []
        stream = np.concatenate([features, np.repeat(bg_frame[None, :], repeats=8, axis=0)], axis=0)
        for frame in stream:
            result = segmenter.update(frame)
            for seg in result.sign_segments:
                total_segments += 1
                clip = segmenter.get_feature_span(seg.start, seg.end)
                if clip is None or clip.shape[0] == 0:
                    continue
                clip = resample_to_fixed_T(clip, T=32, method="linear")
                probs, _ = pose_model.infer_probs(clip)
                decoded = decoder.update(
                    probs=probs,
                    labels=pose_model.labels,
                    topk=3,
                    no_event_index=no_event_index,
                )
                if decoded.committed and decoded.committed_word is not None:
                    committed_words.append(decoded.committed_word)
                    emitted_for_sample.append(decoded.committed_word)

        per_sample.append(
            {
                "sample": sample_path.name,
                "expected_words": expected_words,
                "committed_words": emitted_for_sample,
                "exact_match": emitted_for_sample == expected_words,
            }
        )
        if emitted_for_sample == expected_words:
            exact_match_samples += 1

    return {
        "pose_labels": pose_model.labels,
        "segments_detected": int(total_segments),
        "committed_words": committed_words,
        "expected_total_words": int(expected_total_words),
        "committed_total_words": int(len(committed_words)),
        "extra_commits": int(max(0, len(committed_words) - expected_total_words)),
        "exact_match_samples": int(exact_match_samples),
        "sample_count": int(len(per_sample)),
        "samples": per_sample,
        "ok": bool(total_segments > 0 and committed_words),
        "limitations": [
            "Replay runs direct ONNX/runtime components on generated pose features, not camera JPEG frames.",
            "Current validation settings favor recall and may over-segment; mismatch versus expected transcript is tracked but not treated as failure for this technical validation.",
        ],
    }


def summarize_smoke_log(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return {
            "log_path": repo_rel(path),
            "log_rows": 0,
            "positive_segment_frames": 0,
            "committed_frames": 0,
            "all_statuses": [],
        }

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)

    positive_segment_frames = 0
    committed_frames = 0
    statuses: set[str] = set()
    for row in rows:
        statuses.add(str(row.get("status", "")))
        sign_segments = int(row.get("sign_segments", 0) or 0)
        phrase_segments = int(row.get("phrase_segments", 0) or 0)
        if sign_segments > 0 or phrase_segments > 0:
            positive_segment_frames += 1
        if bool(row.get("committed", False)):
            committed_frames += 1

    return {
        "log_path": repo_rel(path),
        "log_rows": int(len(rows)),
        "positive_segment_frames": int(positive_segment_frames),
        "committed_frames": int(committed_frames),
        "all_statuses": sorted(status for status in statuses if status),
    }


def update_validation_config(artifacts_dir: Path) -> str:
    original_text = CONFIG_PATH.read_text(encoding="utf-8")
    current = yaml.safe_load(original_text) or {}
    if not isinstance(current, dict):
        raise ValueError(f"config must be a mapping: {CONFIG_PATH}")

    cfg = copy.deepcopy(current)
    cfg["recognition_mode"] = "pose_words"
    seg = cfg.setdefault("segmentation", {})
    seg["enabled"] = True
    seg["model_path"] = repo_rel(artifacts_dir / "bio_segmenter.onnx")
    seg["config_path"] = repo_rel(artifacts_dir / "bio_config.json")
    seg["thresholds_path"] = repo_rel(artifacts_dir / "bio_thresholds.json")
    seg["window"] = 32
    seg["step"] = 1
    seg["min_len"] = 1
    seg["max_len"] = 80
    seg["merge_gap"] = 0
    seg["cool_off_frames"] = 0
    seg["max_buffer"] = 128
    seg["ort_num_threads"] = 1

    pose_model = cfg.setdefault("pose_word_model", {})
    pose_model["path"] = repo_rel(artifacts_dir / "pose_word_model.onnx")
    pose_model["labels_path"] = repo_rel(artifacts_dir / "pose_word_labels.txt")
    pose_model["config_path"] = repo_rel(artifacts_dir / "pose_word_config.json")
    pose_model["clip_frames"] = 32
    pose_model["topk"] = 3
    pose_model["ort_num_threads"] = 1

    pose_thr = cfg.setdefault("pose_word_thresholds", {})
    pose_thr["no_event_label"] = "_no_event"
    pose_thr["th_no_event"] = 0.55
    pose_thr["th_unknown"] = 0.35
    pose_thr["th_margin"] = 0.01

    pose_commit = cfg.setdefault("pose_word_commit_logic", {})
    pose_commit["ema_alpha"] = 1.0
    pose_commit["hold_segments"] = 1
    pose_commit["cooldown_segments"] = 0
    pose_commit["dedup_same_word"] = False

    CONFIG_PATH.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return original_text


def wait_for_health(base_url: str, timeout_sec: float) -> dict[str, Any]:
    deadline = time.time() + float(timeout_sec)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=2.0) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except (URLError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.4)
    raise RuntimeError(f"backend health did not become available: {last_error}")


def run_backend_smoke(args: argparse.Namespace, artifacts_dir: Path, py_bin: str) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{int(args.port)}"
    original_config = update_validation_config(artifacts_dir)
    server_proc: subprocess.Popen[str] | None = None
    server_log = SERVER_LOG_PATH.open("w", encoding="utf-8")
    try:
        server_proc = subprocess.Popen(
            [
                py_bin,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(int(args.port)),
            ],
            cwd=str(BACKEND_DIR),
            env=dict(os.environ),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        health = wait_for_health(base_url, timeout_sec=20.0)
        smoke = run_step(
            [
                py_bin,
                "backend/scripts/smoke_pose_words.py",
                "--base-url",
                base_url,
                "--duration-sec",
                str(float(args.smoke_duration_sec)),
                "--fps",
                str(float(args.smoke_fps)),
                "--require-pose-words",
                "true",
                "--require-segmentation",
                "true",
                "--log-path",
                repo_rel(SMOKE_LOG_PATH),
            ]
        )
        smoke_log = summarize_smoke_log(SMOKE_LOG_PATH)
        return {
            "base_url": base_url,
            "health": health,
            "smoke": {
                "cmd": smoke["cmd"],
                "elapsed_sec": smoke["elapsed_sec"],
                "stdout_tail": smoke["stdout"].splitlines()[-12:],
                "stderr_tail": smoke["stderr"].splitlines()[-12:],
                **smoke_log,
                "positive_runtime_result": bool(smoke_log["positive_segment_frames"] > 0 or smoke_log["committed_frames"] > 0),
            },
        }
    finally:
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=5.0)
        server_log.close()


def main() -> int:
    args = build_parser().parse_args()
    py_bin = choose_python(args)

    prepare_dirs(clean=bool(args.clean))
    fixtures = prepare_validation_data()
    commands: list[dict[str, Any]] = []

    commands.append(
        run_step(
            [
                py_bin,
                "backend/scripts/build_bio_continuous_dataset.py",
                "--pose_index",
                repo_rel(BIO_SOURCE_INDEX_PATH),
                "--out_dir",
                repo_rel(BIO_DATA_DIR),
                "--min_words",
                "2",
                "--max_words",
                "3",
                "--pause_min",
                "1",
                "--pause_max",
                "2",
                "--phrase_min_words",
                "1",
                "--phrase_max_words",
                "2",
                "--seed",
                "17",
                "--splits_by_signer",
                "false",
                "--resample_fps",
                "25",
                "--use_shoulder_norm",
                "false",
                "--use_hands_3d_norm",
                "false",
                "--pause_noise_std",
                "0.0",
                "--train_samples",
                "8",
                "--val_samples",
                "2",
                "--test_samples",
                "2",
            ]
        )
    )
    commands.append(
        run_step(
            [
                py_bin,
                "backend/train/train_pose_word_model.py",
                "--dataset-index",
                repo_rel(POSE_INDEX_PATH),
                "--output-dir",
                repo_rel(ARTIFACTS_ROOT / "pose_word_training"),
                "--epochs",
                "6",
                "--batch-size",
                "4",
                "--device",
                str(args.device),
                "--conv-channels",
                "32",
                "--gru-hidden",
                "24",
                "--gru-layers",
                "1",
                "--dropout",
                "0.1",
                "--clip-frames",
                "32",
                "--artifact-kind",
                "validation",
                "--dataset-kind",
                "synthetic_fixture",
                "--source-pipeline",
                "run_pose_words_validation",
            ]
        )
    )
    commands.append(
        run_step(
            [
                py_bin,
                "backend/train/export_pose_word_onnx.py",
                "--checkpoint",
                repo_rel(ARTIFACTS_ROOT / "pose_word_training" / "pose_word_best.pt"),
                "--output",
                repo_rel(ARTIFACTS_ROOT / "pose_word_model.onnx"),
                "--labels-out",
                repo_rel(ARTIFACTS_ROOT / "pose_word_labels.txt"),
                "--config-out",
                repo_rel(ARTIFACTS_ROOT / "pose_word_config.json"),
                "--verify",
                "true",
                "--artifact-kind",
                "validation",
                "--dataset-kind",
                "synthetic_fixture",
                "--source-pipeline",
                "run_pose_words_validation",
            ]
        )
    )
    commands.append(
        run_step(
            [
                py_bin,
                "backend/train/train_pose_bio_segmenter.py",
                "--data-dir",
                repo_rel(BIO_DATA_DIR),
                "--save-dir",
                repo_rel(ARTIFACTS_ROOT / "bio_training"),
                "--thresholds-out",
                repo_rel(ARTIFACTS_ROOT / "bio_thresholds.json"),
                "--epochs",
                "12",
                "--batch-size",
                "2",
                "--hidden-size",
                "16",
                "--num-layers",
                "1",
                "--dropout",
                "0.05",
                "--bidirectional",
                "true",
                "--alpha-phrase",
                "1.0",
                "--threshold-grid",
                "0.3,0.4,0.5,0.6,0.7",
                "--boundary-tolerance",
                "0",
                "--segment-iou-threshold",
                "0.5",
                "--min-segment-len",
                "1",
                "--use-shoulder-norm",
                "false",
                "--use-hands-3d-norm",
                "false",
                "--device",
                str(args.device),
                "--artifact-kind",
                "validation",
                "--dataset-kind",
                "synthetic_fixture",
                "--source-pipeline",
                "run_pose_words_validation",
            ]
        )
    )
    commands.append(
        run_step(
            [
                py_bin,
                "backend/train/export_bio_segmenter_onnx.py",
                "--checkpoint",
                repo_rel(ARTIFACTS_ROOT / "bio_training" / "best_model.pt"),
                "--onnx-out",
                repo_rel(ARTIFACTS_ROOT / "bio_segmenter.onnx"),
                "--thresholds-in",
                repo_rel(ARTIFACTS_ROOT / "bio_thresholds.json"),
                "--thresholds-out",
                repo_rel(ARTIFACTS_ROOT / "bio_thresholds.json"),
                "--bio-config-out",
                repo_rel(ARTIFACTS_ROOT / "bio_config.json"),
                "--window-size",
                "64",
                "--dynamic-time",
                "true",
                "--artifact-kind",
                "validation",
                "--dataset-kind",
                "synthetic_fixture",
                "--source-pipeline",
                "run_pose_words_validation",
            ]
        )
    )

    pose_training = load_json(ARTIFACTS_ROOT / "pose_word_training" / "training_config.json")
    bio_training = load_json(ARTIFACTS_ROOT / "bio_training" / "train_summary.json")
    pose_config = load_json(ARTIFACTS_ROOT / "pose_word_config.json")
    bio_config = load_json(ARTIFACTS_ROOT / "bio_config.json")
    bio_thresholds = load_json(ARTIFACTS_ROOT / "bio_thresholds.json")

    runtime_replay = run_runtime_replay(ARTIFACTS_ROOT)
    backend_smoke = run_backend_smoke(args, ARTIFACTS_ROOT, py_bin)

    report = {
        "status": "ok",
        "generated_at_unix": float(time.time()),
        "validation_scope": "technical_validation_only",
        "artifacts_dir": repo_rel(ARTIFACTS_ROOT),
        "data_root": repo_rel(DATA_ROOT),
        "commands": commands,
        "fixtures": fixtures,
        "pose_word_training": {
            "best_val_top1": float(pose_training.get("best_val_top1", 0.0)),
            "val_top1": float((pose_training.get("val") or {}).get("top1", 0.0)),
            "test_top1": float(((pose_training.get("test") or {}).get("top1", 0.0)) if pose_training.get("test") else 0.0),
        },
        "bio_training": {
            "best_epoch": int(bio_training.get("best_epoch", 0)),
            "val_sign_boundary_f1": float((((bio_training.get("best") or {}).get("val_sign") or {}).get("boundary_f1", 0.0))),
            "val_phrase_boundary_f1": float((((bio_training.get("best") or {}).get("val_phrase") or {}).get("boundary_f1", 0.0))),
            "test_sign_boundary_f1": float((((bio_training.get("test") or {}).get("sign") or {}).get("boundary_f1", 0.0)) if bio_training.get("test") else 0.0),
            "test_phrase_boundary_f1": float((((bio_training.get("test") or {}).get("phrase") or {}).get("boundary_f1", 0.0)) if bio_training.get("test") else 0.0),
        },
        "artifacts": {
            "pose_word_model": artifact_summary(ARTIFACTS_ROOT / "pose_word_model.onnx"),
            "pose_word_labels": artifact_summary(ARTIFACTS_ROOT / "pose_word_labels.txt"),
            "pose_word_config": artifact_summary(ARTIFACTS_ROOT / "pose_word_config.json"),
            "bio_segmenter": artifact_summary(ARTIFACTS_ROOT / "bio_segmenter.onnx"),
            "bio_thresholds": artifact_summary(ARTIFACTS_ROOT / "bio_thresholds.json"),
            "bio_config": artifact_summary(ARTIFACTS_ROOT / "bio_config.json"),
        },
        "artifact_metadata": {
            "pose_word_config": {
                "artifact_kind": pose_config.get("artifact_kind"),
                "dataset_kind": pose_config.get("dataset_kind"),
                "trained": pose_config.get("trained"),
                "source_pipeline": pose_config.get("source_pipeline"),
                "generated_by": pose_config.get("generated_by"),
            },
            "bio_config": {
                "artifact_kind": bio_config.get("artifact_kind"),
                "dataset_kind": bio_config.get("dataset_kind"),
                "trained": bio_config.get("trained"),
                "source_pipeline": bio_config.get("source_pipeline"),
                "generated_by": bio_config.get("generated_by"),
            },
            "bio_thresholds": {
                "artifact_kind": bio_thresholds.get("artifact_kind"),
                "dataset_kind": bio_thresholds.get("dataset_kind"),
                "trained": bio_thresholds.get("trained"),
                "source_pipeline": bio_thresholds.get("source_pipeline"),
                "generated_by": bio_thresholds.get("generated_by"),
            },
        },
        "runtime_replay": runtime_replay,
        "backend_smoke": backend_smoke,
        "server_log": repo_rel(SERVER_LOG_PATH),
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[validation] report={repo_rel(REPORT_PATH)}")
    print(f"[validation] pose_word_best_val_top1={report['pose_word_training']['best_val_top1']:.4f}")
    print(f"[validation] bio_test_sign_boundary_f1={report['bio_training']['test_sign_boundary_f1']:.4f}")
    print(f"[validation] runtime_segments={report['runtime_replay']['segments_detected']}")
    print(f"[validation] runtime_commits={len(report['runtime_replay']['committed_words'])}")
    print(f"[validation] smoke_log={report['backend_smoke']['smoke']['log_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
