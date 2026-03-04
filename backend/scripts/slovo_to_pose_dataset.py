#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.pose import PoseFrame, PoseLandmarksGroup, compose_features_sequence  # noqa: E402

PATH_KEYS = ("path", "pose_path", "features_path", "feature_path", "npz_path", "npy_path", "clip_path")
LABEL_KEYS = ("label", "word", "gloss", "class", "text")
SIGNER_KEYS = ("signer_id", "user_id", "signer", "person_id", "subject_id")
SPLIT_KEYS = ("split", "subset")
FPS_KEYS = ("fps", "source_fps", "video_fps")
ALLOWED_SPLITS = {"train", "val", "test"}


@dataclass(slots=True)
class PoseSample:
    sample_id: str
    clip_id: str
    label: str
    signer_id: str
    split: str
    fps: float
    features: np.ndarray
    source_path: str
    meta: dict[str, Any]


@dataclass(slots=True)
class SplitConfig:
    val_ratio: float
    test_ratio: float



def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got '{value}'")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pose-word dataset from isolated Slovo pose clips into npz/index format.",
    )
    parser.add_argument("--pose-index", required=True, help="Path to index file (jsonl/json/csv) or directory with clips.")
    parser.add_argument("--out-dir", required=True, help="Output directory for train/val/test pose features.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when split is generated.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio when split is generated.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split reproducibility.")
    parser.add_argument("--split-by-signer", type=parse_bool, default=True, help="Split by signer to avoid leakage.")
    parser.add_argument("--resample-fps", type=float, default=25.0, help="Target fps for temporal resampling.")
    parser.add_argument("--use-shoulder-norm", type=parse_bool, default=True, help="Apply shoulder normalization.")
    parser.add_argument("--use-hands-3d-norm", type=parse_bool, default=False, help="Apply 3D canonical hand normalization.")
    parser.add_argument("--overwrite", type=parse_bool, default=False, help="Allow writing into non-empty output directory.")
    return parser



def _pick_first(record: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return default



def _parse_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows



def _parse_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            return [x for x in payload["items"] if isinstance(x, dict)]
        rows: list[dict[str, Any]] = []
        for split_name in ("train", "val", "test"):
            split_items = payload.get(split_name)
            if isinstance(split_items, list):
                for item in split_items:
                    if isinstance(item, dict):
                        enriched = dict(item)
                        enriched.setdefault("split", split_name)
                        rows.append(enriched)
        if rows:
            return rows
    raise ValueError(f"unsupported JSON format: {path}")



def _parse_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({str(k): v for k, v in row.items() if k})
    return rows



def _scan_directory_for_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed_suffixes = {".npz", ".npy", ".json", ".pose"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        rel = str(path.relative_to(root))
        rows.append(
            {
                "path": rel,
                "label": path.parent.name,
                "signer_id": "_unknown_signer",
            }
        )
    return rows



def load_records(pose_index: Path) -> tuple[list[dict[str, Any]], Path]:
    if not pose_index.exists():
        raise FileNotFoundError(f"pose index not found: {pose_index}")

    if pose_index.is_dir():
        for candidate in ("index.jsonl", "index.json", "index.csv"):
            candidate_path = pose_index / candidate
            if candidate_path.exists():
                return load_records(candidate_path)
        return _scan_directory_for_records(pose_index), pose_index

    suffix = pose_index.suffix.lower()
    if suffix == ".jsonl":
        return _parse_json_lines(pose_index), pose_index.parent
    if suffix == ".json":
        return _parse_json(pose_index), pose_index.parent
    if suffix == ".csv":
        return _parse_csv(pose_index), pose_index.parent
    raise ValueError(f"unsupported pose index format: {pose_index}")



def _parse_meta(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return {}
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}



def _ensure_2d_features(array: np.ndarray) -> np.ndarray:
    features = np.asarray(array, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    elif features.ndim == 3 and features.shape[-1] == 3:
        features = features.reshape(features.shape[0], -1)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [T,F], got {features.shape}")
    if features.shape[0] < 1:
        raise ValueError("feature sequence is empty")
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features.astype(np.float32)



def _array_or_none(npz_data: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in npz_data.files:
            return np.asarray(npz_data[key], dtype=np.float32)
    return None



def _pose_frames_from_landmarks(
    body: np.ndarray,
    left_hand: np.ndarray | None,
    right_hand: np.ndarray | None,
    body_conf: np.ndarray | None = None,
    left_conf: np.ndarray | None = None,
    right_conf: np.ndarray | None = None,
) -> list[PoseFrame]:
    if body.ndim != 3 or body.shape[1:] != (33, 3):
        raise ValueError(f"body must have shape [T,33,3], got {body.shape}")
    t_len = int(body.shape[0])

    if left_hand is not None and (left_hand.ndim != 3 or left_hand.shape[1:] != (21, 3)):
        raise ValueError(f"left_hand must have shape [T,21,3], got {left_hand.shape}")
    if right_hand is not None and (right_hand.ndim != 3 or right_hand.shape[1:] != (21, 3)):
        raise ValueError(f"right_hand must have shape [T,21,3], got {right_hand.shape}")

    frames: list[PoseFrame] = []
    for i in range(t_len):
        frames.append(
            PoseFrame(
                timestamp=float(i),
                body=PoseLandmarksGroup(
                    points=body[i],
                    confidence=body_conf[i] if body_conf is not None else None,
                ),
                left_hand=(
                    PoseLandmarksGroup(points=left_hand[i], confidence=left_conf[i] if left_conf is not None else None)
                    if left_hand is not None
                    else None
                ),
                right_hand=(
                    PoseLandmarksGroup(points=right_hand[i], confidence=right_conf[i] if right_conf is not None else None)
                    if right_hand is not None
                    else None
                ),
                meta={},
            )
        )
    return frames



def _load_pose_file(path: Path) -> np.ndarray:
    try:
        from pose_format import Pose  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pose-format is required to read .pose files. Install pose-format or pre-convert to npz/npy."
        ) from exc

    pose = Pose.read(path.read_bytes())
    tensor = np.asarray(pose.body.data, dtype=np.float32)
    if tensor.ndim == 4:
        tensor = tensor[:, 0, :, :]
    if tensor.ndim != 3:
        raise ValueError(f"unsupported .pose shape: {tensor.shape}")
    if tensor.shape[-1] > 3:
        tensor = tensor[..., :3]
    return _ensure_2d_features(tensor)



def _load_clip_features(
    path: Path,
    *,
    use_shoulder_norm: bool,
    use_hands_3d_norm: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _ensure_2d_features(np.load(path, allow_pickle=True)), {}

    if suffix == ".pose":
        return _load_pose_file(path), {}

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"json clip must be object: {path}")
        if "features" in payload:
            return _ensure_2d_features(np.asarray(payload["features"], dtype=np.float32)), _parse_meta(payload.get("meta"))
        if "body" in payload:
            body = np.asarray(payload["body"], dtype=np.float32)
            left_hand = np.asarray(payload.get("left_hand"), dtype=np.float32) if payload.get("left_hand") is not None else None
            right_hand = np.asarray(payload.get("right_hand"), dtype=np.float32) if payload.get("right_hand") is not None else None
            frames = _pose_frames_from_landmarks(body, left_hand, right_hand)
            features, _ = compose_features_sequence(
                frames,
                apply_shoulder_norm=use_shoulder_norm,
                canonical_hands_3d=use_hands_3d_norm,
                include_velocity=False,
            )
            return _ensure_2d_features(features), _parse_meta(payload.get("meta"))
        raise ValueError(f"json clip must contain 'features' or 'body': {path}")

    if suffix != ".npz":
        raise ValueError(f"unsupported clip format: {path}")

    with np.load(path, allow_pickle=True) as data:
        meta = _parse_meta(data["meta"]) if "meta" in data.files else {}
        if "features" in data.files:
            return _ensure_2d_features(np.asarray(data["features"], dtype=np.float32)), meta

        body = _array_or_none(data, ("body",))
        if body is None:
            raise ValueError(f"npz must contain 'features' or 'body': {path}")

        left_hand = _array_or_none(data, ("left_hand", "lh"))
        right_hand = _array_or_none(data, ("right_hand", "rh"))
        body_conf = _array_or_none(data, ("body_confidence", "body_visibility"))
        left_conf = _array_or_none(data, ("left_hand_confidence", "lh_confidence"))
        right_conf = _array_or_none(data, ("right_hand_confidence", "rh_confidence"))

        frames = _pose_frames_from_landmarks(
            body=body,
            left_hand=left_hand,
            right_hand=right_hand,
            body_conf=body_conf,
            left_conf=left_conf,
            right_conf=right_conf,
        )
        features, _ = compose_features_sequence(
            frames,
            apply_shoulder_norm=use_shoulder_norm,
            canonical_hands_3d=use_hands_3d_norm,
            include_velocity=False,
        )
        return _ensure_2d_features(features), meta



def _resample_features(features: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps <= 0 or dst_fps <= 0:
        return features.astype(np.float32)
    if abs(src_fps - dst_fps) < 1e-6:
        return features.astype(np.float32)

    t_len, feat_dim = features.shape
    target_t = max(1, int(round(t_len * float(dst_fps) / float(src_fps))))
    if target_t == t_len:
        return features.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=t_len, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_t, dtype=np.float32)
    out = np.empty((target_t, feat_dim), dtype=np.float32)
    for i in range(feat_dim):
        out[:, i] = np.interp(x_new, x_old, features[:, i]).astype(np.float32)
    return out



def _normalize_split(value: str) -> str:
    text = str(value).strip().lower()
    if text == "valid":
        text = "val"
    return text if text in ALLOWED_SPLITS else ""



def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "sample"



def _build_sample(
    record: dict[str, Any],
    *,
    base_dir: Path,
    idx: int,
    cfg: argparse.Namespace,
) -> PoseSample | None:
    path_value = _pick_first(record, PATH_KEYS, "")
    source_path = ""
    extra_meta: dict[str, Any] = {}

    if path_value:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"clip path not found: {path}")
        features, extra_meta = _load_clip_features(
            path,
            use_shoulder_norm=cfg.use_shoulder_norm,
            use_hands_3d_norm=cfg.use_hands_3d_norm,
        )
        source_path = str(path)
    elif "features" in record:
        features = _ensure_2d_features(np.asarray(record["features"], dtype=np.float32))
        source_path = "<inline>"
    else:
        return None

    label = str(_pick_first(record, LABEL_KEYS, _pick_first(extra_meta, LABEL_KEYS, ""))).strip()
    if not label and source_path and source_path != "<inline>":
        label = Path(source_path).parent.name
    if not label:
        raise ValueError(f"label is missing for sample idx={idx}")

    signer_id = str(_pick_first(record, SIGNER_KEYS, _pick_first(extra_meta, SIGNER_KEYS, "_unknown_signer"))).strip()
    if not signer_id:
        signer_id = "_unknown_signer"

    raw_split = _pick_first(record, SPLIT_KEYS, _pick_first(extra_meta, SPLIT_KEYS, ""))
    split = _normalize_split(str(raw_split))

    clip_id = str(record.get("clip_id") or extra_meta.get("clip_id") or f"clip_{idx:07d}").strip()
    if not clip_id:
        clip_id = f"clip_{idx:07d}"

    raw_fps = _pick_first(record, FPS_KEYS, _pick_first(extra_meta, FPS_KEYS, cfg.resample_fps))
    src_fps = float(raw_fps) if str(raw_fps).strip() != "" else float(cfg.resample_fps)
    if src_fps <= 0:
        src_fps = float(cfg.resample_fps)
    features = _resample_features(features, src_fps=src_fps, dst_fps=float(cfg.resample_fps))

    sample_id = f"{_slug(label)}_{_slug(clip_id)}_{idx:07d}"
    merged_meta = dict(extra_meta)
    for key in ("session_id", "camera", "source", "source_id"):
        if key in record and key not in merged_meta:
            merged_meta[key] = record[key]

    return PoseSample(
        sample_id=sample_id,
        clip_id=clip_id,
        label=label,
        signer_id=signer_id,
        split=split,
        fps=float(cfg.resample_fps),
        features=features.astype(np.float32),
        source_path=source_path,
        meta=merged_meta,
    )



def _split_by_signer(samples: list[PoseSample], *, seed: int, cfg: SplitConfig) -> dict[str, list[PoseSample]]:
    signer_to_samples: dict[str, list[PoseSample]] = {}
    for sample in samples:
        signer_to_samples.setdefault(sample.signer_id, []).append(sample)

    signer_ids = sorted(signer_to_samples)
    rng = random.Random(seed)
    rng.shuffle(signer_ids)

    total = len(samples)
    target_test = max(1, int(round(total * cfg.test_ratio)))
    target_val = max(1, int(round(total * cfg.val_ratio)))

    split_signers: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    val_count = 0
    test_count = 0
    for signer_id in signer_ids:
        count = len(signer_to_samples[signer_id])
        if test_count < target_test:
            split_signers["test"].append(signer_id)
            test_count += count
        elif val_count < target_val:
            split_signers["val"].append(signer_id)
            val_count += count
        else:
            split_signers["train"].append(signer_id)

    if not split_signers["train"] and split_signers["val"]:
        split_signers["train"].append(split_signers["val"].pop())
    if not split_signers["train"] and split_signers["test"]:
        split_signers["train"].append(split_signers["test"].pop())

    result = {"train": [], "val": [], "test": []}
    for split_name, signer_list in split_signers.items():
        for signer_id in signer_list:
            result[split_name].extend(signer_to_samples.get(signer_id, []))
    return result



def _split_random(samples: list[PoseSample], *, seed: int, cfg: SplitConfig) -> dict[str, list[PoseSample]]:
    rng = random.Random(seed)
    order = list(samples)
    rng.shuffle(order)

    total = len(order)
    test_n = max(1, int(round(total * cfg.test_ratio)))
    val_n = max(1, int(round(total * cfg.val_ratio)))
    train_n = max(1, total - val_n - test_n)

    train = order[:train_n]
    val = order[train_n : train_n + val_n]
    test = order[train_n + val_n :]

    if not val and train:
        val = [train.pop()]
    if not test and train:
        test = [train.pop()]

    return {"train": train, "val": val, "test": test}



def split_samples(
    samples: list[PoseSample],
    *,
    seed: int,
    split_by_signer: bool,
    split_cfg: SplitConfig,
) -> dict[str, list[PoseSample]]:
    explicit: dict[str, list[PoseSample]] = {"train": [], "val": [], "test": []}
    pending: list[PoseSample] = []

    for sample in samples:
        if sample.split in ALLOWED_SPLITS:
            explicit[sample.split].append(sample)
        else:
            pending.append(sample)

    if pending:
        if split_by_signer and any(sample.signer_id != "_unknown_signer" for sample in pending):
            auto = _split_by_signer(pending, seed=seed, cfg=split_cfg)
        else:
            auto = _split_random(pending, seed=seed, cfg=split_cfg)
        for split_name in ALLOWED_SPLITS:
            explicit[split_name].extend(auto[split_name])

    if not explicit["train"]:
        fallback = explicit["val"] or explicit["test"]
        if fallback:
            explicit["train"].append(fallback.pop())

    return explicit



def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def _labels_sorted(samples: Iterable[PoseSample]) -> list[str]:
    labels = sorted({sample.label for sample in samples})
    if not labels:
        raise ValueError("no labels found in dataset")
    return labels



def _summarize(
    split_samples_map: dict[str, list[PoseSample]],
    labels: list[str],
    out_index_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        samples = split_samples_map[split_name]
        frame_counts = [int(sample.features.shape[0]) for sample in samples]
        signer_counts = Counter(sample.signer_id for sample in samples)
        label_counts = Counter(sample.label for sample in samples)
        per_split[split_name] = {
            "samples": int(len(samples)),
            "frames_total": int(sum(frame_counts)),
            "frames_min": int(min(frame_counts)) if frame_counts else 0,
            "frames_max": int(max(frame_counts)) if frame_counts else 0,
            "frames_mean": float(np.mean(frame_counts)) if frame_counts else 0.0,
            "unique_signers": int(len(signer_counts)),
            "labels": {label: int(label_counts.get(label, 0)) for label in labels},
        }

    feature_dims = sorted({int(row["feature_dim"]) for row in out_index_rows})
    split_pairs = {str((row["split"], row["signer_id"])) for row in out_index_rows}

    return {
        "samples_total": int(sum(len(v) for v in split_samples_map.values())),
        "labels_total": int(len(labels)),
        "labels": labels,
        "feature_dims": feature_dims,
        "split_signer_pairs": int(len(split_pairs)),
        "splits": per_split,
    }



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pose_index = Path(args.pose_index).resolve()
    out_dir = Path(args.out_dir).resolve()
    split_cfg = SplitConfig(val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio))

    if split_cfg.val_ratio <= 0 or split_cfg.test_ratio <= 0 or (split_cfg.val_ratio + split_cfg.test_ratio) >= 1.0:
        raise ValueError("invalid split ratios: require 0 < val_ratio,test_ratio and val+test < 1")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out_dir} (use --overwrite true)")
    out_dir.mkdir(parents=True, exist_ok=True)

    records, base_dir = load_records(pose_index)
    samples: list[PoseSample] = []
    for idx, record in enumerate(records):
        sample = _build_sample(record, base_dir=base_dir, idx=idx, cfg=args)
        if sample is None:
            continue
        samples.append(sample)

    if not samples:
        raise ValueError("no valid samples were loaded")

    split_map = split_samples(
        samples,
        seed=int(args.seed),
        split_by_signer=bool(args.split_by_signer),
        split_cfg=split_cfg,
    )

    labels = _labels_sorted(samples)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    all_index_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    for split_name in ("train", "val", "test"):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for item_no, sample in enumerate(split_map[split_name]):
            sample_id = sample.sample_id
            if sample_id in used_ids:
                sample_id = f"{sample_id}_{item_no:04d}"
            used_ids.add(sample_id)

            file_path = split_dir / f"{sample_id}.npz"
            meta_payload = dict(sample.meta)
            meta_payload.update(
                {
                    "label": sample.label,
                    "signer_id": sample.signer_id,
                    "clip_id": sample.clip_id,
                    "fps": float(sample.fps),
                    "source_path": sample.source_path,
                }
            )

            np.savez_compressed(
                file_path,
                features=sample.features.astype(np.float32),
                label=np.asarray(sample.label),
                label_idx=np.asarray(label_to_idx[sample.label], dtype=np.int64),
                signer_id=np.asarray(sample.signer_id),
                clip_id=np.asarray(sample.clip_id),
                split=np.asarray(split_name),
                source_path=np.asarray(sample.source_path),
                meta=np.asarray(json.dumps(meta_payload, ensure_ascii=False)),
            )

            row = {
                "sample_id": sample_id,
                "clip_id": sample.clip_id,
                "label": sample.label,
                "label_idx": int(label_to_idx[sample.label]),
                "signer_id": sample.signer_id,
                "split": split_name,
                "path": str(file_path.relative_to(out_dir)),
                "num_frames": int(sample.features.shape[0]),
                "feature_dim": int(sample.features.shape[1]),
                "fps": float(sample.fps),
                "source_path": sample.source_path,
            }
            all_index_rows.append(row)

    index_path = out_dir / "index.jsonl"
    _write_jsonl(index_path, all_index_rows)

    for split_name in ("train", "val", "test"):
        split_rows = [row for row in all_index_rows if row["split"] == split_name]
        _write_jsonl(out_dir / f"{split_name}.jsonl", split_rows)

    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    stats = _summarize(split_map, labels=labels, out_index_rows=all_index_rows)
    stats.update(
        {
            "seed": int(args.seed),
            "pose_index": str(pose_index),
            "resample_fps": float(args.resample_fps),
            "use_shoulder_norm": bool(args.use_shoulder_norm),
            "use_hands_3d_norm": bool(args.use_hands_3d_norm),
            "split_by_signer": bool(args.split_by_signer),
            "ratios": {
                "train": float(1.0 - split_cfg.val_ratio - split_cfg.test_ratio),
                "val": float(split_cfg.val_ratio),
                "test": float(split_cfg.test_ratio),
            },
        }
    )
    (out_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[slovo_to_pose_dataset] completed")
    print(f"  out_dir={out_dir}")
    print(f"  samples_total={stats['samples_total']}")
    print(f"  labels_total={stats['labels_total']}")
    for split_name in ("train", "val", "test"):
        split_stats = stats["splits"][split_name]
        print(
            f"  {split_name}: samples={split_stats['samples']} frames_total={split_stats['frames_total']} unique_signers={split_stats['unique_signers']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
