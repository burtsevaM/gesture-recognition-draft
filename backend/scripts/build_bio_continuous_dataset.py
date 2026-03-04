#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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


BIO_B = 0
BIO_I = 1
BIO_O = 2
BIO_MAPPING = {"B": BIO_B, "I": BIO_I, "O": BIO_O}

PATH_KEYS = ("features_path", "feature_path", "pose_path", "path", "npz_path", "npy_path")
LABEL_KEYS = ("label", "word", "gloss", "class", "text")
SIGNER_KEYS = ("signer_id", "user_id", "signer", "person_id", "subject_id")
FPS_KEYS = ("fps", "source_fps", "video_fps")


@dataclass(slots=True)
class PoseClip:
    clip_id: str
    label: str
    signer_id: str
    fps: float
    features: np.ndarray
    source_path: str


@dataclass(slots=True)
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build synthetic continuous BIO dataset from isolated pose clips.")
    parser.add_argument("--pose_index", required=True, help="Path to index.jsonl/.json or directory with pose clips.")
    parser.add_argument("--out_dir", required=True, help="Output directory for train/val/test npz.")
    parser.add_argument("--min_words", type=int, default=3)
    parser.add_argument("--max_words", type=int, default=12)
    parser.add_argument("--pause_min", type=int, default=3)
    parser.add_argument("--pause_max", type=int, default=12)
    parser.add_argument("--phrase_min_words", type=int, default=2)
    parser.add_argument("--phrase_max_words", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits_by_signer", type=parse_bool, default=True)
    parser.add_argument("--resample_fps", type=float, default=25.0)
    parser.add_argument("--window_fps", type=float, default=None)
    parser.add_argument("--use_shoulder_norm", type=parse_bool, default=True)
    parser.add_argument("--use_hands_3d_norm", type=parse_bool, default=False)
    parser.add_argument("--pause_noise_std", type=float, default=0.005)
    parser.add_argument("--pause_inside_phrase_as_i", type=parse_bool, default=False)
    parser.add_argument("--train_samples", type=int, default=0, help="Override number of generated train sequences.")
    parser.add_argument("--val_samples", type=int, default=0, help="Override number of generated val sequences.")
    parser.add_argument("--test_samples", type=int, default=0, help="Override number of generated test sequences.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
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
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(item, dict):
                continue
            rows.append(item)
    return rows


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
        value = value.strip()
        if not value:
            return {}
        try:
            obj = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return obj if isinstance(obj, dict) else {}
    return {}


def _ensure_2d_features(array: np.ndarray) -> np.ndarray:
    features = np.asarray(array, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [T, F], got {features.shape}")
    if features.shape[0] < 1:
        raise ValueError("features must contain at least one frame")
    if not np.all(np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
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
        raise ValueError(f"body landmarks must have shape [T,33,3], got {body.shape}")
    t = body.shape[0]
    frames: list[PoseFrame] = []

    if left_hand is not None and (left_hand.ndim != 3 or left_hand.shape[1:] != (21, 3)):
        raise ValueError(f"left hand landmarks must have shape [T,21,3], got {left_hand.shape}")
    if right_hand is not None and (right_hand.ndim != 3 or right_hand.shape[1:] != (21, 3)):
        raise ValueError(f"right hand landmarks must have shape [T,21,3], got {right_hand.shape}")

    for i in range(t):
        body_group = PoseLandmarksGroup(points=body[i], confidence=body_conf[i] if body_conf is not None else None)
        lh_group = None
        rh_group = None
        if left_hand is not None:
            lh_group = PoseLandmarksGroup(points=left_hand[i], confidence=left_conf[i] if left_conf is not None else None)
        if right_hand is not None:
            rh_group = PoseLandmarksGroup(points=right_hand[i], confidence=right_conf[i] if right_conf is not None else None)
        frames.append(
            PoseFrame(
                timestamp=float(i),
                body=body_group,
                left_hand=lh_group,
                right_hand=rh_group,
                meta={},
            )
        )
    return frames


def _load_from_pose_file(path: Path) -> np.ndarray:
    try:
        from pose_format import Pose  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pose-format is required to read .pose files. Install pose-format or preconvert to npz/npy features."
        ) from exc

    raw = path.read_bytes()
    pose = Pose.read(raw)  # pragma: no cover - depends on optional package
    data = np.asarray(pose.body.data, dtype=np.float32)
    # Common shape: [T, person, landmarks, dims]
    if data.ndim == 4:
        data = data[:, 0, :, :]
    if data.ndim != 3:
        raise ValueError(f"unsupported .pose tensor shape: {data.shape}")
    if data.shape[-1] > 3:
        data = data[..., :3]
    return _ensure_2d_features(data.reshape(data.shape[0], -1))


def _load_clip_from_path(
    path: Path,
    *,
    use_shoulder_norm: bool,
    use_hands_3d_norm: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"clip file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _ensure_2d_features(np.load(path, allow_pickle=True)), {}

    if suffix == ".pose":
        features = _load_from_pose_file(path)
        return features, {}

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"json clip must be an object: {path}")
        if "features" not in payload:
            raise ValueError(f"json clip must contain 'features': {path}")
        features = _ensure_2d_features(np.asarray(payload["features"], dtype=np.float32))
        meta = payload.get("meta", {})
        return features, meta if isinstance(meta, dict) else {}

    if suffix != ".npz":
        raise ValueError(f"unsupported clip format: {path}")

    with np.load(path, allow_pickle=True) as data:
        meta = _parse_meta(data["meta"]) if "meta" in data.files else {}
        if "features" in data.files:
            features = _ensure_2d_features(np.asarray(data["features"], dtype=np.float32))
            return features, meta

        body = _array_or_none(data, ("body",))
        if body is None:
            raise ValueError(f"npz file must contain 'features' or 'body': {path}")
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

    t, f = features.shape
    target_t = max(1, int(round(t * float(dst_fps) / float(src_fps))))
    if target_t == t:
        return features.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, num=t, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_t, dtype=np.float32)
    out = np.empty((target_t, f), dtype=np.float32)
    for i in range(f):
        out[:, i] = np.interp(x_new, x_old, features[:, i]).astype(np.float32)
    return out


def _build_clip(record: dict[str, Any], *, base_dir: Path, target_fps: float, cfg: argparse.Namespace, idx: int) -> PoseClip | None:
    path_value = _pick_first(record, PATH_KEYS, "")
    meta_from_path: dict[str, Any] = {}
    if path_value:
        path = Path(str(path_value))
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        features, meta_from_path = _load_clip_from_path(
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

    label = str(_pick_first(record, LABEL_KEYS, "")).strip()
    if not label:
        label = str(_pick_first(meta_from_path, LABEL_KEYS, "")).strip()
    if not label:
        candidate = Path(source_path).parent.name if source_path != "<inline>" else ""
        label = candidate.strip()
    if not label:
        raise ValueError(f"label is missing for clip source={source_path}")

    signer_id = str(_pick_first(record, SIGNER_KEYS, _pick_first(meta_from_path, SIGNER_KEYS, "_unknown_signer"))).strip()
    if not signer_id:
        signer_id = "_unknown_signer"

    clip_fps = float(_pick_first(record, FPS_KEYS, _pick_first(meta_from_path, FPS_KEYS, target_fps)))
    if clip_fps <= 0:
        clip_fps = target_fps
    features = _resample_features(features, src_fps=clip_fps, dst_fps=target_fps)

    clip_id = str(record.get("clip_id", "")).strip() or f"clip_{idx:07d}"
    return PoseClip(
        clip_id=clip_id,
        label=label,
        signer_id=signer_id,
        fps=float(target_fps),
        features=features.astype(np.float32),
        source_path=source_path,
    )


def _load_records(pose_index: Path) -> tuple[list[dict[str, Any]], Path]:
    if pose_index.is_file():
        if pose_index.suffix.lower() == ".jsonl":
            return _parse_json_lines(pose_index), pose_index.parent
        if pose_index.suffix.lower() == ".json":
            payload = json.loads(pose_index.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [x for x in payload if isinstance(x, dict)], pose_index.parent
            if isinstance(payload, dict):
                if "items" in payload and isinstance(payload["items"], list):
                    return [x for x in payload["items"] if isinstance(x, dict)], pose_index.parent
                for key in ("train", "val", "test"):
                    if key in payload and isinstance(payload[key], list):
                        rows = []
                        for item in payload[key]:
                            if isinstance(item, dict):
                                rows.append(item)
                        return rows, pose_index.parent
            raise ValueError(f"unsupported json index format: {pose_index}")
        raise ValueError(f"unsupported index file format: {pose_index}")

    if not pose_index.is_dir():
        raise FileNotFoundError(f"pose_index not found: {pose_index}")

    index_jsonl = pose_index / "index.jsonl"
    if index_jsonl.exists():
        return _parse_json_lines(index_jsonl), index_jsonl.parent

    rows: list[dict[str, Any]] = []
    for file_path in sorted(pose_index.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".npz", ".npy", ".pose", ".json"}:
            continue
        rows.append(
            {
                "path": str(file_path.relative_to(pose_index)),
                "label": file_path.parent.name,
                "signer_id": "_unknown_signer",
            }
        )
    return rows, pose_index


def load_pose_clips(pose_index: Path, *, target_fps: float, cfg: argparse.Namespace) -> list[PoseClip]:
    records, base_dir = _load_records(pose_index)
    clips: list[PoseClip] = []
    for idx, record in enumerate(records):
        clip = _build_clip(record, base_dir=base_dir, target_fps=target_fps, cfg=cfg, idx=idx)
        if clip is None:
            continue
        clips.append(clip)
    if not clips:
        raise ValueError("no valid clips found")
    return clips


def _split_by_signer(clips: list[PoseClip], *, seed: int, ratios: SplitConfig) -> dict[str, list[PoseClip]]:
    signer_to_clips: dict[str, list[PoseClip]] = {}
    for clip in clips:
        signer_to_clips.setdefault(clip.signer_id, []).append(clip)

    signer_ids = sorted(signer_to_clips)
    rng = random.Random(seed)
    rng.shuffle(signer_ids)

    total = len(clips)
    target_val = max(1, int(round(total * ratios.val_ratio)))
    target_test = max(1, int(round(total * ratios.test_ratio)))

    train_signers: list[str] = []
    val_signers: list[str] = []
    test_signers: list[str] = []
    val_count = 0
    test_count = 0

    for signer_id in signer_ids:
        n = len(signer_to_clips[signer_id])
        if test_count < target_test:
            test_signers.append(signer_id)
            test_count += n
        elif val_count < target_val:
            val_signers.append(signer_id)
            val_count += n
        else:
            train_signers.append(signer_id)

    if not train_signers and val_signers:
        train_signers.append(val_signers.pop())
    if not train_signers and test_signers:
        train_signers.append(test_signers.pop())

    def gather(signers: list[str]) -> list[PoseClip]:
        out: list[PoseClip] = []
        for signer_id in signers:
            out.extend(signer_to_clips.get(signer_id, []))
        return out

    return {
        "train": gather(train_signers),
        "val": gather(val_signers),
        "test": gather(test_signers),
    }


def _split_random(clips: list[PoseClip], *, seed: int, ratios: SplitConfig) -> dict[str, list[PoseClip]]:
    rng = random.Random(seed)
    order = list(clips)
    rng.shuffle(order)

    n = len(order)
    n_test = max(1, int(round(n * ratios.test_ratio)))
    n_val = max(1, int(round(n * ratios.val_ratio)))
    n_train = max(1, n - n_val - n_test)

    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]
    if not val and train:
        val = [train.pop()]
    if not test and train:
        test = [train.pop()]
    return {"train": train, "val": val, "test": test}


def split_clips(
    clips: list[PoseClip],
    *,
    seed: int,
    ratios: SplitConfig,
    by_signer: bool,
) -> dict[str, list[PoseClip]]:
    if by_signer:
        unique_signers = {clip.signer_id for clip in clips if clip.signer_id != "_unknown_signer"}
        if unique_signers:
            return _split_by_signer(clips, seed=seed, ratios=ratios)
    return _split_random(clips, seed=seed, ratios=ratios)


def _auto_sample_count(num_clips: int, explicit_value: int) -> int:
    if explicit_value > 0:
        return explicit_value
    return max(1, int(num_clips))


def _phrase_ids(num_words: int, *, min_words: int, max_words: int, rng: random.Random) -> list[int]:
    ids: list[int] = []
    phrase_id = 0
    i = 0
    while i < num_words:
        size = rng.randint(max(1, min_words), max(1, max_words))
        size = max(1, min(size, num_words - i))
        ids.extend([phrase_id] * size)
        phrase_id += 1
        i += size
    return ids


def _pause_frames(last_frame: np.ndarray, length: int, noise_std: float, *, rng_np: np.random.Generator) -> np.ndarray:
    base = np.repeat(last_frame.reshape(1, -1), repeats=max(1, length), axis=0).astype(np.float32)
    if noise_std > 0.0:
        noise = rng_np.normal(loc=0.0, scale=float(noise_std), size=base.shape).astype(np.float32)
        base = base + noise
    return base.astype(np.float32)


def _to_serializable_meta(meta: dict[str, Any]) -> str:
    return json.dumps(meta, ensure_ascii=False)


def _bio_counts(values: np.ndarray) -> dict[str, int]:
    cnt = Counter(int(x) for x in values.tolist())
    return {"B": int(cnt.get(BIO_B, 0)), "I": int(cnt.get(BIO_I, 0)), "O": int(cnt.get(BIO_O, 0))}


def build_continuous_sample(
    *,
    clip_pool: list[PoseClip],
    sample_id: str,
    split_name: str,
    rng: random.Random,
    rng_np: np.random.Generator,
    min_words: int,
    max_words: int,
    pause_min: int,
    pause_max: int,
    phrase_min_words: int,
    phrase_max_words: int,
    pause_inside_phrase_as_i: bool,
    pause_noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    num_words = rng.randint(max(1, min_words), max(1, max_words))
    selected = [rng.choice(clip_pool) for _ in range(num_words)]
    phrase_ids = _phrase_ids(num_words, min_words=phrase_min_words, max_words=phrase_max_words, rng=rng)

    features_chunks: list[np.ndarray] = []
    sign_chunks: list[np.ndarray] = []
    phrase_chunks: list[np.ndarray] = []
    word_meta: list[dict[str, Any]] = []
    pause_meta: list[dict[str, Any]] = []

    cursor = 0
    for word_idx, clip in enumerate(selected):
        clip_feat = clip.features.astype(np.float32)
        t = int(clip_feat.shape[0])
        start = cursor
        end = cursor + t - 1
        phrase_id = int(phrase_ids[word_idx])
        is_phrase_start = word_idx == 0 or phrase_ids[word_idx - 1] != phrase_id

        features_chunks.append(clip_feat)

        sign = np.full((t,), BIO_I, dtype=np.int64)
        sign[0] = BIO_B
        sign_chunks.append(sign)

        phrase = np.full((t,), BIO_I, dtype=np.int64)
        if is_phrase_start:
            phrase[0] = BIO_B
        phrase_chunks.append(phrase)

        word_meta.append(
            {
                "word_index": int(word_idx),
                "label": clip.label,
                "clip_id": clip.clip_id,
                "signer_id": clip.signer_id,
                "start": int(start),
                "end": int(end),
                "length": int(t),
                "phrase_id": int(phrase_id),
                "is_phrase_start": bool(is_phrase_start),
                "source_path": clip.source_path,
            }
        )

        cursor = end + 1

        if word_idx >= num_words - 1:
            continue

        pause_len = int(rng.randint(max(0, pause_min), max(0, pause_max)))
        if pause_len <= 0:
            continue
        pause_feat = _pause_frames(clip_feat[-1], pause_len, pause_noise_std, rng_np=rng_np)
        features_chunks.append(pause_feat)

        sign_pause = np.full((pause_len,), BIO_O, dtype=np.int64)
        sign_chunks.append(sign_pause)

        same_phrase = phrase_ids[word_idx] == phrase_ids[word_idx + 1]
        use_i = bool(pause_inside_phrase_as_i and same_phrase)
        phrase_pause_value = BIO_I if use_i else BIO_O
        phrase_pause = np.full((pause_len,), phrase_pause_value, dtype=np.int64)
        phrase_chunks.append(phrase_pause)

        pause_meta.append(
            {
                "between_words": [int(word_idx), int(word_idx + 1)],
                "start": int(cursor),
                "end": int(cursor + pause_len - 1),
                "length": int(pause_len),
                "inside_phrase": bool(same_phrase),
                "phrase_value": "I" if use_i else "O",
            }
        )
        cursor += pause_len

    features = np.concatenate(features_chunks, axis=0).astype(np.float32)
    sign_bio = np.concatenate(sign_chunks, axis=0).astype(np.int64)
    phrase_bio = np.concatenate(phrase_chunks, axis=0).astype(np.int64)

    if not (len(features) == len(sign_bio) == len(phrase_bio)):
        raise RuntimeError("generated sequence length mismatch")

    phrase_meta: dict[int, dict[str, Any]] = {}
    for word in word_meta:
        pid = int(word["phrase_id"])
        if pid not in phrase_meta:
            phrase_meta[pid] = {
                "phrase_id": pid,
                "word_indices": [int(word["word_index"])],
                "start": int(word["start"]),
                "end": int(word["end"]),
            }
        else:
            phrase_meta[pid]["word_indices"].append(int(word["word_index"]))
            phrase_meta[pid]["end"] = int(word["end"])

    phrases = [phrase_meta[k] for k in sorted(phrase_meta)]

    meta = {
        "sample_id": sample_id,
        "split": split_name,
        "seed": int(seed),
        "bio_mapping": BIO_MAPPING,
        "num_frames": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "words": word_meta,
        "phrases": phrases,
        "pauses": pause_meta,
        "signer_ids": sorted({w["signer_id"] for w in word_meta}),
        "word_labels": [w["label"] for w in word_meta],
    }
    return features, sign_bio, phrase_bio, meta


def _sequence_stats(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "count": int(arr.size),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _share(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {"B": 0.0, "I": 0.0, "O": 0.0}
    return {k: float(v / total) for k, v in counts.items()}


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    pose_index = Path(args.pose_index).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.min_words > args.max_words:
        raise ValueError("--min_words cannot be greater than --max_words")
    if args.pause_min > args.pause_max:
        raise ValueError("--pause_min cannot be greater than --pause_max")
    if args.phrase_min_words > args.phrase_max_words:
        raise ValueError("--phrase_min_words cannot be greater than --phrase_max_words")
    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    target_fps = float(args.window_fps if args.window_fps is not None else args.resample_fps)
    if target_fps <= 0:
        raise ValueError("target fps must be > 0")

    clips = load_pose_clips(pose_index, target_fps=target_fps, cfg=args)
    split_map = split_clips(
        clips,
        seed=int(args.seed),
        ratios=SplitConfig(args.train_ratio, args.val_ratio, args.test_ratio),
        by_signer=bool(args.splits_by_signer),
    )

    rng = random.Random(int(args.seed))
    rng_np = np.random.default_rng(int(args.seed))

    stats: dict[str, Any] = {
        "config": {
            "pose_index": str(pose_index),
            "target_fps": float(target_fps),
            "min_words": int(args.min_words),
            "max_words": int(args.max_words),
            "pause_min": int(args.pause_min),
            "pause_max": int(args.pause_max),
            "phrase_min_words": int(args.phrase_min_words),
            "phrase_max_words": int(args.phrase_max_words),
            "seed": int(args.seed),
            "splits_by_signer": bool(args.splits_by_signer),
            "use_shoulder_norm": bool(args.use_shoulder_norm),
            "use_hands_3d_norm": bool(args.use_hands_3d_norm),
            "pause_inside_phrase_as_i": bool(args.pause_inside_phrase_as_i),
            "pause_noise_std": float(args.pause_noise_std),
        },
        "bio_mapping": BIO_MAPPING,
        "source": {
            "clips_total": int(len(clips)),
            "labels_total": int(len({clip.label for clip in clips})),
            "signers_total": int(len({clip.signer_id for clip in clips})),
        },
        "splits": {},
    }

    total_sign_counts = Counter()
    total_phrase_counts = Counter()
    total_lengths: list[int] = []
    total_pauses: list[int] = []

    requested_samples = {
        "train": int(args.train_samples),
        "val": int(args.val_samples),
        "test": int(args.test_samples),
    }

    for split_name in ("train", "val", "test"):
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for old in split_dir.glob("*.npz"):
            old.unlink()

        pool = split_map.get(split_name, [])
        if not pool:
            stats["splits"][split_name] = {
                "clips": 0,
                "samples": 0,
                "frames": 0,
                "length": _sequence_stats([]),
                "sign_bio_counts": {"B": 0, "I": 0, "O": 0},
                "phrase_bio_counts": {"B": 0, "I": 0, "O": 0},
                "sign_bio_share": {"B": 0.0, "I": 0.0, "O": 0.0},
                "phrase_bio_share": {"B": 0.0, "I": 0.0, "O": 0.0},
                "avg_words_per_sample": 0.0,
                "avg_pause_length": 0.0,
            }
            continue

        n_samples = _auto_sample_count(len(pool), requested_samples[split_name])
        seq_lengths: list[int] = []
        pause_lengths: list[int] = []
        words_per_sample: list[int] = []
        sign_counts = Counter()
        phrase_counts = Counter()

        for i in range(n_samples):
            sample_id = f"{split_name}_{i:06d}"
            features, sign_bio, phrase_bio, meta = build_continuous_sample(
                clip_pool=pool,
                sample_id=sample_id,
                split_name=split_name,
                rng=rng,
                rng_np=rng_np,
                min_words=int(args.min_words),
                max_words=int(args.max_words),
                pause_min=int(args.pause_min),
                pause_max=int(args.pause_max),
                phrase_min_words=int(args.phrase_min_words),
                phrase_max_words=int(args.phrase_max_words),
                pause_inside_phrase_as_i=bool(args.pause_inside_phrase_as_i),
                pause_noise_std=float(args.pause_noise_std),
                seed=int(args.seed),
            )
            out_path = split_dir / f"{sample_id}.npz"
            np.savez_compressed(
                out_path,
                features=features.astype(np.float32),
                sign_bio=sign_bio.astype(np.int64),
                phrase_bio=phrase_bio.astype(np.int64),
                meta=np.array(_to_serializable_meta(meta)),
            )

            seq_lengths.append(int(features.shape[0]))
            words_per_sample.append(len(meta["words"]))
            pause_lengths.extend(int(p["length"]) for p in meta["pauses"])
            sign_counts.update(_bio_counts(sign_bio))
            phrase_counts.update(_bio_counts(phrase_bio))
            total_sign_counts.update(_bio_counts(sign_bio))
            total_phrase_counts.update(_bio_counts(phrase_bio))
            total_lengths.append(int(features.shape[0]))
            total_pauses.extend(int(p["length"]) for p in meta["pauses"])

        stats["splits"][split_name] = {
            "clips": int(len(pool)),
            "samples": int(n_samples),
            "frames": int(sum(seq_lengths)),
            "length": _sequence_stats(seq_lengths),
            "sign_bio_counts": {
                "B": int(sign_counts.get("B", 0)),
                "I": int(sign_counts.get("I", 0)),
                "O": int(sign_counts.get("O", 0)),
            },
            "phrase_bio_counts": {
                "B": int(phrase_counts.get("B", 0)),
                "I": int(phrase_counts.get("I", 0)),
                "O": int(phrase_counts.get("O", 0)),
            },
            "sign_bio_share": _share(
                {
                    "B": int(sign_counts.get("B", 0)),
                    "I": int(sign_counts.get("I", 0)),
                    "O": int(sign_counts.get("O", 0)),
                }
            ),
            "phrase_bio_share": _share(
                {
                    "B": int(phrase_counts.get("B", 0)),
                    "I": int(phrase_counts.get("I", 0)),
                    "O": int(phrase_counts.get("O", 0)),
                }
            ),
            "avg_words_per_sample": float(np.mean(words_per_sample)) if words_per_sample else 0.0,
            "avg_pause_length": float(np.mean(pause_lengths)) if pause_lengths else 0.0,
        }

    stats["total"] = {
        "samples": int(sum(split_info["samples"] for split_info in stats["splits"].values())),
        "frames": int(sum(split_info["frames"] for split_info in stats["splits"].values())),
        "length": _sequence_stats(total_lengths),
        "sign_bio_counts": {
            "B": int(total_sign_counts.get("B", 0)),
            "I": int(total_sign_counts.get("I", 0)),
            "O": int(total_sign_counts.get("O", 0)),
        },
        "phrase_bio_counts": {
            "B": int(total_phrase_counts.get("B", 0)),
            "I": int(total_phrase_counts.get("I", 0)),
            "O": int(total_phrase_counts.get("O", 0)),
        },
        "avg_pause_length": float(np.mean(total_pauses)) if total_pauses else 0.0,
    }

    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    stats = build_dataset(args)
    print("[build_bio_continuous_dataset] done")
    print(f"  out_dir={Path(args.out_dir).resolve()}")
    for split_name in ("train", "val", "test"):
        split_stats = stats["splits"].get(split_name, {})
        print(
            f"  {split_name}: clips={split_stats.get('clips', 0)} "
            f"samples={split_stats.get('samples', 0)} "
            f"frames={split_stats.get('frames', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

