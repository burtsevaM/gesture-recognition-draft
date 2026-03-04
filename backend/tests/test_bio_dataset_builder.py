from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import build_bio_continuous_dataset as bio_builder


def _write_clip(path: Path, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, features=features.astype(np.float32))


def _load_meta(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as payload:
        raw = payload["meta"]
    if isinstance(raw, np.ndarray):
        raw = raw.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return json.loads(str(raw))


def test_bio_builder_generates_valid_boundaries(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    index_path = tmp_path / "index.jsonl"
    out_dir = tmp_path / "out"

    records = []
    signer_ids = ["s1", "s2", "s3"]
    labels = ["привет", "мир", "да", "нет"]
    for i in range(12):
        label = labels[i % len(labels)]
        signer_id = signer_ids[i % len(signer_ids)]
        clip_path = clips_dir / f"{label}_{i:03d}.npz"
        frames = 4 + (i % 3)
        feature_dim = 8
        features = np.linspace(0.0, 1.0, num=frames * feature_dim, dtype=np.float32).reshape(frames, feature_dim)
        _write_clip(clip_path, features)
        records.append(
            {
                "path": str(clip_path.relative_to(tmp_path)),
                "label": label,
                "signer_id": signer_id,
                "fps": 25,
            }
        )

    with index_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    exit_code = bio_builder.main(
        [
            "--pose_index",
            str(index_path),
            "--out_dir",
            str(out_dir),
            "--min_words",
            "3",
            "--max_words",
            "3",
            "--pause_min",
            "2",
            "--pause_max",
            "2",
            "--phrase_min_words",
            "2",
            "--phrase_max_words",
            "2",
            "--seed",
            "7",
            "--splits_by_signer",
            "false",
            "--resample_fps",
            "25",
            "--use_shoulder_norm",
            "false",
            "--use_hands_3d_norm",
            "false",
            "--train_samples",
            "2",
            "--val_samples",
            "1",
            "--test_samples",
            "1",
        ]
    )
    assert exit_code == 0

    all_npz = sorted((out_dir / "train").glob("*.npz")) + sorted((out_dir / "val").glob("*.npz")) + sorted((out_dir / "test").glob("*.npz"))
    assert all_npz, "expected generated npz files"

    for sample_path in all_npz:
        with np.load(sample_path, allow_pickle=True) as payload:
            features = payload["features"]
            sign_bio = payload["sign_bio"]
            phrase_bio = payload["phrase_bio"]

        meta = _load_meta(sample_path)
        assert len(features) == len(sign_bio) == len(phrase_bio)

        for word in meta["words"]:
            start = int(word["start"])
            end = int(word["end"])
            assert sign_bio[start] == bio_builder.BIO_B
            if end > start:
                assert np.all(sign_bio[start + 1 : end + 1] == bio_builder.BIO_I)

            if bool(word["is_phrase_start"]):
                assert phrase_bio[start] == bio_builder.BIO_B
            else:
                assert phrase_bio[start] == bio_builder.BIO_I

        for pause in meta["pauses"]:
            start = int(pause["start"])
            end = int(pause["end"])
            assert np.all(sign_bio[start : end + 1] == bio_builder.BIO_O)
            expected_phrase = bio_builder.BIO_I if pause["phrase_value"] == "I" else bio_builder.BIO_O
            assert np.all(phrase_bio[start : end + 1] == expected_phrase)

    stats_path = out_dir / "dataset_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["bio_mapping"] == {"B": 0, "I": 1, "O": 2}
    assert "splits" in stats

