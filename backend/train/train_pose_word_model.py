#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.pose_words.segment_utils import resample_to_fixed_T  # noqa: E402
from train.pose_word_arch import PoseWordClassifier  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except Exception:
        return str(path.resolve())


@dataclass(slots=True)
class Batch:
    features: torch.Tensor
    labels: torch.Tensor


class PoseWordDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        dataset_root: Path,
        labels: list[str],
        clip_frames: int,
        max_samples: int = 0,
    ) -> None:
        if max_samples > 0:
            rows = rows[: max_samples]
        if not rows:
            raise ValueError("empty split rows")

        self.rows = rows
        self.dataset_root = dataset_root
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.clip_frames = int(max(4, clip_frames))

        first = self._load_features(self._resolve_path(self.rows[0]))
        self.feature_dim = int(first.shape[1])

    def _resolve_path(self, row: dict[str, Any]) -> Path:
        raw_path = str(row.get("path") or row.get("features_path") or "").strip()
        if not raw_path:
            raise ValueError("index row is missing 'path'")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self.dataset_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"feature file not found: {path}")
        return path

    @staticmethod
    def _load_features(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=True) as data:
                if "features" not in data.files:
                    raise ValueError(f"npz must contain 'features': {path}")
                features = np.asarray(data["features"], dtype=np.float32)
        elif path.suffix.lower() == ".npy":
            features = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
        else:
            raise ValueError(f"unsupported sample file type: {path}")

        if features.ndim != 2:
            raise ValueError(f"features must have shape [T,F], got {features.shape} in {path}")
        if features.shape[0] < 1:
            raise ValueError(f"empty feature sequence in {path}")
        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return features.astype(np.float32)

    def _target_from_row(self, row: dict[str, Any]) -> int:
        if "label_idx" in row:
            return int(row["label_idx"])
        label = str(row.get("label") or "").strip()
        if not label:
            raise ValueError("row must contain either label_idx or label")
        if label not in self.label_to_idx:
            raise ValueError(f"unknown label '{label}'")
        return int(self.label_to_idx[label])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        features = self._load_features(self._resolve_path(row))
        features = resample_to_fixed_T(features, T=self.clip_frames, method="linear")
        if int(features.shape[1]) != self.feature_dim:
            raise ValueError(
                f"feature dim mismatch in sample {row.get('sample_id', idx)}: expected {self.feature_dim}, got {features.shape[1]}"
            )
        target = self._target_from_row(row)
        return {
            "features": features.astype(np.float32),
            "target": int(target),
        }



def collate_batch(items: list[dict[str, Any]]) -> Batch:
    features = np.stack([np.asarray(item["features"], dtype=np.float32) for item in items], axis=0)
    labels = np.asarray([int(item["target"]) for item in items], dtype=np.int64)
    return Batch(
        features=torch.from_numpy(features),
        labels=torch.from_numpy(labels),
    )



def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train pose-word classifier (1D-CNN + BiGRU).")
    parser.add_argument("--dataset-index", type=Path, required=True, help="Path to index.jsonl from slovo_to_pose_dataset.py")
    parser.add_argument("--output-dir", type=Path, default=Path("backend/artifacts/pose_word_training"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-frames", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--conv-channels", type=int, default=192)
    parser.add_argument("--gru-hidden", type=int, default=192)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--save-last", type=parse_bool, default=True)
    parser.add_argument("--artifact-kind", type=str, default="runtime", help="dummy|validation|runtime")
    parser.add_argument("--dataset-kind", type=str, default="unknown", help="synthetic_fixture|mini_validation|real_dataset|unknown")
    parser.add_argument("--source-pipeline", type=str, default="train_pose_word_model", help="Marker for produced checkpoint lineage")
    return parser



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def select_device(raw: str) -> torch.device:
    text = str(raw).strip().lower()
    if text in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)



def load_index_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset index not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                rows.append(row)
    if not rows:
        raise ValueError(f"index is empty: {path}")
    return rows



def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    split_map: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = str(row.get("split") or "train").strip().lower()
        if split == "valid":
            split = "val"
        if split not in split_map:
            split = "train"
        split_map[split].append(row)

    if not split_map["train"]:
        raise ValueError("train split is empty")
    if not split_map["val"]:
        # fallback for tiny datasets
        split_map["val"] = list(split_map["train"][: max(1, min(32, len(split_map["train"])) )])
    return split_map



def load_labels(rows: list[dict[str, Any]], dataset_root: Path) -> list[str]:
    labels_path = dataset_root / "labels.txt"
    if labels_path.exists():
        labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if labels:
            return labels

    labels = sorted({str(row.get("label") or "").strip() for row in rows if str(row.get("label") or "").strip()})
    if not labels:
        raise ValueError("cannot infer labels list from index rows")
    return labels



def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    k_val = max(1, min(int(k), int(logits.shape[1])))
    topk = torch.topk(logits, k=k_val, dim=1).indices
    correct = topk.eq(targets.view(-1, 1)).any(dim=1)
    return float(correct.float().mean().item())



def evaluate(
    model: PoseWordClassifier,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    criterion: nn.Module,
    labels: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x = batch.features.to(device)
            y = batch.labels.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            batch_size = int(y.shape[0])
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    logits_cpu = torch.cat(all_logits, dim=0)
    targets_cpu = torch.cat(all_targets, dim=0)

    top1 = accuracy_topk(logits_cpu, targets_cpu, k=1)
    top5 = accuracy_topk(logits_cpu, targets_cpu, k=5)

    preds = torch.argmax(logits_cpu, dim=1)
    num_classes = len(labels)
    per_class_correct = np.zeros((num_classes,), dtype=np.int64)
    per_class_total = np.zeros((num_classes,), dtype=np.int64)

    for cls_idx in range(num_classes):
        mask = targets_cpu.numpy() == cls_idx
        per_class_total[cls_idx] = int(mask.sum())
        if per_class_total[cls_idx] == 0:
            continue
        per_class_correct[cls_idx] = int((preds.numpy()[mask] == cls_idx).sum())

    per_class = {
        labels[i]: {
            "correct": int(per_class_correct[i]),
            "total": int(per_class_total[i]),
            "accuracy": float(per_class_correct[i] / per_class_total[i]) if per_class_total[i] > 0 else 0.0,
        }
        for i in range(num_classes)
    }

    return {
        "loss": float(total_loss / max(1, total_samples)),
        "top1": float(top1),
        "top5": float(top5),
        "samples": int(total_samples),
        "per_class": per_class,
    }



def train_one_epoch(
    model: PoseWordClassifier,
    loader: DataLoader[Batch],
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for batch in loader:
        x = batch.features.to(device)
        y = batch.labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_top1 += accuracy_topk(logits.detach(), y.detach(), k=1) * batch_size
        total_top5 += accuracy_topk(logits.detach(), y.detach(), k=5) * batch_size

    return {
        "loss": float(total_loss / max(1, total_samples)),
        "top1": float(total_top1 / max(1, total_samples)),
        "top5": float(total_top5 / max(1, total_samples)),
    }



def build_checkpoint(
    *,
    model: PoseWordClassifier,
    labels: list[str],
    clip_frames: int,
    dataset_flags: dict[str, Any],
    args: argparse.Namespace,
    best_val: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "labels": labels,
        "model_config": {
            "input_dim": int(model.input_dim),
            "num_classes": int(model.num_classes),
            "conv_channels": int(model.conv_channels),
            "gru_hidden": int(model.gru_hidden),
            "gru_layers": int(model.gru_layers),
            "dropout": float(model.dropout),
            "clip_frames": int(clip_frames),
        },
        "dataset_flags": dataset_flags,
        "artifact_kind": str(args.artifact_kind).strip() or "runtime",
        "dataset_kind": str(args.dataset_kind).strip() or "unknown",
        "trained": True,
        "source_pipeline": str(args.source_pipeline).strip() or "train_pose_word_model",
        "train_args": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "best_val": best_val,
        "saved_at": float(time.time()),
    }



def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = select_device(str(args.device))

    dataset_index = Path(args.dataset_index).resolve()
    dataset_root = dataset_index.parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT_DIR / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_index_rows(dataset_index)
    splits = split_rows(rows)
    labels = load_labels(rows, dataset_root)

    dataset_flags: dict[str, Any] = {}
    stats_path = dataset_root / "dataset_stats.json"
    if stats_path.exists():
        try:
            dataset_flags = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            dataset_flags = {}

    train_ds = PoseWordDataset(
        splits["train"],
        dataset_root=dataset_root,
        labels=labels,
        clip_frames=args.clip_frames,
        max_samples=int(args.max_train_samples),
    )
    val_ds = PoseWordDataset(
        splits["val"],
        dataset_root=dataset_root,
        labels=labels,
        clip_frames=args.clip_frames,
        max_samples=int(args.max_val_samples),
    )
    test_ds = (
        PoseWordDataset(
            splits["test"],
            dataset_root=dataset_root,
            labels=labels,
            clip_frames=args.clip_frames,
            max_samples=int(args.max_test_samples),
        )
        if splits["test"]
        else None
    )

    feature_dim = int(train_ds.feature_dim)
    if int(val_ds.feature_dim) != feature_dim:
        raise ValueError(f"feature dim mismatch train/val: {feature_dim} vs {val_ds.feature_dim}")
    if test_ds is not None and int(test_ds.feature_dim) != feature_dim:
        raise ValueError(f"feature dim mismatch test: {feature_dim} vs {test_ds.feature_dim}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=collate_batch,
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            collate_fn=collate_batch,
        )
        if test_ds is not None
        else None
    )

    model = PoseWordClassifier(
        input_dim=feature_dim,
        num_classes=len(labels),
        conv_channels=int(args.conv_channels),
        gru_hidden=int(args.gru_hidden),
        gru_layers=int(args.gru_layers),
        dropout=float(args.dropout),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    history: list[dict[str, Any]] = []
    best_val_top1 = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )
        val_metrics = evaluate(model, val_loader, device=device, criterion=criterion, labels=labels)

        epoch_row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": {k: v for k, v in val_metrics.items() if k != "per_class"},
        }
        history.append(epoch_row)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} train_top5={train_metrics['top5']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} val_top5={val_metrics['top5']:.4f}"
        )

        if float(val_metrics["top1"]) > best_val_top1:
            best_val_top1 = float(val_metrics["top1"])
            best_state = build_checkpoint(
                model=model,
                labels=labels,
                clip_frames=int(args.clip_frames),
                dataset_flags=dataset_flags,
                args=args,
                best_val=val_metrics,
            )
            torch.save(best_state, output_dir / "pose_word_best.pt")

    if best_state is None:
        raise RuntimeError("training failed: no best checkpoint captured")

    model.load_state_dict(best_state["model_state"])
    val_final = evaluate(model, val_loader, device=device, criterion=criterion, labels=labels)
    test_final = (
        evaluate(model, test_loader, device=device, criterion=criterion, labels=labels)
        if test_loader is not None
        else None
    )

    summary = {
        "dataset_index": _display_path(dataset_index),
        "dataset_root": _display_path(dataset_root),
        "feature_dim": int(feature_dim),
        "clip_frames": int(args.clip_frames),
        "labels": labels,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "device": str(device),
        "artifact_kind": str(args.artifact_kind).strip() or "runtime",
        "dataset_kind": str(args.dataset_kind).strip() or "unknown",
        "trained": True,
        "source_pipeline": str(args.source_pipeline).strip() or "train_pose_word_model",
        "model": {
            "conv_channels": int(args.conv_channels),
            "gru_hidden": int(args.gru_hidden),
            "gru_layers": int(args.gru_layers),
            "dropout": float(args.dropout),
        },
        "best_val_top1": float(best_val_top1),
        "val": val_final,
        "test": test_final,
        "history": history,
        "dataset_flags": dataset_flags,
    }

    (output_dir / "training_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.save_last):
        last_state = build_checkpoint(
            model=model,
            labels=labels,
            clip_frames=int(args.clip_frames),
            dataset_flags=dataset_flags,
            args=args,
            best_val=val_final,
        )
        torch.save(last_state, output_dir / "pose_word_last.pt")

    print("[train_pose_word_model] done")
    print(f"  best_checkpoint={output_dir / 'pose_word_best.pt'}")
    print(f"  best_val_top1={best_val_top1:.4f}")
    print(f"  val_top5={val_final['top5']:.4f}")
    if test_final is not None:
        print(f"  test_top1={float(test_final['top1']):.4f} test_top5={float(test_final['top5']):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
