from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from .bio_decode_eval import BIO_MAPPING, evaluate_prob_sequences
except ImportError:  # pragma: no cover - script mode fallback
    from bio_decode_eval import BIO_MAPPING, evaluate_prob_sequences

IGNORE_INDEX = -100


@dataclass(slots=True)
class Batch:
    features: torch.Tensor
    sign_labels: torch.Tensor
    phrase_labels: torch.Tensor
    lengths: torch.Tensor
    sample_ids: list[str]


@dataclass(slots=True)
class ThresholdResult:
    th_b: float
    th_o: float
    boundary_f1: float
    segment_f1: float
    segment_mean_iou: float
    metrics: dict[str, float | int]


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool value, got '{value}'")


def parse_weights(raw: str) -> np.ndarray:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError("class weights must have 3 comma-separated values, e.g. '5,1,1'")
    arr = np.asarray([float(x) for x in parts], dtype=np.float32)
    if np.any(arr <= 0):
        raise ValueError("class weights must be > 0")
    return arr


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NPZSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, files: list[Path]) -> None:
        self.files = sorted(files)
        if not self.files:
            raise ValueError("dataset split is empty")

        first = self._load_file(self.files[0])
        self.feature_dim = int(first["features"].shape[1])

    @staticmethod
    def _load_file(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=True) as data:
            features = np.asarray(data["features"], dtype=np.float32)
            sign = np.asarray(data["sign_bio"], dtype=np.int64).reshape(-1)
            phrase = np.asarray(data["phrase_bio"], dtype=np.int64).reshape(-1)

        if features.ndim != 2:
            raise ValueError(f"features must have shape [T,F], got {features.shape} in {path}")
        if len(features) != len(sign) or len(features) != len(phrase):
            raise ValueError(f"length mismatch in {path}: features={len(features)} sign={len(sign)} phrase={len(phrase)}")
        if len(features) < 1:
            raise ValueError(f"empty sequence in {path}")
        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return {
            "features": features.astype(np.float32),
            "sign": sign.astype(np.int64),
            "phrase": phrase.astype(np.int64),
        }

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.files[idx]
        item = self._load_file(path)
        return {
            "features": item["features"],
            "sign": item["sign"],
            "phrase": item["phrase"],
            "sample_id": path.stem,
        }


class PoseBioSegmenter(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=lstm_dropout,
        )
        out_dim = int(hidden_size) * (2 if bidirectional else 1)
        self.sign_head = nn.Linear(out_dim, 3)
        self.phrase_head = nn.Linear(out_dim, 3)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if lengths is not None:
            packed = pack_padded_sequence(
                features,
                lengths=lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=features.shape[1],
            )
        else:
            encoded, _ = self.encoder(features)

        sign_logits = self.sign_head(encoded)
        phrase_logits = self.phrase_head(encoded)
        return sign_logits, phrase_logits


def collate_batch(items: list[dict[str, Any]]) -> Batch:
    lengths = [int(item["features"].shape[0]) for item in items]
    max_len = max(lengths)
    feature_dim = int(items[0]["features"].shape[1])

    b = len(items)
    features = np.zeros((b, max_len, feature_dim), dtype=np.float32)
    sign = np.full((b, max_len), IGNORE_INDEX, dtype=np.int64)
    phrase = np.full((b, max_len), IGNORE_INDEX, dtype=np.int64)
    sample_ids: list[str] = []

    for i, item in enumerate(items):
        length = lengths[i]
        features[i, :length, :] = item["features"]
        sign[i, :length] = item["sign"]
        phrase[i, :length] = item["phrase"]
        sample_ids.append(str(item["sample_id"]))

    return Batch(
        features=torch.from_numpy(features),
        sign_labels=torch.from_numpy(sign),
        phrase_labels=torch.from_numpy(phrase),
        lengths=torch.tensor(lengths, dtype=torch.long),
        sample_ids=sample_ids,
    )


def list_split_files(data_dir: Path, split_name: str) -> list[Path]:
    split_dir = data_dir / split_name
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("*.npz"))


def label_counts(files: list[Path], key: str) -> np.ndarray:
    counts = np.zeros((3,), dtype=np.float64)
    for file_path in files:
        with np.load(file_path, allow_pickle=True) as data:
            labels = np.asarray(data[key], dtype=np.int64).reshape(-1)
        for cls_idx in (0, 1, 2):
            counts[cls_idx] += int(np.sum(labels == cls_idx))
    return counts


def class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    if counts.shape != (3,):
        raise ValueError(f"counts must have shape (3,), got {counts.shape}")
    safe = np.maximum(counts, 1.0)
    total = float(np.sum(safe))
    inv = total / safe
    inv = inv / np.min(inv)
    return inv.astype(np.float32)


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, int]:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        mask = labels != IGNORE_INDEX
        total = int(mask.sum().item())
        if total == 0:
            return 0.0, 0
        correct = int(((pred == labels) & mask).sum().item())
        return float(correct / total), total


def select_device(raw: str) -> torch.device:
    normalized = str(raw).strip().lower()
    if normalized in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def tune_thresholds(
    probs_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    *,
    grid: list[float],
    boundary_tolerance: int,
    segment_iou_threshold: float,
    min_segment_len: int,
) -> ThresholdResult:
    best: ThresholdResult | None = None

    for th_b in grid:
        for th_o in grid:
            metrics = evaluate_prob_sequences(
                probs_list,
                labels_list,
                th_b=float(th_b),
                th_o=float(th_o),
                boundary_tolerance=int(boundary_tolerance),
                segment_iou_threshold=float(segment_iou_threshold),
                min_segment_len=int(min_segment_len),
            )
            current = ThresholdResult(
                th_b=float(th_b),
                th_o=float(th_o),
                boundary_f1=float(metrics["boundary_f1"]),
                segment_f1=float(metrics["segment_f1"]),
                segment_mean_iou=float(metrics["segment_mean_iou"]),
                metrics=metrics,
            )
            if best is None:
                best = current
                continue

            better = False
            if current.boundary_f1 > best.boundary_f1 + 1e-9:
                better = True
            elif abs(current.boundary_f1 - best.boundary_f1) <= 1e-9:
                if current.segment_f1 > best.segment_f1 + 1e-9:
                    better = True
                elif abs(current.segment_f1 - best.segment_f1) <= 1e-9:
                    if current.segment_mean_iou > best.segment_mean_iou + 1e-9:
                        better = True

            if better:
                best = current

    if best is None:
        raise RuntimeError("failed to tune thresholds")
    return best


def run_epoch_train(
    model: PoseBioSegmenter,
    loader: DataLoader[Batch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sign_loss_fn: nn.Module,
    phrase_loss_fn: nn.Module,
    alpha_phrase: float,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_sign_acc = 0.0
    total_phrase_acc = 0.0
    total_sign_frames = 0
    total_phrase_frames = 0

    for batch in loader:
        x = batch.features.to(device)
        sign_y = batch.sign_labels.to(device)
        phrase_y = batch.phrase_labels.to(device)
        lengths = batch.lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        sign_logits, phrase_logits = model(x, lengths)

        loss_sign = sign_loss_fn(sign_logits.reshape(-1, 3), sign_y.reshape(-1))
        loss_phrase = phrase_loss_fn(phrase_logits.reshape(-1, 3), phrase_y.reshape(-1))
        loss = loss_sign + float(alpha_phrase) * loss_phrase
        loss.backward()
        optimizer.step()

        sign_acc, sign_frames = masked_accuracy(sign_logits, sign_y)
        phrase_acc, phrase_frames = masked_accuracy(phrase_logits, phrase_y)

        total_loss += float(loss.item())
        total_sign_acc += sign_acc * sign_frames
        total_phrase_acc += phrase_acc * phrase_frames
        total_sign_frames += sign_frames
        total_phrase_frames += phrase_frames

    n_batches = max(1, len(loader))
    return {
        "loss": total_loss / n_batches,
        "sign_frame_acc": float(total_sign_acc / total_sign_frames) if total_sign_frames > 0 else 0.0,
        "phrase_frame_acc": float(total_phrase_acc / total_phrase_frames) if total_phrase_frames > 0 else 0.0,
    }


def run_epoch_eval(
    model: PoseBioSegmenter,
    loader: DataLoader[Batch],
    device: torch.device,
    sign_loss_fn: nn.Module,
    phrase_loss_fn: nn.Module,
    alpha_phrase: float,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    model.eval()
    total_loss = 0.0
    total_sign_acc = 0.0
    total_phrase_acc = 0.0
    total_sign_frames = 0
    total_phrase_frames = 0

    sign_probs_all: list[np.ndarray] = []
    phrase_probs_all: list[np.ndarray] = []
    sign_labels_all: list[np.ndarray] = []
    phrase_labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = batch.features.to(device)
            sign_y = batch.sign_labels.to(device)
            phrase_y = batch.phrase_labels.to(device)
            lengths = batch.lengths.to(device)

            sign_logits, phrase_logits = model(x, lengths)
            loss_sign = sign_loss_fn(sign_logits.reshape(-1, 3), sign_y.reshape(-1))
            loss_phrase = phrase_loss_fn(phrase_logits.reshape(-1, 3), phrase_y.reshape(-1))
            loss = loss_sign + float(alpha_phrase) * loss_phrase

            sign_acc, sign_frames = masked_accuracy(sign_logits, sign_y)
            phrase_acc, phrase_frames = masked_accuracy(phrase_logits, phrase_y)

            total_loss += float(loss.item())
            total_sign_acc += sign_acc * sign_frames
            total_phrase_acc += phrase_acc * phrase_frames
            total_sign_frames += sign_frames
            total_phrase_frames += phrase_frames

            sign_probs = F.softmax(sign_logits, dim=-1).cpu().numpy().astype(np.float32)
            phrase_probs = F.softmax(phrase_logits, dim=-1).cpu().numpy().astype(np.float32)
            sign_np = sign_y.cpu().numpy().astype(np.int64)
            phrase_np = phrase_y.cpu().numpy().astype(np.int64)
            lengths_np = lengths.cpu().numpy().astype(np.int64)

            for i, seq_len in enumerate(lengths_np.tolist()):
                l = int(seq_len)
                sign_probs_all.append(sign_probs[i, :l, :])
                phrase_probs_all.append(phrase_probs[i, :l, :])
                sign_labels_all.append(sign_np[i, :l])
                phrase_labels_all.append(phrase_np[i, :l])

    n_batches = max(1, len(loader))
    summary = {
        "loss": total_loss / n_batches,
        "sign_frame_acc": float(total_sign_acc / total_sign_frames) if total_sign_frames > 0 else 0.0,
        "phrase_frame_acc": float(total_phrase_acc / total_phrase_frames) if total_phrase_frames > 0 else 0.0,
    }
    return summary, sign_probs_all, phrase_probs_all, sign_labels_all, phrase_labels_all


def parse_grid(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise ValueError("threshold grid is empty")
    return values


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BiLSTM BIO segmenter for pose features.")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val/test *.npz")
    parser.add_argument("--save-dir", default="backend/artifacts/bio_training", help="Directory for checkpoints and logs")
    parser.add_argument("--thresholds-out", default="backend/artifacts/bio_thresholds.json", help="Path for best thresholds JSON")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=parse_bool, default=True)
    parser.add_argument("--alpha-phrase", type=float, default=1.0)
    parser.add_argument("--auto-class-weights", type=parse_bool, default=True)
    parser.add_argument("--default-sign-weights", default="5,1,1")
    parser.add_argument("--default-phrase-weights", default="5,1,1")
    parser.add_argument("--threshold-grid", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--boundary-tolerance", type=int, default=0)
    parser.add_argument("--segment-iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-segment-len", type=int, default=1)
    parser.add_argument("--use-shoulder-norm", type=parse_bool, default=True)
    parser.add_argument("--use-hands-3d-norm", type=parse_bool, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(int(args.seed))
    device = select_device(args.device)

    data_dir = Path(args.data_dir).resolve()
    save_dir = Path(args.save_dir).resolve()
    thresholds_out = Path(args.thresholds_out).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_split_files(data_dir, "train")
    val_files = list_split_files(data_dir, "val")
    test_files = list_split_files(data_dir, "test")
    if not train_files or not val_files:
        raise ValueError("train and val splits must not be empty")

    train_ds = NPZSequenceDataset(train_files)
    val_ds = NPZSequenceDataset(val_files)
    test_ds = NPZSequenceDataset(test_files) if test_files else None

    if val_ds.feature_dim != train_ds.feature_dim:
        raise ValueError("feature dimension mismatch between train and val")
    if test_ds is not None and test_ds.feature_dim != train_ds.feature_dim:
        raise ValueError("feature dimension mismatch between train and test")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
        drop_last=False,
    )
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=collate_batch,
            drop_last=False,
        )

    if bool(args.auto_class_weights):
        sign_counts = label_counts(train_files, "sign_bio")
        phrase_counts = label_counts(train_files, "phrase_bio")
        sign_weights_np = class_weights_from_counts(sign_counts)
        phrase_weights_np = class_weights_from_counts(phrase_counts)
    else:
        sign_weights_np = parse_weights(args.default_sign_weights)
        phrase_weights_np = parse_weights(args.default_phrase_weights)

    model = PoseBioSegmenter(
        input_dim=train_ds.feature_dim,
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        bidirectional=bool(args.bidirectional),
    ).to(device)

    sign_weight_tensor = torch.tensor(sign_weights_np, dtype=torch.float32, device=device)
    phrase_weight_tensor = torch.tensor(phrase_weights_np, dtype=torch.float32, device=device)

    sign_loss_fn = nn.CrossEntropyLoss(weight=sign_weight_tensor, ignore_index=IGNORE_INDEX)
    phrase_loss_fn = nn.CrossEntropyLoss(weight=phrase_weight_tensor, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    grid = parse_grid(args.threshold_grid)
    best_key = -1.0
    best_epoch = 0
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    started_at = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch_train(
            model,
            train_loader,
            optimizer,
            device,
            sign_loss_fn,
            phrase_loss_fn,
            alpha_phrase=float(args.alpha_phrase),
        )

        val_metrics, sign_probs_val, phrase_probs_val, sign_labels_val, phrase_labels_val = run_epoch_eval(
            model,
            val_loader,
            device,
            sign_loss_fn,
            phrase_loss_fn,
            alpha_phrase=float(args.alpha_phrase),
        )

        sign_thr = tune_thresholds(
            sign_probs_val,
            sign_labels_val,
            grid=grid,
            boundary_tolerance=int(args.boundary_tolerance),
            segment_iou_threshold=float(args.segment_iou_threshold),
            min_segment_len=int(args.min_segment_len),
        )
        phrase_thr = tune_thresholds(
            phrase_probs_val,
            phrase_labels_val,
            grid=grid,
            boundary_tolerance=int(args.boundary_tolerance),
            segment_iou_threshold=float(args.segment_iou_threshold),
            min_segment_len=int(args.min_segment_len),
        )

        summary = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
            "val_sign": sign_thr.metrics,
            "val_phrase": phrase_thr.metrics,
            "thresholds": {
                "sign": {"th_b": sign_thr.th_b, "th_o": sign_thr.th_o},
                "phrase": {"th_b": phrase_thr.th_b, "th_o": phrase_thr.th_o},
            },
        }
        history.append(summary)

        epoch_key = float(sign_thr.boundary_f1 + phrase_thr.boundary_f1)
        if epoch_key > best_key:
            best_key = epoch_key
            best_epoch = int(epoch)
            best_payload = summary
            checkpoint = {
                "epoch": int(epoch),
                "model_state": model.state_dict(),
                "feature_dim": int(train_ds.feature_dim),
                "model_config": {
                    "hidden_size": int(args.hidden_size),
                    "num_layers": int(args.num_layers),
                    "dropout": float(args.dropout),
                    "bidirectional": bool(args.bidirectional),
                },
                "train_config": {
                    "alpha_phrase": float(args.alpha_phrase),
                    "seed": int(args.seed),
                    "threshold_grid": grid,
                    "boundary_tolerance": int(args.boundary_tolerance),
                    "segment_iou_threshold": float(args.segment_iou_threshold),
                    "min_segment_len": int(args.min_segment_len),
                    "use_shoulder_norm": bool(args.use_shoulder_norm),
                    "use_hands_3d_norm": bool(args.use_hands_3d_norm),
                },
                "class_weights": {
                    "sign": sign_weights_np.tolist(),
                    "phrase": phrase_weights_np.tolist(),
                },
                "best_thresholds": summary["thresholds"],
                "bio_mapping": BIO_MAPPING,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"sign_frame_acc={val_metrics['sign_frame_acc']:.4f} "
            f"phrase_frame_acc={val_metrics['phrase_frame_acc']:.4f} "
            f"sign_BF1={sign_thr.boundary_f1:.4f} phrase_BF1={phrase_thr.boundary_f1:.4f}"
        )

    if best_payload is None:
        raise RuntimeError("training failed: no best checkpoint produced")

    ckpt_path = save_dir / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_report: dict[str, Any] | None = None
    if test_loader is not None:
        test_metrics, sign_probs_test, phrase_probs_test, sign_labels_test, phrase_labels_test = run_epoch_eval(
            model,
            test_loader,
            device,
            sign_loss_fn,
            phrase_loss_fn,
            alpha_phrase=float(args.alpha_phrase),
        )
        sign_best = best_payload["thresholds"]["sign"]
        phrase_best = best_payload["thresholds"]["phrase"]
        sign_eval = evaluate_prob_sequences(
            sign_probs_test,
            sign_labels_test,
            th_b=float(sign_best["th_b"]),
            th_o=float(sign_best["th_o"]),
            boundary_tolerance=int(args.boundary_tolerance),
            segment_iou_threshold=float(args.segment_iou_threshold),
            min_segment_len=int(args.min_segment_len),
        )
        phrase_eval = evaluate_prob_sequences(
            phrase_probs_test,
            phrase_labels_test,
            th_b=float(phrase_best["th_b"]),
            th_o=float(phrase_best["th_o"]),
            boundary_tolerance=int(args.boundary_tolerance),
            segment_iou_threshold=float(args.segment_iou_threshold),
            min_segment_len=int(args.min_segment_len),
        )
        test_report = {
            "frame": test_metrics,
            "sign": sign_eval,
            "phrase": phrase_eval,
        }

    thresholds_payload = {
        "bio_mapping": BIO_MAPPING,
        "best_epoch": int(best_epoch),
        "sign": best_payload["thresholds"]["sign"],
        "phrase": best_payload["thresholds"]["phrase"],
        "val_sign": best_payload["val_sign"],
        "val_phrase": best_payload["val_phrase"],
        "boundary_tolerance": int(args.boundary_tolerance),
        "segment_iou_threshold": float(args.segment_iou_threshold),
        "min_segment_len": int(args.min_segment_len),
        "created_at_unix": float(time.time()),
    }
    save_json(thresholds_out, thresholds_payload)
    save_json(save_dir / "bio_thresholds.json", thresholds_payload)

    summary_payload = {
        "config": {
            "data_dir": str(data_dir),
            "save_dir": str(save_dir),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "hidden_size": int(args.hidden_size),
            "num_layers": int(args.num_layers),
            "dropout": float(args.dropout),
            "bidirectional": bool(args.bidirectional),
            "alpha_phrase": float(args.alpha_phrase),
            "use_shoulder_norm": bool(args.use_shoulder_norm),
            "use_hands_3d_norm": bool(args.use_hands_3d_norm),
            "seed": int(args.seed),
            "device": str(device),
        },
        "feature_dim": int(train_ds.feature_dim),
        "class_weights": {
            "sign": sign_weights_np.tolist(),
            "phrase": phrase_weights_np.tolist(),
        },
        "best_epoch": int(best_epoch),
        "best": best_payload,
        "history": history,
        "test": test_report,
        "elapsed_sec": float(time.time() - started_at),
    }
    save_json(save_dir / "train_summary.json", summary_payload)

    print("[train_pose_bio_segmenter] done")
    print(f"  best_epoch={best_epoch}")
    print(f"  checkpoint={ckpt_path}")
    print(f"  thresholds={thresholds_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
