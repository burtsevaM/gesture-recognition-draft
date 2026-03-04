from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class BioThresholdConfig:
    sign_th_b: float = 0.5
    sign_th_o: float = 0.5
    phrase_th_b: float = 0.5
    phrase_th_o: float = 0.5


def _to_probs(logits_or_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits_or_probs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"BIO output must have shape [T, 3], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    sums = arr.sum(axis=1)
    if np.all((arr >= 0.0) & (arr <= 1.0)) and np.allclose(sums, 1.0, atol=1e-2):
        return arr

    max_v = np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr - max_v)
    denom = np.maximum(exp.sum(axis=1, keepdims=True), 1e-9)
    return (exp / denom).astype(np.float32)


class BioSegmenterOnnxModel:
    def __init__(self, *, model_path: str | Path, ort_num_threads: int = 1) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"BIO segmenter ONNX not found: {self.model_path}")

        self.ort_num_threads = max(1, int(ort_num_threads))
        self._session: Any = None
        self._input_name = ""
        self._sign_output_name = ""
        self._phrase_output_name = ""
        self.input_feature_dim: int | None = None
        self._init_session()

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("onnxruntime is required for BIO segmenter inference") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.ort_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = tuple(model_input.shape)
        if len(shape) != 3:
            raise ValueError(f"BIO segmenter input must be rank-3 [B,T,F], got {shape}")
        feature_dim = shape[-1]
        if isinstance(feature_dim, int) and feature_dim > 0:
            self.input_feature_dim = int(feature_dim)

        outputs = self._session.get_outputs()
        if len(outputs) < 2:
            raise ValueError("BIO segmenter ONNX must have two outputs: sign_probs, phrase_probs")
        self._sign_output_name = outputs[0].name
        self._phrase_output_name = outputs[1].name

    def infer(self, features_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        features = np.asarray(features_tf, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [T, F], got {features.shape}")
        if features.shape[0] < 1:
            raise ValueError("features must contain at least one frame")
        if self.input_feature_dim is not None and features.shape[1] != self.input_feature_dim:
            raise ValueError(
                f"BIO segmenter feature dim mismatch: expected {self.input_feature_dim}, got {features.shape[1]}"
            )

        x = np.expand_dims(features, axis=0).astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)
        started = time.perf_counter()
        sign_raw, phrase_raw = self._session.run(
            [self._sign_output_name, self._phrase_output_name],
            {self._input_name: x},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0

        sign = np.asarray(sign_raw, dtype=np.float32)
        phrase = np.asarray(phrase_raw, dtype=np.float32)
        if sign.ndim == 3:
            sign = sign[0]
        if phrase.ndim == 3:
            phrase = phrase[0]

        return _to_probs(sign), _to_probs(phrase), float(latency_ms)


class PoseWordOnnxModel:
    def __init__(
        self,
        *,
        model_path: str | Path,
        labels_path: str | Path,
        ort_num_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"pose word ONNX not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"pose word labels not found: {self.labels_path}")

        self.labels = self._load_labels(self.labels_path)
        if not self.labels:
            raise ValueError(f"labels file is empty: {self.labels_path}")

        self.ort_num_threads = max(1, int(ort_num_threads))
        self._session: Any = None
        self._input_name = ""
        self._output_name = ""
        self.input_feature_dim: int | None = None
        self._init_session()

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        labels: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                labels.append(text)
        return labels

    def _init_session(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise ImportError("onnxruntime is required for pose word inference") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.ort_num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self._session.get_inputs()[0]
        self._input_name = model_input.name
        shape = tuple(model_input.shape)
        if len(shape) != 3:
            raise ValueError(f"pose word ONNX input must be rank-3 [B,T,F], got {shape}")
        feature_dim = shape[-1]
        if isinstance(feature_dim, int) and feature_dim > 0:
            self.input_feature_dim = int(feature_dim)
        self._output_name = self._session.get_outputs()[0].name

    def find_no_event_index(self, label_name: str) -> int | None:
        wanted = str(label_name).strip().lower()
        aliases = {
            wanted,
            wanted.replace("-", "_"),
            wanted.replace("_", "-"),
            "_no_event",
            "no_event",
            "none",
            "---",
            "background",
        }
        for idx, label in enumerate(self.labels):
            normalized = label.strip().lower()
            if normalized in aliases:
                return int(idx)
        return None

    def infer_probs(self, features_tf: np.ndarray) -> tuple[np.ndarray, float]:
        features = np.asarray(features_tf, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [T, F], got {features.shape}")
        if features.shape[0] < 1:
            raise ValueError("features must contain at least one frame")
        if self.input_feature_dim is not None and features.shape[1] != self.input_feature_dim:
            raise ValueError(
                f"pose word feature dim mismatch: expected {self.input_feature_dim}, got {features.shape[1]}"
            )

        x = np.expand_dims(features, axis=0).astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)
        started = time.perf_counter()
        raw = self._session.run([self._output_name], {self._input_name: x})[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        out = np.asarray(raw, dtype=np.float32)
        if out.ndim == 3:
            # [B, T, C] -> mean over time for segment-level class posterior.
            out = out.mean(axis=1)
        if out.ndim == 2:
            out = out[0]
        if out.ndim != 1:
            raise ValueError(f"unsupported pose word output shape: {np.asarray(raw).shape}")
        if out.shape[0] != len(self.labels):
            raise ValueError(
                f"pose word output size {out.shape[0]} does not match labels size {len(self.labels)}"
            )

        probs = self._to_probs(out)
        return probs, float(latency_ms)

    @staticmethod
    def _to_probs(values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(v)):
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        total = float(v.sum())
        if float(v.min()) >= 0.0 and float(v.max()) <= 1.0 and math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return v
        m = float(np.max(v))
        exp = np.exp(v - m)
        denom = float(exp.sum())
        if denom <= 0:
            return np.zeros_like(v)
        return (exp / denom).astype(np.float32)


def load_bio_thresholds(path: str | Path) -> BioThresholdConfig:
    thresholds_path = Path(path)
    if not thresholds_path.exists():
        return BioThresholdConfig()

    try:
        payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    except Exception:
        return BioThresholdConfig()

    sign = payload.get("sign", {}) if isinstance(payload, dict) else {}
    phrase = payload.get("phrase", {}) if isinstance(payload, dict) else {}
    return BioThresholdConfig(
        sign_th_b=float(sign.get("th_b", 0.5)),
        sign_th_o=float(sign.get("th_o", 0.5)),
        phrase_th_b=float(phrase.get("th_b", 0.5)),
        phrase_th_o=float(phrase.get("th_o", 0.5)),
    )
