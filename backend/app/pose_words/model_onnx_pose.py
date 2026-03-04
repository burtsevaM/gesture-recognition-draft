from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


class PoseWordOnnxModel:
    """ONNX Runtime wrapper for pose-word classifier.

    Input: features [T, F]
    Output: probability vector [C]
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        labels_path: str | Path,
        config_path: str | Path | None = None,
        ort_num_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.config_path = Path(config_path) if config_path is not None else None
        if not self.model_path.exists():
            raise FileNotFoundError(f"pose word ONNX not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"pose word labels not found: {self.labels_path}")
        if self.config_path is not None and not self.config_path.exists():
            raise FileNotFoundError(f"pose word config not found: {self.config_path}")

        self.labels = self._load_labels(self.labels_path)
        if not self.labels:
            raise ValueError(f"labels file is empty: {self.labels_path}")

        self.runtime_config: dict[str, Any] = {}
        self.config_feature_dim: int | None = None
        self.config_clip_frames: int | None = None
        if self.config_path is not None:
            self.runtime_config = self._load_runtime_config(self.config_path)

        self.ort_num_threads = max(1, int(ort_num_threads))
        self._session: Any = None
        self._input_name = ""
        self._output_name = ""
        self.input_feature_dim: int | None = None
        self.input_clip_frames: int | None = None
        self._init_session()
        self._validate_against_runtime_config()

    @staticmethod
    def _load_labels(path: Path) -> list[str]:
        labels: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                labels.append(text)
        return labels

    def _load_runtime_config(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"failed to parse pose word config: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"pose word config must be JSON object: {path}")

        input_cfg = payload.get("input") if isinstance(payload.get("input"), dict) else {}
        shape = input_cfg.get("shape") if isinstance(input_cfg.get("shape"), list) else None
        cfg_clip = None
        cfg_feat = None
        if isinstance(shape, list) and len(shape) >= 3:
            if isinstance(shape[1], (int, float)):
                cfg_clip = int(shape[1])
            if isinstance(shape[2], (int, float)):
                cfg_feat = int(shape[2])

        if cfg_clip is None:
            if isinstance(payload.get("clip_frames"), (int, float)):
                cfg_clip = int(payload["clip_frames"])
            elif isinstance(payload.get("model"), dict) and isinstance(payload["model"].get("clip_frames"), (int, float)):
                cfg_clip = int(payload["model"]["clip_frames"])
        if cfg_feat is None:
            if isinstance(payload.get("input_dim"), (int, float)):
                cfg_feat = int(payload["input_dim"])
            elif isinstance(payload.get("model"), dict) and isinstance(payload["model"].get("input_dim"), (int, float)):
                cfg_feat = int(payload["model"]["input_dim"])

        self.config_clip_frames = cfg_clip if cfg_clip and cfg_clip > 0 else None
        self.config_feature_dim = cfg_feat if cfg_feat and cfg_feat > 0 else None
        return payload

    def _validate_against_runtime_config(self) -> None:
        if self.config_path is None:
            return

        expected_labels = self.runtime_config.get("labels_total")
        if isinstance(expected_labels, (int, float)) and int(expected_labels) > 0:
            if int(expected_labels) != len(self.labels):
                raise ValueError(
                    f"pose word labels size mismatch: config={int(expected_labels)} file={len(self.labels)}"
                )

        if self.config_clip_frames is not None and self.input_clip_frames is not None:
            if int(self.config_clip_frames) != int(self.input_clip_frames):
                raise ValueError(
                    "pose word clip length mismatch between config and ONNX: "
                    f"config={self.config_clip_frames}, onnx={self.input_clip_frames}"
                )
        if self.config_feature_dim is not None and self.input_feature_dim is not None:
            if int(self.config_feature_dim) != int(self.input_feature_dim):
                raise ValueError(
                    "pose word feature dim mismatch between config and ONNX: "
                    f"config={self.config_feature_dim}, onnx={self.input_feature_dim}"
                )

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
        clip_frames = shape[1]
        feature_dim = shape[2]
        if isinstance(clip_frames, int) and clip_frames > 0:
            self.input_clip_frames = int(clip_frames)
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
        if self.input_clip_frames is not None and features.shape[0] != self.input_clip_frames:
            raise ValueError(
                f"pose word clip length mismatch: expected {self.input_clip_frames}, got {features.shape[0]}"
            )
        if self.input_feature_dim is not None and features.shape[1] != self.input_feature_dim:
            raise ValueError(
                f"pose word feature dim mismatch: expected {self.input_feature_dim}, got {features.shape[1]}"
            )

        if not np.all(np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        x = np.expand_dims(features, axis=0).astype(np.float32, copy=False)
        x = np.ascontiguousarray(x)
        started = time.perf_counter()
        raw = self._session.run([self._output_name], {self._input_name: x})[0]
        latency_ms = (time.perf_counter() - started) * 1000.0

        out = np.asarray(raw, dtype=np.float32)
        if out.ndim == 3:
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
        max_v = float(np.max(v))
        exp = np.exp(v - max_v)
        denom = float(exp.sum())
        if denom <= 0.0:
            return np.zeros_like(v)
        return (exp / denom).astype(np.float32)


__all__ = ["PoseWordOnnxModel"]
