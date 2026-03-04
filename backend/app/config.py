from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    recognition_mode: str = "letters"
    use_shoulder_norm: bool = True
    use_hands_3d_norm: bool = False
    perf_enabled: bool = False
    perf_window_size: int = 120
    perf_ema_alpha: float = 0.2
    pose_worker_enabled: bool = True
    pose_worker_queue_size: int = 3
    pose_worker_output_size: int = 4
    segmentation_enabled: bool = False
    segmentation_model_path: str = "backend/artifacts/bio_segmenter.onnx"
    segmentation_config_path: str = "backend/artifacts/bio_config.json"
    segmentation_thresholds_path: str = "backend/artifacts/bio_thresholds.json"
    segmentation_window: int = 256
    segmentation_step: int = 8
    segmentation_min_len: int = 6
    segmentation_max_len: int = 150
    segmentation_merge_gap: int = 2
    segmentation_cool_off_frames: int = 4
    segmentation_max_buffer: int = 512
    segmentation_ort_num_threads: int = 1
    segmentation_sign_th_b: float = 0.5
    segmentation_sign_th_o: float = 0.5
    segmentation_phrase_th_b: float = 0.5
    segmentation_phrase_th_o: float = 0.5
    pose_word_model_path: str = "backend/artifacts/pose_word_model.onnx"
    pose_word_labels_path: str = "backend/artifacts/pose_word_labels.txt"
    pose_word_config_path: str = "backend/artifacts/pose_word_config.json"
    pose_word_clip_frames: int = 32
    pose_word_topk: int = 5
    pose_word_no_event_label: str = "_no_event"
    pose_word_th_no_event: float = 0.60
    pose_word_th_unknown: float = 0.55
    pose_word_th_margin: float = 0.10
    pose_word_ema_alpha: float = 0.3
    pose_word_hold_segments: int = 2
    pose_word_cooldown_segments: int = 2
    pose_word_dedup_same_word: bool = True
    pose_word_ort_num_threads: int = 1
    frontend_fps: int = 12
    jpeg_quality: float = 0.75
    hold_ms: int = 700
    cooldown_ms: int = 500
    retrieval_k: int = 5
    sim_none: float = 0.38
    sim_vlm_th: float = 0.52
    margin_th: float = 0.06
    precommit_ratio: float = 0.8
    min_bbox_area: float = 0.04
    enable_vlm_judge: bool = False
    vlm_min_confidence: float = 0.65
    vlm_timeout_ms: int = 1500
    vlm_base_url: str = "http://localhost:1234/v1"
    vlm_model: str = "qwen3-vl-4b-instruct"
    embedding_model: str = "facebook/dinov2-small"
    device: str = "auto"
    letters_allowlist: list[str] | None = None
    none_label_dir: str = "_none"
    debug_overlay: bool = True
    log_uncertain_events: bool = True
    uncertain_streak_frames: int = 4
    switch_min_frames: int = 3
    vlm_min_interval_ms: int = 1800
    hand_landmarker_model_path: str = "backend/artifacts/models/hand_landmarker.task"
    hand_bbox_padding: float = 0.2
    hand_focus_ratio: float = 1.0
    hand_wrist_extension_ratio: float = 0.18
    hand_bg_suppression: bool = True
    hand_bg_darken_factor: float = 0.45
    hand_mask_dilate_ratio: float = 0.08
    hand_mask_blur_sigma: float = 3.0
    max_num_hands: int = 1
    hand_min_detection_confidence: float = 0.55
    hand_min_presence_confidence: float = 0.55
    hand_min_tracking_confidence: float = 0.55
    word_model_path: str = "backend/artifacts/slovo_word_model.onnx"
    word_labels_path: str = "backend/artifacts/labels.txt"
    word_input_size: int = 224
    word_window_frames: int = 32
    word_frame_interval: int = 2
    word_step: int = 4
    word_topk: int = 5
    word_mean: tuple[float, float, float] = (123.675, 116.28, 103.53)
    word_std: tuple[float, float, float] = (58.395, 57.12, 57.375)
    word_letterbox: bool = True
    word_pad_value: int = 114
    word_use_hand_presence_gate: bool = True
    word_no_event_label: str = "---"
    word_th_no_event: float = 0.60
    word_th_unknown: float = 0.55
    word_th_margin: float = 0.10
    word_ema_alpha: float = 0.3
    word_hold_frames: int = 6
    word_cooldown_frames: int = 10
    word_dedup_same_word: bool = True
    word_max_fps_inference: float = 0.0
    word_ort_num_threads: int = 1
    word_runtime_log_enabled: bool = True
    word_runtime_log_path: str = "backend/artifacts/words_runtime.jsonl"

    @staticmethod
    def _merge_nested_overrides(raw: dict[str, Any]) -> dict[str, Any]:
        payload = dict(raw)

        sections: list[tuple[str, dict[str, str]]] = [
            (
                "word_model",
                {
                    "path": "word_model_path",
                    "labels_path": "word_labels_path",
                    "input_size": "word_input_size",
                    "window_frames": "word_window_frames",
                    "frame_interval": "word_frame_interval",
                    "step": "word_step",
                    "topk": "word_topk",
                    "mean": "word_mean",
                    "std": "word_std",
                    "letterbox": "word_letterbox",
                    "pad_value": "word_pad_value",
                    "use_hand_presence_gate": "word_use_hand_presence_gate",
                },
            ),
            (
                "thresholds",
                {
                    "no_event_label": "word_no_event_label",
                    "th_no_event": "word_th_no_event",
                    "th_unknown": "word_th_unknown",
                    "th_margin": "word_th_margin",
                },
            ),
            (
                "smoothing",
                {
                    "ema_alpha": "word_ema_alpha",
                },
            ),
            (
                "commit_logic",
                {
                    "hold_frames": "word_hold_frames",
                    "cooldown_frames": "word_cooldown_frames",
                    "dedup_same_word": "word_dedup_same_word",
                },
            ),
            (
                "performance",
                {
                    "max_fps_inference": "word_max_fps_inference",
                    "ort_num_threads": "word_ort_num_threads",
                },
            ),
            (
                "runtime_log",
                {
                    "enabled": "word_runtime_log_enabled",
                    "path": "word_runtime_log_path",
                },
            ),
            (
                "segmentation",
                {
                    "enabled": "segmentation_enabled",
                    "model_path": "segmentation_model_path",
                    "config_path": "segmentation_config_path",
                    "thresholds_path": "segmentation_thresholds_path",
                    "window": "segmentation_window",
                    "step": "segmentation_step",
                    "min_len": "segmentation_min_len",
                    "max_len": "segmentation_max_len",
                    "merge_gap": "segmentation_merge_gap",
                    "cool_off_frames": "segmentation_cool_off_frames",
                    "max_buffer": "segmentation_max_buffer",
                    "ort_num_threads": "segmentation_ort_num_threads",
                },
            ),
            (
                "perf",
                {
                    "enabled": "perf_enabled",
                    "window_size": "perf_window_size",
                    "ema_alpha": "perf_ema_alpha",
                },
            ),
            (
                "pose_worker",
                {
                    "enabled": "pose_worker_enabled",
                    "queue_size": "pose_worker_queue_size",
                    "output_size": "pose_worker_output_size",
                },
            ),
            (
                "pose_word_model",
                {
                    "path": "pose_word_model_path",
                    "labels_path": "pose_word_labels_path",
                    "config_path": "pose_word_config_path",
                    "clip_frames": "pose_word_clip_frames",
                    "topk": "pose_word_topk",
                    "ort_num_threads": "pose_word_ort_num_threads",
                },
            ),
            (
                "pose_word_thresholds",
                {
                    "no_event_label": "pose_word_no_event_label",
                    "th_no_event": "pose_word_th_no_event",
                    "th_unknown": "pose_word_th_unknown",
                    "th_margin": "pose_word_th_margin",
                },
            ),
            (
                "pose_word_commit_logic",
                {
                    "ema_alpha": "pose_word_ema_alpha",
                    "hold_segments": "pose_word_hold_segments",
                    "cooldown_segments": "pose_word_cooldown_segments",
                    "dedup_same_word": "pose_word_dedup_same_word",
                },
            ),
        ]

        for section_name, mapping in sections:
            section = raw.get(section_name)
            if not isinstance(section, dict):
                continue
            for source_key, target_key in mapping.items():
                if source_key in section:
                    payload[target_key] = section[source_key]
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AppConfig":
        raw = cls._merge_nested_overrides(raw)
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in known}
        cfg = cls(**filtered)
        cfg.recognition_mode = str(cfg.recognition_mode).strip().lower() or "letters"
        if cfg.recognition_mode not in {"letters", "words", "pose_words"}:
            cfg.recognition_mode = "letters"
        cfg.letters_allowlist = cfg.letters_allowlist or []
        cfg.perf_window_size = max(10, int(cfg.perf_window_size))
        cfg.perf_ema_alpha = min(1.0, max(0.01, float(cfg.perf_ema_alpha)))
        cfg.pose_worker_queue_size = max(1, int(cfg.pose_worker_queue_size))
        cfg.pose_worker_output_size = max(1, int(cfg.pose_worker_output_size))
        cfg.segmentation_window = max(8, int(cfg.segmentation_window))
        cfg.segmentation_step = max(1, int(cfg.segmentation_step))
        cfg.segmentation_min_len = max(1, int(cfg.segmentation_min_len))
        cfg.segmentation_max_len = max(cfg.segmentation_min_len, int(cfg.segmentation_max_len))
        cfg.segmentation_merge_gap = max(0, int(cfg.segmentation_merge_gap))
        cfg.segmentation_cool_off_frames = max(0, int(cfg.segmentation_cool_off_frames))
        cfg.segmentation_max_buffer = max(cfg.segmentation_window, int(cfg.segmentation_max_buffer))
        cfg.segmentation_ort_num_threads = max(1, int(cfg.segmentation_ort_num_threads))
        cfg.pose_word_clip_frames = max(4, int(cfg.pose_word_clip_frames))
        cfg.pose_word_topk = max(1, int(cfg.pose_word_topk))
        cfg.pose_word_hold_segments = max(1, int(cfg.pose_word_hold_segments))
        cfg.pose_word_cooldown_segments = max(0, int(cfg.pose_word_cooldown_segments))
        cfg.pose_word_ort_num_threads = max(1, int(cfg.pose_word_ort_num_threads))
        try:
            cfg.word_mean = tuple(float(x) for x in cfg.word_mean)  # type: ignore[assignment]
            cfg.word_std = tuple(float(x) for x in cfg.word_std)  # type: ignore[assignment]
            if len(cfg.word_mean) != 3 or len(cfg.word_std) != 3:
                raise ValueError
        except Exception:
            cfg.word_mean = (123.675, 116.28, 103.53)
            cfg.word_std = (58.395, 57.12, 57.375)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config at {cfg_path} must be a mapping.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["letters_allowlist"] = self.letters_allowlist or []
        result["word_mean"] = list(self.word_mean)
        result["word_std"] = list(self.word_std)
        return result

    def write_yaml(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)


def load_config(config_path: str | Path) -> AppConfig:
    return AppConfig.from_yaml(config_path)


def merge_config_values(config_path: str | Path, updates: dict[str, Any]) -> AppConfig:
    current = load_config(config_path)
    payload = current.to_dict()
    payload.update(updates)
    merged = AppConfig.from_dict(payload)
    merged.write_yaml(config_path)
    return merged
