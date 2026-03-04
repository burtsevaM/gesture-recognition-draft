from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional import at module load
    cv2 = None
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect
from typing import cast

from .config import AppConfig, load_config
from .embedding import DinoEmbedder
from .hand_detector import HandDetection, HandDetector
from .logging_utils import UncertainEventLogger
from .perf import FrameProfiler, PerfAggregator
from .pose import PoseExtractor, PosePipelineWorker, PoseWorkerResult, compose_features, hand_normalize_3d, shoulder_normalize
from .pose.datatypes import PoseFrame, PoseLandmarksGroup
from .pose_words import PoseWordOnnxModel, resample_to_fixed_T
from .retrieval import GalleryIndex, RetrievalHit
from .segmentation import (
    BioSegmenterOnnxModel,
    StreamingBioSegmenter,
    load_bio_thresholds,
)
from .schemas import TopKItem, VLMDecision, build_inference_message
from .state_machine import HoldToCommitStateMachine
from .vlm_judge import JudgeResult, VLMJudge
from .words.model_onnx import WordOnnxModel
from .words.metrics import WordRuntimeMetrics
from .words.service import WordRecognitionService, WordServiceConfig
from .words.decoder import WordDecisionDecoder, WordThresholds

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
GALLERY_DIR = BACKEND_DIR / "gallery"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
CONFIG_PATH = BACKEND_DIR / "config.yaml"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
META_PATH = ARTIFACTS_DIR / "meta.json"
UNCERTAIN_DIR = ARTIFACTS_DIR / "uncertain_events"
VLM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm-judge")
LOGGER = logging.getLogger(__name__)


class RuntimeContext:
    def __init__(self) -> None:
        self.lock = Lock()
        self.config = load_config(CONFIG_PATH)

        self._hand_detector: HandDetector | None = None
        self._pose_extractor: PoseExtractor | None = None
        self._embedder: DinoEmbedder | None = None
        self._gallery_index: GalleryIndex | None = None
        self._word_model: WordOnnxModel | None = None
        self._bio_segmenter_model: BioSegmenterOnnxModel | None = None
        self._pose_word_model: PoseWordOnnxModel | None = None
        self._vlm_judge: VLMJudge | None = None
        self._event_logger: UncertainEventLogger | None = None

        self.errors: dict[str, str] = {}
        self._logged_missing_paths: set[str] = set()

    def reload_config(self) -> AppConfig:
        self.config = load_config(CONFIG_PATH)
        self._logged_missing_paths.clear()
        return self.config

    @staticmethod
    def _resolve_runtime_path(path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        return path

    def _log_missing_artifact_once(self, path: Path) -> None:
        token = str(path)
        if token in self._logged_missing_paths:
            return
        self._logged_missing_paths.add(token)
        LOGGER.error("Missing required pose_words artifact: %s", path)

    def _pose_words_artifact_paths(self) -> dict[str, Path]:
        cfg = self.config
        required: dict[str, Path] = {}
        if cfg.recognition_mode != "pose_words":
            return required

        if cfg.segmentation_enabled:
            required["pose_word_model.onnx"] = self._resolve_runtime_path(cfg.pose_word_model_path)
            required["pose_word_labels.txt"] = self._resolve_runtime_path(cfg.pose_word_labels_path)
            required["pose_word_config.json"] = self._resolve_runtime_path(cfg.pose_word_config_path)
            required["bio_segmenter.onnx"] = self._resolve_runtime_path(cfg.segmentation_model_path)
            required["bio_config.json"] = self._resolve_runtime_path(cfg.segmentation_config_path)
            required["bio_thresholds.json"] = self._resolve_runtime_path(cfg.segmentation_thresholds_path)
        return required

    def pose_words_missing_artifacts(self) -> list[str]:
        missing: list[str] = []
        for artifact_name, artifact_path in self._pose_words_artifact_paths().items():
            if artifact_path.exists():
                continue
            self._log_missing_artifact_once(artifact_path)
            missing.append(artifact_name)
        return missing

    def get_hand_detector(self) -> HandDetector | None:
        with self.lock:
            if self._hand_detector is not None:
                return self._hand_detector
            try:
                cfg = self.config
                model_path = Path(cfg.hand_landmarker_model_path)
                if not model_path.is_absolute():
                    model_path = (ROOT_DIR / model_path).resolve()
                self._hand_detector = HandDetector(
                    str(model_path),
                    min_bbox_area=cfg.min_bbox_area,
                    bbox_padding=cfg.hand_bbox_padding,
                    focus_ratio=cfg.hand_focus_ratio,
                    wrist_extension_ratio=cfg.hand_wrist_extension_ratio,
                    bg_suppression=cfg.hand_bg_suppression,
                    bg_darken_factor=cfg.hand_bg_darken_factor,
                    mask_dilate_ratio=cfg.hand_mask_dilate_ratio,
                    mask_blur_sigma=cfg.hand_mask_blur_sigma,
                    max_num_hands=cfg.max_num_hands,
                    min_detection_confidence=cfg.hand_min_detection_confidence,
                    min_presence_confidence=cfg.hand_min_presence_confidence,
                    min_tracking_confidence=cfg.hand_min_tracking_confidence,
                )
                self.errors.pop("hand_detector", None)
            except Exception as exc:
                self.errors["hand_detector"] = str(exc)
                return None
            return self._hand_detector

    def get_embedder(self) -> DinoEmbedder | None:
        with self.lock:
            if self._embedder is not None:
                return self._embedder
            try:
                cfg = self.config
                self._embedder = DinoEmbedder(model_name=cfg.embedding_model, device=cfg.device)
                self.errors.pop("embedder", None)
            except Exception as exc:
                self.errors["embedder"] = str(exc)
                return None
            return self._embedder

    def get_pose_extractor(self) -> PoseExtractor | None:
        with self.lock:
            if self._pose_extractor is not None:
                return self._pose_extractor
            try:
                self._pose_extractor = PoseExtractor(include_face=False)
                self.errors.pop("pose_extractor", None)
            except Exception as exc:
                self.errors["pose_extractor"] = str(exc)
                return None
            return self._pose_extractor

    def get_gallery_index(self) -> GalleryIndex | None:
        with self.lock:
            if self._gallery_index is not None:
                return self._gallery_index
            try:
                self._gallery_index = GalleryIndex.load(INDEX_PATH, META_PATH)
                self.errors.pop("gallery_index", None)
            except Exception as exc:
                self.errors["gallery_index"] = str(exc)
                return None
            return self._gallery_index

    def get_vlm_judge(self) -> VLMJudge | None:
        if not self.config.enable_vlm_judge:
            return None
        with self.lock:
            if self._vlm_judge is not None:
                return self._vlm_judge
            try:
                cfg = self.config
                self._vlm_judge = VLMJudge(
                    base_url=cfg.vlm_base_url,
                    model=cfg.vlm_model,
                    timeout_ms=cfg.vlm_timeout_ms,
                )
                self.errors.pop("vlm", None)
            except Exception as exc:
                self.errors["vlm"] = str(exc)
                return None
            return self._vlm_judge

    def get_word_model(self) -> WordOnnxModel | None:
        with self.lock:
            if self._word_model is not None:
                return self._word_model
            try:
                cfg = self.config
                model_path = Path(cfg.word_model_path)
                labels_path = Path(cfg.word_labels_path)
                if not model_path.is_absolute():
                    model_path = (ROOT_DIR / model_path).resolve()
                if not labels_path.is_absolute():
                    labels_path = (ROOT_DIR / labels_path).resolve()
                self._word_model = WordOnnxModel(
                    model_path=model_path,
                    labels_path=labels_path,
                    input_size=cfg.word_input_size,
                    ort_num_threads=cfg.word_ort_num_threads,
                    mean=tuple(cfg.word_mean),
                    std=tuple(cfg.word_std),
                    letterbox=cfg.word_letterbox,
                    pad_value=cfg.word_pad_value,
                )
                self.errors.pop("word_model", None)
            except Exception as exc:
                self.errors["word_model"] = str(exc)
                return None
            return self._word_model

    def get_bio_segmenter_model(self) -> BioSegmenterOnnxModel | None:
        with self.lock:
            if self._bio_segmenter_model is not None:
                return self._bio_segmenter_model
            try:
                cfg = self.config
                model_path = self._resolve_runtime_path(cfg.segmentation_model_path)
                config_path = self._resolve_runtime_path(cfg.segmentation_config_path)
                self._bio_segmenter_model = BioSegmenterOnnxModel(
                    model_path=model_path,
                    config_path=config_path,
                    ort_num_threads=cfg.segmentation_ort_num_threads,
                )
                self.errors.pop("bio_segmenter_model", None)
            except Exception as exc:
                self.errors["bio_segmenter_model"] = str(exc)
                LOGGER.error("BIO segmenter init failed: %s", exc)
                return None
            return self._bio_segmenter_model

    def get_pose_word_model(self) -> PoseWordOnnxModel | None:
        with self.lock:
            if self._pose_word_model is not None:
                return self._pose_word_model
            try:
                cfg = self.config
                model_path = self._resolve_runtime_path(cfg.pose_word_model_path)
                labels_path = self._resolve_runtime_path(cfg.pose_word_labels_path)
                config_path = self._resolve_runtime_path(cfg.pose_word_config_path)
                self._pose_word_model = PoseWordOnnxModel(
                    model_path=model_path,
                    labels_path=labels_path,
                    config_path=config_path,
                    ort_num_threads=cfg.pose_word_ort_num_threads,
                )
                self.errors.pop("pose_word_model", None)
            except Exception as exc:
                self.errors["pose_word_model"] = str(exc)
                LOGGER.error("Pose word model init failed: %s", exc)
                return None
            return self._pose_word_model

    def get_event_logger(self) -> UncertainEventLogger:
        with self.lock:
            if self._event_logger is None:
                self._event_logger = UncertainEventLogger(UNCERTAIN_DIR)
            return self._event_logger

    def allowed_labels(self) -> list[str]:
        if self.config.recognition_mode == "pose_words":
            return []

        if self.config.recognition_mode == "words":
            model = self.get_word_model()
            if model is None:
                return []
            no_event_idx = model.find_no_event_index(self.config.word_no_event_label)
            labels = []
            for i, label in enumerate(model.labels):
                if no_event_idx is not None and i == no_event_idx:
                    continue
                labels.append(label)
            return sorted(labels)

        index = self.get_gallery_index()
        if index and index.metadata:
            return sorted({str(item.get("letter", "")) for item in index.metadata if item.get("letter")})

        labels = []
        for p in GALLERY_DIR.iterdir() if GALLERY_DIR.exists() else []:
            if not p.is_dir():
                continue
            if p.name == self.config.none_label_dir:
                continue
            if self.config.letters_allowlist and p.name not in self.config.letters_allowlist:
                continue
            labels.append(p.name)
        return sorted(labels)

    def health(self) -> dict[str, Any]:
        cfg = self.config
        missing_artifacts = self.pose_words_missing_artifacts() if cfg.recognition_mode == "pose_words" else []
        segmentation_ready = False
        pose_word_model_ready = False
        pose_extractor_ready = False
        pose_words_ready = False
        runtime_ready = False
        if cfg.recognition_mode == "pose_words":
            hand_ready = False
            embed_ready = False
            idx = None
            index_loaded = False
            word_model_ready = False
            pose_extractor_ready = self.get_pose_extractor() is not None
            if cfg.segmentation_enabled:
                segmentation_ready = self.get_bio_segmenter_model() is not None
                pose_word_model_ready = self.get_pose_word_model() is not None
            index_size = 0
            pose_words_ready = bool(pose_extractor_ready) and (
                not cfg.segmentation_enabled or (segmentation_ready and pose_word_model_ready and not missing_artifacts)
            )
            runtime_ready = pose_words_ready
        elif cfg.recognition_mode == "words":
            hand_ready = True
            embed_ready = False
            idx = None
            index_loaded = False
            word_model = self.get_word_model()
            word_model_ready = word_model is not None
            index_size = len(word_model.labels) if word_model is not None else 0
            runtime_ready = bool(word_model_ready)
        else:
            hand_ready = self.get_hand_detector() is not None
            embed_ready = self.get_embedder() is not None
            idx = self.get_gallery_index()
            index_loaded = idx is not None and idx.size > 0
            word_model_ready = False
            index_size = idx.size if idx else 0
            runtime_ready = hand_ready and embed_ready and index_loaded

        vlm_reachable = False
        vlm_message = "disabled"
        judge = self.get_vlm_judge()
        if judge is not None:
            vlm_reachable, vlm_message = judge.health()

        return {
            "ok": True,
            "ready": bool(runtime_ready),
            "recognition_mode": cfg.recognition_mode,
            "config": cfg.to_dict(),
            "hand_detector_ready": hand_ready,
            "embedding_ready": embed_ready,
            "index_loaded": index_loaded,
            "index_size": index_size,
            "word_model_ready": word_model_ready,
            "pose_words_ready": pose_words_ready,
            "pose_word_model_ready": pose_word_model_ready,
            "segmentation_ready": segmentation_ready,
            "pose_extractor_ready": pose_extractor_ready,
            "missing_artifacts": missing_artifacts,
            "vlm_enabled": cfg.enable_vlm_judge,
            "vlm_reachable": vlm_reachable,
            "vlm_message": vlm_message,
            "allowed_labels": self.allowed_labels(),
            "errors": self.errors,
        }


class SessionProcessor:
    def __init__(self, runtime: RuntimeContext) -> None:
        self.runtime = runtime
        cfg = runtime.config
        self.recognition_mode = str(cfg.recognition_mode).lower()
        self.state = HoldToCommitStateMachine(
            hold_ms=cfg.hold_ms,
            cooldown_ms=cfg.cooldown_ms,
            precommit_ratio=cfg.precommit_ratio,
            uncertain_streak_frames=cfg.uncertain_streak_frames,
            switch_min_frames=getattr(cfg, "switch_min_frames", 3),
        )
        self.words_service: WordRecognitionService | None = None
        self.words_init_error: str | None = None
        self.pose_segmenter: StreamingBioSegmenter | None = None
        self.pose_word_model: PoseWordOnnxModel | None = None
        self.pose_word_decoder: WordDecisionDecoder | None = None
        self.pose_word_metrics: WordRuntimeMetrics = WordRuntimeMetrics()
        self.pose_no_event_index: int | None = None
        self.pose_init_error: str | None = None
        self.pose_worker: PosePipelineWorker | None = None
        self.pose_last_worker_frame_id: int = -1
        self.pose_cached_worker_result: PoseWorkerResult | None = None
        self.pose_last_payload: dict[str, Any] | None = None
        self.pose_last_ws_send_ms: float = 0.0
        self.pose_seen_segment_keys: deque[tuple[int, int]] = deque(maxlen=512)
        self.pose_seen_segment_keys_set: set[tuple[int, int]] = set()
        self.pose_in_timestamps_ms: deque[int] = deque(maxlen=180)
        self.pose_out_timestamps_ms: deque[int] = deque(maxlen=180)
        self.pose_process_ms_ema: float = 0.0
        self.pose_last_segment_event: dict[str, Any] | None = None
        perf_window_size = max(10, int(getattr(cfg, "perf_window_size", 180)))
        perf_ema_alpha = float(getattr(cfg, "perf_ema_alpha", 0.25))
        self.pose_perf = PerfAggregator(
            window_size=perf_window_size,
            ema_alpha=perf_ema_alpha,
        )
        if self.recognition_mode == "words":
            model = runtime.get_word_model()
            if model is None:
                self.words_init_error = runtime.errors.get("word_model", "word model is unavailable")
            else:
                log_path = Path(cfg.word_runtime_log_path)
                if not log_path.is_absolute():
                    log_path = (ROOT_DIR / log_path).resolve()
                self.words_service = WordRecognitionService(
                    model=model,
                    config=WordServiceConfig(
                        window_frames=cfg.word_window_frames,
                        frame_interval=cfg.word_frame_interval,
                        step=cfg.word_step,
                        topk=cfg.word_topk,
                        max_fps_inference=cfg.word_max_fps_inference,
                        no_event_label=cfg.word_no_event_label,
                        ema_alpha=cfg.word_ema_alpha,
                        hold_frames=cfg.word_hold_frames,
                        cooldown_frames=cfg.word_cooldown_frames,
                        dedup_same_word=cfg.word_dedup_same_word,
                        thresholds=WordThresholds(
                            th_no_event=cfg.word_th_no_event,
                            th_unknown=cfg.word_th_unknown,
                            th_margin=cfg.word_th_margin,
                        ),
                        log_enabled=cfg.word_runtime_log_enabled,
                        log_path=log_path,
                    ),
                )
        elif self.recognition_mode == "pose_words" and bool(getattr(cfg, "segmentation_enabled", False)):
            missing_artifacts = runtime.pose_words_missing_artifacts()
            if missing_artifacts:
                missing_str = ", ".join(missing_artifacts)
                self.pose_init_error = f"missing artifacts: {missing_str}"
            bio_model = runtime.get_bio_segmenter_model() if self.pose_init_error is None else None
            self.pose_word_model = runtime.get_pose_word_model() if self.pose_init_error is None else None
            if self.pose_init_error is not None:
                pass
            elif bio_model is None:
                self.pose_init_error = runtime.errors.get("bio_segmenter_model", "BIO segmenter model is unavailable")
            elif self.pose_word_model is None:
                self.pose_init_error = runtime.errors.get("pose_word_model", "pose word model is unavailable")
            else:
                thresholds_path = Path(cfg.segmentation_thresholds_path)
                if not thresholds_path.is_absolute():
                    thresholds_path = (ROOT_DIR / thresholds_path).resolve()
                thresholds = load_bio_thresholds(thresholds_path)
                self.pose_segmenter = StreamingBioSegmenter(
                    model=bio_model,
                    window=cfg.segmentation_window,
                    step=cfg.segmentation_step,
                    min_len=cfg.segmentation_min_len,
                    max_len=cfg.segmentation_max_len,
                    merge_gap=cfg.segmentation_merge_gap,
                    cool_off_frames=cfg.segmentation_cool_off_frames,
                    sign_th_b=thresholds.sign_th_b if thresholds.sign_th_b > 0 else cfg.segmentation_sign_th_b,
                    sign_th_o=thresholds.sign_th_o if thresholds.sign_th_o > 0 else cfg.segmentation_sign_th_o,
                    phrase_th_b=thresholds.phrase_th_b if thresholds.phrase_th_b > 0 else cfg.segmentation_phrase_th_b,
                    phrase_th_o=thresholds.phrase_th_o if thresholds.phrase_th_o > 0 else cfg.segmentation_phrase_th_o,
                    max_buffer=cfg.segmentation_max_buffer,
                )
                self.pose_word_decoder = WordDecisionDecoder(
                    ema_alpha=cfg.pose_word_ema_alpha,
                    thresholds=WordThresholds(
                        th_no_event=cfg.pose_word_th_no_event,
                        th_unknown=cfg.pose_word_th_unknown,
                        th_margin=cfg.pose_word_th_margin,
                    ),
                    hold_frames=cfg.pose_word_hold_segments,
                    cooldown_frames=cfg.pose_word_cooldown_segments,
                    dedup_same_word=cfg.pose_word_dedup_same_word,
                )
                self.pose_no_event_index = self.pose_word_model.find_no_event_index(cfg.pose_word_no_event_label)

        if self.recognition_mode == "pose_words" and bool(getattr(cfg, "pose_worker_enabled", True)):
            extractor = runtime.get_pose_extractor()
            if extractor is None:
                if self.pose_init_error is None:
                    self.pose_init_error = runtime.errors.get("pose_extractor", "pose extractor is unavailable")
            else:
                self.pose_worker = PosePipelineWorker(
                    extractor=extractor,
                    use_shoulder_norm=bool(cfg.use_shoulder_norm),
                    use_hands_3d_norm=bool(cfg.use_hands_3d_norm),
                    queue_size=int(getattr(cfg, "pose_worker_queue_size", 4)),
                    output_size=int(getattr(cfg, "pose_worker_output_size", 4)),
                    perf_enabled=bool(getattr(cfg, "perf_enabled", False)),
                    perf_aggregator=PerfAggregator(
                        window_size=perf_window_size,
                        ema_alpha=perf_ema_alpha,
                    ),
                )
                self.pose_worker.start()

        self.cached_vlm_candidate_key: str | None = None
        self.cached_vlm_result: JudgeResult | None = None
        self.last_vlm_call_ms: int = 0
        self.last_vlm_decision: VLMDecision = VLMDecision()
        self.last_vlm_decision_ms: int = 0
        self.pending_vlm_future: Future[tuple[VLMDecision, bytes | None]] | None = None
        self.pending_vlm_candidate_key: str | None = None
        self.pending_vlm_trigger: str = ""
        self.pending_vlm_context: dict[str, Any] | None = None
        self.pending_vlm_query_crop_bgr: np.ndarray | None = None
        self.pending_vlm_topk_items: list[TopKItem] = []

    def clear_text(self) -> None:
        self.state.clear_text()
        if self.words_service is not None:
            self.words_service.clear_text()
        if self.pose_word_decoder is not None:
            self.pose_word_decoder.clear_text()
            self.pose_seen_segment_keys.clear()
            self.pose_seen_segment_keys_set.clear()

    def close(self) -> None:
        if self.pose_worker is not None:
            self.pose_worker.stop(timeout_sec=0.5)
            self.pose_worker = None

    def uses_pose_worker(self) -> bool:
        return self.recognition_mode == "pose_words" and self.pose_worker is not None

    def enqueue_pose_frame(self, frame_bgr: np.ndarray, now_ms: int, *, decode_jpeg_ms: float | None = None) -> None:
        if self.pose_worker is None:
            return
        self.pose_worker.submit_frame(frame_bgr, now_ms, decode_jpeg_ms=decode_jpeg_ms)

    def process_pose_latest(self, now_ms: int) -> dict[str, Any]:
        return self._process_pose_words(None, now_ms, worker_only=True)

    def set_last_ws_send_ms(self, value_ms: float) -> None:
        self.pose_last_ws_send_ms = float(max(0.0, float(value_ms)))

    @staticmethod
    def _copy_landmarks_group(group: PoseLandmarksGroup | None) -> PoseLandmarksGroup | None:
        if group is None:
            return None
        conf = None if group.confidence is None else group.confidence.copy()
        return PoseLandmarksGroup(points=group.points.copy(), confidence=conf)

    def _extract_pose_sync(
        self,
        frame_bgr: np.ndarray,
        profiler: FrameProfiler,
    ) -> tuple[PoseFrame | None, PoseFrame | None, np.ndarray | None, bool, str | None]:
        extractor = self.runtime.get_pose_extractor()
        if extractor is None:
            return (
                None,
                None,
                None,
                False,
                self.runtime.errors.get("pose_extractor", "pose extractor is unavailable"),
            )

        if cv2 is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_bgr[:, :, ::-1]

        try:
            with profiler.stage("mediapipe_ms"):
                pose_frame = extractor.process(frame_rgb)
        except Exception as exc:
            return None, None, None, False, f"pose extraction error: {exc}"

        if pose_frame is None:
            return None, None, None, False, None

        with profiler.stage("normalize_ms"):
            norm_frame = self._normalize_pose_frame(pose_frame)
        hand_present = bool(norm_frame.left_hand is not None or norm_frame.right_hand is not None)
        with profiler.stage("feature_ms"):
            feature_vec, _ = compose_features(
                norm_frame,
                apply_shoulder_norm=False,
                hide_legs_before_body=True,
                canonical_hands_3d=False,
            )
            feature_vec = np.asarray(feature_vec, dtype=np.float32).reshape(-1)
        return pose_frame, norm_frame, feature_vec, hand_present, None

    @classmethod
    def _copy_pose_frame(cls, frame: PoseFrame) -> PoseFrame:
        return PoseFrame(
            timestamp=float(frame.timestamp),
            body=cls._copy_landmarks_group(frame.body),
            left_hand=cls._copy_landmarks_group(frame.left_hand),
            right_hand=cls._copy_landmarks_group(frame.right_hand),
            face=cls._copy_landmarks_group(frame.face),
            meta=dict(frame.meta),
        )

    @staticmethod
    def _group_to_skeleton_payload(group: PoseLandmarksGroup | None) -> dict[str, Any] | None:
        if group is None:
            return None
        payload: dict[str, Any] = {
            "points": group.points.astype(np.float32).tolist(),
        }
        if group.confidence is not None:
            payload["confidence"] = group.confidence.astype(np.float32).tolist()
        return payload

    @classmethod
    def _frame_to_skeleton_payload(cls, frame: PoseFrame | None) -> dict[str, Any]:
        if frame is None:
            return {"body": None, "lh": None, "rh": None}
        return {
            "body": cls._group_to_skeleton_payload(frame.body),
            "lh": cls._group_to_skeleton_payload(frame.left_hand),
            "rh": cls._group_to_skeleton_payload(frame.right_hand),
        }

    def _normalize_pose_frame(self, pose_frame: PoseFrame) -> PoseFrame:
        cfg = self.runtime.config
        if cfg.use_shoulder_norm:
            normalized, _ = shoulder_normalize([pose_frame], safe_mode=True)
            if normalized:
                frame = normalized[0]
            else:
                frame = self._copy_pose_frame(pose_frame)
        else:
            frame = self._copy_pose_frame(pose_frame)

        if cfg.use_hands_3d_norm:
            if frame.left_hand is not None:
                frame.left_hand.points = hand_normalize_3d(frame.left_hand.points)
            if frame.right_hand is not None:
                frame.right_hand.points = hand_normalize_3d(frame.right_hand.points)
        return frame

    @staticmethod
    def _serialize_segments(segments: list[Any]) -> list[dict[str, float | int]]:
        return [
            {
                "start": int(getattr(seg, "start", 0)),
                "end": int(getattr(seg, "end", -1)),
                "score": float(getattr(seg, "score", 0.0)),
            }
            for seg in segments
        ]

    @staticmethod
    def _compute_fps(timestamps_ms: deque[int]) -> float:
        if len(timestamps_ms) < 2:
            return 0.0
        first = int(timestamps_ms[0])
        last = int(timestamps_ms[-1])
        dt_ms = max(1, last - first)
        return float((len(timestamps_ms) - 1) * 1000.0 / dt_ms)

    def _remember_pose_segment_key(self, key: tuple[int, int]) -> None:
        if key in self.pose_seen_segment_keys_set:
            return
        if len(self.pose_seen_segment_keys) >= self.pose_seen_segment_keys.maxlen:
            old = self.pose_seen_segment_keys.popleft()
            self.pose_seen_segment_keys_set.discard(old)
        self.pose_seen_segment_keys.append(key)
        self.pose_seen_segment_keys_set.add(key)

    def _current_bio_payload(self, cfg: AppConfig, bio_debug: dict[str, Any] | None = None) -> dict[str, Any]:
        cfg_th_b = float(getattr(cfg, "segmentation_sign_th_b", 0.5))
        cfg_th_o = float(getattr(cfg, "segmentation_sign_th_o", 0.5))
        th_b = float(getattr(self.pose_segmenter, "sign_th_b", cfg_th_b)) if self.pose_segmenter else cfg_th_b
        th_o = float(getattr(self.pose_segmenter, "sign_th_o", cfg_th_o)) if self.pose_segmenter else cfg_th_o
        payload = {
            "enabled": bool(getattr(cfg, "segmentation_enabled", False)),
            "th_B": th_b,
            "th_O": th_o,
            "window": int(getattr(cfg, "segmentation_window", 0)),
            "step": int(getattr(cfg, "segmentation_step", 0)),
            "min_len": int(getattr(cfg, "segmentation_min_len", 0)),
            "merge_gap": int(getattr(cfg, "segmentation_merge_gap", 0)),
        }
        if bio_debug:
            payload["index_mode"] = bio_debug.get("index_mode", "global")
            payload["max_len"] = int(bio_debug.get("max_len", getattr(cfg, "segmentation_max_len", 0)))
            payload["cool_off_frames"] = int(
                bio_debug.get("cool_off_frames", getattr(cfg, "segmentation_cool_off_frames", 0))
            )
        return payload

    def _finalize_pose_payload(
        self,
        *,
        payload: dict[str, Any],
        now_ms: int,
        started_s: float,
        stage_values_ms: dict[str, float] | None = None,
        bio_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.runtime.config
        self.pose_out_timestamps_ms.append(int(now_ms))
        process_ms = max(0.0, (time.perf_counter() - started_s) * 1000.0)
        if self.pose_process_ms_ema <= 0.0:
            self.pose_process_ms_ema = process_ms
        else:
            self.pose_process_ms_ema = (self.pose_process_ms_ema * 0.8) + (process_ms * 0.2)

        payload_latency = payload.get("debug", {}).get("latency_ms")
        latency_ms = float(payload_latency) if isinstance(payload_latency, (int, float)) else float(process_ms)
        fps_in = self._compute_fps(self.pose_in_timestamps_ms)
        fps_total = self._compute_fps(self.pose_out_timestamps_ms)
        fps_pose = float(1000.0 / self.pose_process_ms_ema) if self.pose_process_ms_ema > 1e-6 else 0.0

        stage_values = dict(stage_values_ms or {})
        stage_values["total_ms"] = float(process_ms)
        if bool(getattr(cfg, "perf_enabled", False)):
            self.pose_perf.update(stage_values)
            perf_payload: dict[str, Any] = {
                "latency_ms": float(latency_ms),
                "fps_in": float(fps_in),
                "fps_pose": float(fps_pose),
                "fps_total": float(fps_total),
                "dropped_frames_count": int(self.pose_worker.dropped_frames_count) if self.pose_worker is not None else 0,
                "worker_processed_frames": int(self.pose_worker.processed_frames) if self.pose_worker is not None else 0,
            }
            perf_payload.update(self.pose_perf.snapshot_flat())
            payload["perf"] = perf_payload
        payload["bio"] = self._current_bio_payload(cfg, bio_debug=bio_debug)
        if self.pose_last_segment_event is not None:
            payload["segment_event"] = dict(self.pose_last_segment_event)
        self.pose_last_payload = dict(payload)
        return payload

    def _none_pose_message(
        self,
        *,
        now_ms: int,
        error: str = "",
        hand_present: bool = False,
        skeleton_raw: dict[str, Any] | None = None,
        skeleton_norm: dict[str, Any] | None = None,
        segments: dict[str, list[dict[str, float | int]]] | None = None,
        bio_debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self.runtime.config
        text_value = self.pose_word_decoder.text_value if self.pose_word_decoder is not None else self.state.text_value
        seg_enabled = bool(getattr(cfg, "segmentation_enabled", False))
        hold_target = int(getattr(cfg, "pose_word_hold_segments", cfg.hold_ms)) if seg_enabled else cfg.hold_ms
        hold_unit = "segments" if seg_enabled else "frames"
        payload = build_inference_message(
            status="NONE",
            letter="NONE",
            word="NONE",
            score=0.0,
            confidence=0.0,
            hand_present=bool(hand_present),
            bbox_norm=[0.0, 0.0, 0.0, 0.0],
            hold_elapsed_ms=0,
            hold_target_ms=max(1, int(hold_target)),
            text_value=text_value,
            committed_now=False,
            topk=[],
            vlm=VLMDecision(),
            sim1=0.0,
            sim2=0.0,
            margin=0.0,
            uncertain=False,
            cooldown_left_ms=0,
            mode="pose_words",
            hold_unit=hold_unit,
            latency_ms=None,
            fp_per_minute=self.pose_word_metrics.fp_per_minute(),
            avg_infer_latency_ms=self.pose_word_metrics.latency_summary().avg_ms,
            p95_infer_latency_ms=self.pose_word_metrics.latency_summary().p95_ms,
        )
        payload["timestamp_ms"] = int(now_ms)
        payload["skeleton"] = {
            "raw": skeleton_raw or {"body": None, "lh": None, "rh": None},
            "norm": skeleton_norm or {"body": None, "lh": None, "rh": None},
        }
        if segments is not None:
            payload["segments"] = segments
        if bio_debug:
            payload.setdefault("debug", {})["bio"] = bio_debug
        if error:
            payload["error"] = error
        return payload

    def _process_pose_words(
        self,
        frame_bgr: np.ndarray | None,
        now_ms: int,
        *,
        decode_jpeg_ms: float | None = None,
        worker_only: bool = False,
    ) -> dict[str, Any]:
        cfg = self.runtime.config
        started_s = time.perf_counter()
        self.pose_in_timestamps_ms.append(int(now_ms))
        profiler = FrameProfiler(self.pose_perf, enabled=bool(getattr(cfg, "perf_enabled", False)))
        profiler.record("decode_jpeg_ms", decode_jpeg_ms)
        profiler.record("ws_send_ms", self.pose_last_ws_send_ms)

        pose_frame: PoseFrame | None = None
        norm_frame: PoseFrame | None = None
        feature_vec: np.ndarray | None = None
        hand_present = False
        worker_result: PoseWorkerResult | None = None

        if self.pose_worker is not None:
            if frame_bgr is not None and not worker_only:
                self.pose_worker.submit_frame(frame_bgr, now_ms, decode_jpeg_ms=decode_jpeg_ms)
            latest_result = self.pose_worker.get_latest_result()
            if latest_result is not None:
                self.pose_cached_worker_result = latest_result
            worker_result = self.pose_cached_worker_result
            if worker_result is None:
                if frame_bgr is not None and not worker_only:
                    pose_frame, norm_frame, feature_vec, hand_present, sync_error = self._extract_pose_sync(
                        frame_bgr,
                        profiler,
                    )
                    if sync_error is not None:
                        stage_values = profiler.finish()
                        return self._finalize_pose_payload(
                            payload=self._none_pose_message(now_ms=now_ms, error=sync_error),
                            now_ms=now_ms,
                            started_s=started_s,
                            stage_values_ms=stage_values,
                        )
                else:
                    stage_values = profiler.finish()
                    return self._finalize_pose_payload(
                        payload=self._none_pose_message(
                            now_ms=now_ms,
                            error="pose worker warming up",
                        ),
                        now_ms=now_ms,
                        started_s=started_s,
                        stage_values_ms=stage_values,
                    )
            elif int(worker_result.frame_id) <= self.pose_last_worker_frame_id:
                if frame_bgr is not None and not worker_only:
                    pose_frame, norm_frame, feature_vec, hand_present, sync_error = self._extract_pose_sync(
                        frame_bgr,
                        profiler,
                    )
                    if sync_error is not None:
                        stage_values = profiler.finish()
                        return self._finalize_pose_payload(
                            payload=self._none_pose_message(now_ms=now_ms, error=sync_error),
                            now_ms=now_ms,
                            started_s=started_s,
                            stage_values_ms=stage_values,
                        )
                elif self.pose_last_payload is not None:
                    cached_payload = dict(self.pose_last_payload)
                    cached_payload["timestamp_ms"] = int(now_ms)
                    stage_values = profiler.finish()
                    return self._finalize_pose_payload(
                        payload=cached_payload,
                        now_ms=now_ms,
                        started_s=started_s,
                        stage_values_ms=stage_values,
                    )
                else:
                    stage_values = profiler.finish()
                    return self._finalize_pose_payload(
                        payload=self._none_pose_message(now_ms=now_ms, error="pose worker has no new frame yet"),
                        now_ms=now_ms,
                        started_s=started_s,
                        stage_values_ms=stage_values,
                    )
            else:
                if worker_result.timings_ms:
                    for name, value in worker_result.timings_ms.items():
                        profiler.record(name, value)
                if worker_result.error:
                    stage_values = profiler.finish()
                    return self._finalize_pose_payload(
                        payload=self._none_pose_message(
                            now_ms=now_ms,
                            error=f"pose worker error: {worker_result.error}",
                        ),
                        now_ms=now_ms,
                        started_s=started_s,
                        stage_values_ms=stage_values,
                    )
                pose_frame = worker_result.pose_frame
                norm_frame = worker_result.norm_frame
                feature_vec = worker_result.feature_vec
                hand_present = bool(worker_result.hand_present)
                self.pose_last_worker_frame_id = int(worker_result.frame_id)

            if pose_frame is None and frame_bgr is not None and not worker_only:
                pose_frame, norm_frame, feature_vec, hand_present, sync_error = self._extract_pose_sync(
                    frame_bgr,
                    profiler,
                )
                if sync_error is not None:
                    stage_values = profiler.finish()
                    return self._finalize_pose_payload(
                        payload=self._none_pose_message(now_ms=now_ms, error=sync_error),
                        now_ms=now_ms,
                        started_s=started_s,
                        stage_values_ms=stage_values,
                    )

            if pose_frame is None:
                stage_values = profiler.finish()
                return self._finalize_pose_payload(
                    payload=self._none_pose_message(
                        now_ms=now_ms,
                        error="pose worker warming up",
                    ),
                    now_ms=now_ms,
                    started_s=started_s,
                    stage_values_ms=stage_values,
                )
        else:
            if worker_only:
                stage_values = profiler.finish()
                return self._finalize_pose_payload(
                    payload=self._none_pose_message(now_ms=now_ms, error="pose worker is disabled"),
                    now_ms=now_ms,
                    started_s=started_s,
                    stage_values_ms=stage_values,
                )
            if frame_bgr is None:
                stage_values = profiler.finish()
                return self._finalize_pose_payload(
                    payload=self._none_pose_message(now_ms=now_ms, error="empty frame"),
                    now_ms=now_ms,
                    started_s=started_s,
                    stage_values_ms=stage_values,
                )

            pose_frame, norm_frame, feature_vec, hand_present, sync_error = self._extract_pose_sync(frame_bgr, profiler)
            if sync_error is not None:
                stage_values = profiler.finish()
                return self._finalize_pose_payload(
                    payload=self._none_pose_message(now_ms=now_ms, error=sync_error),
                    now_ms=now_ms,
                    started_s=started_s,
                    stage_values_ms=stage_values,
                )

        if pose_frame is None:
            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=self._none_pose_message(now_ms=now_ms),
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
            )

        if norm_frame is None:
            with profiler.stage("normalize_ms"):
                norm_frame = self._normalize_pose_frame(pose_frame)
            hand_present = bool(norm_frame.left_hand is not None or norm_frame.right_hand is not None)

        if feature_vec is None:
            with profiler.stage("feature_ms"):
                feature_vec, _ = compose_features(
                    norm_frame,
                    apply_shoulder_norm=False,
                    hide_legs_before_body=True,
                    canonical_hands_3d=False,
                )
                feature_vec = np.asarray(feature_vec, dtype=np.float32).reshape(-1)

        skeleton_raw = self._frame_to_skeleton_payload(pose_frame)
        skeleton_norm = self._frame_to_skeleton_payload(norm_frame)

        if not bool(getattr(cfg, "segmentation_enabled", False)):
            payload = build_inference_message(
                status="POSE",
                letter="NONE",
                word="NONE",
                score=0.0,
                confidence=0.0,
                hand_present=hand_present,
                bbox_norm=[0.0, 0.0, 0.0, 0.0],
                hold_elapsed_ms=0,
                hold_target_ms=max(1, int(cfg.hold_ms)),
                text_value=self.state.text_value,
                committed_now=False,
                topk=[],
                vlm=VLMDecision(),
                sim1=0.0,
                sim2=0.0,
                margin=0.0,
                uncertain=False,
                cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
                mode="pose_words",
            )
            payload["timestamp_ms"] = int(now_ms)
            payload["skeleton"] = {"raw": skeleton_raw, "norm": skeleton_norm}
            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=payload,
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
            )

        if self.pose_segmenter is None or self.pose_word_model is None or self.pose_word_decoder is None:
            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=self._none_pose_message(
                    now_ms=now_ms,
                    error=self.pose_init_error or "pose_words segmentation runtime is unavailable",
                    hand_present=hand_present,
                    skeleton_raw=skeleton_raw,
                    skeleton_norm=skeleton_norm,
                ),
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
            )

        try:
            segment_result = self.pose_segmenter.update(cast(np.ndarray, feature_vec))
            profiler.record("bio_infer_ms", segment_result.latency_ms)
            profiler.record("bio_decode_ms", segment_result.decode_latency_ms)
        except Exception as exc:
            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=self._none_pose_message(
                    now_ms=now_ms,
                    error=f"segmentation error: {exc}",
                    hand_present=hand_present,
                    skeleton_raw=skeleton_raw,
                    skeleton_norm=skeleton_norm,
                ),
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
            )

        segments_payload = {
            "sign": self._serialize_segments(segment_result.recent_sign_segments),
            "phrase": self._serialize_segments(segment_result.recent_phrase_segments),
        }
        bio_debug = {
            "enabled": True,
            "index_mode": str(segment_result.index_mode),
            "window": int(cfg.segmentation_window),
            "step": int(cfg.segmentation_step),
            "min_len": int(cfg.segmentation_min_len),
            "max_len": int(cfg.segmentation_max_len),
            "merge_gap": int(cfg.segmentation_merge_gap),
            "cool_off_frames": int(cfg.segmentation_cool_off_frames),
            "th_B_sign": float(self.pose_segmenter.sign_th_b),
            "th_O_sign": float(self.pose_segmenter.sign_th_o),
            "th_B_phrase": float(self.pose_segmenter.phrase_th_b),
            "th_O_phrase": float(self.pose_segmenter.phrase_th_o),
            "buffer_len": int(segment_result.buffer_len),
            "buffer_start": int(segment_result.buffer_start),
            "buffer_end": int(segment_result.buffer_end),
            "active_sign": bool(segment_result.active_sign),
            "active_phrase": bool(segment_result.active_phrase),
            "active_sign_progress": float(segment_result.active_sign_progress),
            "active_phrase_progress": float(segment_result.active_phrase_progress),
            "dropped_frames_count": int(worker_result.dropped_frames_count) if worker_result is not None else 0,
        }
        if segment_result.sign_segments:
            latest_seg = segment_result.sign_segments[-1]
            self.pose_last_segment_event = {
                "start": int(latest_seg.start),
                "end": int(latest_seg.end),
                "len": int(max(0, latest_seg.end - latest_seg.start + 1)),
                "score": float(latest_seg.score),
                "timestamp_ms": int(now_ms),
            }

        if not segment_result.sign_segments:
            if segment_result.active_sign:
                hold_target = max(1, int(cfg.segmentation_min_len))
                hold_elapsed = int(round(float(segment_result.active_sign_progress) * hold_target))
                payload = build_inference_message(
                    status="HOLD",
                    letter="NONE",
                    word="NONE",
                    score=0.0,
                    confidence=0.0,
                    hand_present=hand_present,
                    bbox_norm=[0.0, 0.0, 0.0, 0.0],
                    hold_elapsed_ms=hold_elapsed,
                    hold_target_ms=hold_target,
                    text_value=self.pose_word_decoder.text_value,
                    committed_now=False,
                    topk=[],
                    vlm=VLMDecision(),
                    sim1=0.0,
                    sim2=0.0,
                    margin=0.0,
                    uncertain=False,
                    cooldown_left_ms=0,
                    mode="pose_words",
                    hold_unit="segments",
                    latency_ms=segment_result.latency_ms,
                    fp_per_minute=self.pose_word_metrics.fp_per_minute(),
                    avg_infer_latency_ms=self.pose_word_metrics.latency_summary().avg_ms,
                    p95_infer_latency_ms=self.pose_word_metrics.latency_summary().p95_ms,
                )
                payload["timestamp_ms"] = int(now_ms)
                payload["skeleton"] = {"raw": skeleton_raw, "norm": skeleton_norm}
                payload["segments"] = segments_payload
                payload.setdefault("debug", {})["bio"] = bio_debug
                stage_values = profiler.finish()
                return self._finalize_pose_payload(
                    payload=payload,
                    now_ms=now_ms,
                    started_s=started_s,
                    stage_values_ms=stage_values,
                    bio_debug=bio_debug,
                )

            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=self._none_pose_message(
                    now_ms=now_ms,
                    hand_present=hand_present,
                    skeleton_raw=skeleton_raw,
                    skeleton_norm=skeleton_norm,
                    segments=segments_payload,
                    bio_debug=bio_debug,
                ),
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
                bio_debug=bio_debug,
            )

        decoded = None
        topk_pairs: list[tuple[str, float]] = []
        total_latency_ms = float(segment_result.latency_ms or 0.0)
        word_infer_total_ms = 0.0
        decoder_total_ms = 0.0

        if segment_result.latency_ms is not None:
            self.pose_word_metrics.record_inference(float(segment_result.latency_ms))

        for seg in segment_result.sign_segments:
            seg_key = (int(seg.start), int(seg.end))
            if seg_key in self.pose_seen_segment_keys_set:
                continue
            self._remember_pose_segment_key(seg_key)
            seg_feats = self.pose_segmenter.get_feature_span(seg.start, seg.end)
            if seg_feats is None or seg_feats.shape[0] == 0:
                continue
            clip = resample_to_fixed_T(seg_feats, T=cfg.pose_word_clip_frames, method="linear")
            probs, cls_latency = self.pose_word_model.infer_probs(clip)
            word_infer_total_ms += float(cls_latency)
            total_latency_ms += float(cls_latency)
            self.pose_word_metrics.record_inference(float(cls_latency))
            decode_started = time.perf_counter()
            decoded = self.pose_word_decoder.update(
                probs=probs,
                labels=self.pose_word_model.labels,
                topk=cfg.pose_word_topk,
                no_event_index=self.pose_no_event_index,
            )
            decoder_total_ms += float((time.perf_counter() - decode_started) * 1000.0)
            topk_pairs = [(self.pose_word_model.labels[i], float(probs[i])) for i in decoded.topk_indices]

        profiler.record("word_infer_ms", word_infer_total_ms)
        profiler.record("decoder_ms", decoder_total_ms)

        if decoded is None:
            stage_values = profiler.finish()
            return self._finalize_pose_payload(
                payload=self._none_pose_message(
                    now_ms=now_ms,
                    hand_present=hand_present,
                    skeleton_raw=skeleton_raw,
                    skeleton_norm=skeleton_norm,
                    segments=segments_payload,
                    bio_debug=bio_debug,
                ),
                now_ms=now_ms,
                started_s=started_s,
                stage_values_ms=stage_values,
                bio_debug=bio_debug,
            )

        committed_word = decoded.committed_word if decoded.committed else None
        self.pose_word_metrics.record_state(
            timestamp_ms=int(now_ms),
            state=decoded.state,
            committed_word=committed_word,
        )
        latency_summary = self.pose_word_metrics.latency_summary()

        topk_items = [TopKItem(letter=label, score=score, exemplar_path="") for label, score in topk_pairs]
        payload = build_inference_message(
            status=decoded.state,
            letter=decoded.top1_label,
            word=decoded.top1_label,
            score=float(decoded.top1_prob),
            confidence=float(decoded.top1_prob),
            hand_present=hand_present,
            bbox_norm=[0.0, 0.0, 0.0, 0.0],
            hold_elapsed_ms=int(decoded.hold_count),
            hold_target_ms=int(max(1, decoded.hold_target)),
            text_value=self.pose_word_decoder.text_value,
            committed_now=bool(decoded.committed),
            topk=topk_items,
            vlm=VLMDecision(),
            sim1=float(decoded.top1_prob),
            sim2=float(decoded.top2_prob),
            margin=float(decoded.margin),
            uncertain=decoded.state == "UNKNOWN",
            cooldown_left_ms=int(decoded.cooldown_left),
            mode="pose_words",
            hold_unit="segments",
            latency_ms=float(total_latency_ms),
            fp_per_minute=self.pose_word_metrics.fp_per_minute(),
            avg_infer_latency_ms=latency_summary.avg_ms,
            p95_infer_latency_ms=latency_summary.p95_ms,
        )
        payload["timestamp_ms"] = int(now_ms)
        payload["skeleton"] = {"raw": skeleton_raw, "norm": skeleton_norm}
        payload["segments"] = segments_payload
        payload.setdefault("debug", {})["bio"] = bio_debug
        payload["top1"] = {
            "label": decoded.top1_label,
            "prob": float(decoded.top1_prob),
            "no_event_prob": float(decoded.no_event_prob),
        }
        payload["state_detail"] = {
            "hold_segments": int(decoded.hold_count),
            "hold_target_segments": int(decoded.hold_target),
            "hold_progress": float(decoded.hold_progress),
            "cooldown_left_segments": int(decoded.cooldown_left),
        }
        stage_values = profiler.finish()
        return self._finalize_pose_payload(
            payload=payload,
            now_ms=now_ms,
            started_s=started_s,
            stage_values_ms=stage_values,
            bio_debug=bio_debug,
        )

    def _none_words_message(self, *, now_ms: int, error: str = "", hand_present: bool = False) -> dict[str, Any]:
        text_value = ""
        if self.words_service is not None:
            text_value = self.words_service.decoder.text_value
        return build_inference_message(
            status="NONE",
            letter="NONE",
            word="NONE",
            score=0.0,
            confidence=0.0,
            hand_present=hand_present,
            bbox_norm=[0.0, 0.0, 0.0, 0.0],
            hold_elapsed_ms=0,
            hold_target_ms=max(1, int(self.runtime.config.word_hold_frames)),
            text_value=text_value,
            committed_now=False,
            topk=[],
            vlm=VLMDecision(),
            sim1=0.0,
            sim2=0.0,
            margin=0.0,
            uncertain=False,
            cooldown_left_ms=0,
            mode="words",
            hold_unit="frames",
            latency_ms=None,
            fp_per_minute=None,
            avg_infer_latency_ms=None,
            p95_infer_latency_ms=None,
        ) | {"timestamp_ms": int(now_ms), "error": error}

    def _none_message(self, *, detection: HandDetection, now_ms: int, topk_items: list[TopKItem] | None = None):
        return build_inference_message(
            status="NONE",
            letter="NONE",
            score=0.0,
            confidence=0.0,
            hand_present=detection.hand_present,
            bbox_norm=list(detection.bbox_norm),
            hold_elapsed_ms=0,
            hold_target_ms=self.runtime.config.hold_ms,
            text_value=self.state.text_value,
            committed_now=False,
            topk=topk_items or [],
            vlm=VLMDecision(),
            sim1=0.0,
            sim2=0.0,
            margin=0.0,
            uncertain=False,
            cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
        )

    def _run_vlm(
        self,
        *,
        query_rgb: np.ndarray,
        hits: list[RetrievalHit],
        trigger: str,
        sim1: float,
        sim2: float,
        margin: float,
    ) -> tuple[VLMDecision, bytes | None]:
        judge = self.runtime.get_vlm_judge()
        if judge is None:
            return VLMDecision(used=False, reason="VLM judge disabled or unavailable"), None

        try:
            result, collage_jpeg = judge.judge(
                query_rgb=query_rgb,
                topk_hits=hits,
                allowed_labels=set(self.runtime.allowed_labels()),
                trigger=trigger,
            )
            vlm = VLMDecision(
                used=True,
                letter=result.letter,
                confidence=result.confidence,
                reason=result.reason,
                trigger=trigger,
            )
            return vlm, collage_jpeg
        except Exception as exc:
            return VLMDecision(used=False, reason=f"VLM error: {exc}", trigger=trigger), None

    def _clear_pending_vlm(self) -> None:
        self.pending_vlm_future = None
        self.pending_vlm_candidate_key = None
        self.pending_vlm_trigger = ""
        self.pending_vlm_context = None
        self.pending_vlm_query_crop_bgr = None
        self.pending_vlm_topk_items = []

    def _collect_pending_vlm(self, now_ms: int, current_candidate_key: str | None) -> VLMDecision | None:
        future = self.pending_vlm_future
        if future is None or not future.done():
            return None

        if self.pending_vlm_candidate_key != current_candidate_key:
            self._clear_pending_vlm()
            return None

        try:
            decision, collage_jpeg = future.result()
        except Exception as exc:
            decision = VLMDecision(used=False, reason=f"VLM future error: {exc}", trigger=self.pending_vlm_trigger)
            collage_jpeg = None

        self.cached_vlm_result = JudgeResult(
            letter=decision.letter,
            confidence=decision.confidence,
            reason=decision.reason,
            raw="",
            used=decision.used,
            trigger=decision.trigger,
        )
        self.last_vlm_decision = decision
        self.last_vlm_decision_ms = now_ms
        self.state.mark_vlm_called()

        cfg = self.runtime.config
        if cfg.log_uncertain_events and self.pending_vlm_context and self.pending_vlm_query_crop_bgr is not None:
            logger = self.runtime.get_event_logger()
            try:
                logger.log_event(
                    query_crop_bgr=self.pending_vlm_query_crop_bgr,
                    collage_jpeg_bytes=collage_jpeg,
                    payload={
                        "timestamp_ms": now_ms,
                        **self.pending_vlm_context,
                        "post_vlm": decision.to_dict(),
                        "topk": [item.to_dict() for item in self.pending_vlm_topk_items],
                    },
                )
            except Exception:
                pass

        self._clear_pending_vlm()
        return decision

    def _schedule_vlm(
        self,
        *,
        now_ms: int,
        query_rgb: np.ndarray,
        query_crop_bgr: np.ndarray,
        hits: list[RetrievalHit],
        topk_items: list[TopKItem],
        trigger: str,
        candidate_key: str | None,
        pre_vlm: dict[str, Any],
    ) -> bool:
        if self.pending_vlm_future is not None:
            return False
        if candidate_key is None:
            return False

        self.pending_vlm_future = VLM_EXECUTOR.submit(
            self._run_vlm,
            query_rgb=query_rgb.copy(),
            hits=hits,
            trigger=trigger,
            sim1=float(pre_vlm.get("sim1", 0.0)),
            sim2=float(pre_vlm.get("sim2", 0.0)),
            margin=float(pre_vlm.get("margin", 0.0)),
        )
        self.pending_vlm_candidate_key = candidate_key
        self.pending_vlm_trigger = trigger
        self.pending_vlm_context = {
            "trigger": trigger,
            "pre_vlm": pre_vlm,
        }
        self.pending_vlm_query_crop_bgr = query_crop_bgr.copy()
        self.pending_vlm_topk_items = [TopKItem(letter=i.letter, score=i.score, exemplar_path=i.exemplar_path) for i in topk_items]
        self.last_vlm_call_ms = now_ms
        return True

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        now_ms: int,
        *,
        decode_jpeg_ms: float | None = None,
    ) -> dict[str, Any]:
        cfg = self.runtime.config
        if self.recognition_mode == "pose_words":
            return self._process_pose_words(frame_bgr, now_ms, decode_jpeg_ms=decode_jpeg_ms)

        if self.recognition_mode == "words":
            if self.words_service is None:
                return self._none_words_message(now_ms=now_ms, error=self.words_init_error or "words service unavailable")
            if cfg.word_use_hand_presence_gate and cv2 is not None:
                detector = self.runtime.get_hand_detector()
                if detector is not None:
                    detection = detector.detect(frame_bgr, now_ms)
                    if not detection.hand_present:
                        return self._none_words_message(now_ms=now_ms, hand_present=False)
            try:
                return self.words_service.update(frame_bgr, now_ms)
            except Exception as exc:
                return self._none_words_message(now_ms=now_ms, error=f"words inference error: {exc}")

        if cv2 is None:
            detection = HandDetection(False, (0.0, 0.0, 0.0, 0.0), (0, 0, 0, 0), None)
            return self._none_message(detection=detection, now_ms=now_ms)

        hand_detector = self.runtime.get_hand_detector()
        if hand_detector is None:
            detection = HandDetection(False, (0.0, 0.0, 0.0, 0.0), (0, 0, 0, 0), None)
            return self._none_message(detection=detection, now_ms=now_ms)

        detection = hand_detector.detect(frame_bgr, now_ms)

        if self.state.in_cooldown(now_ms):
            cooldown_letter = "NONE"
            cooldown_score = 0.0
            cooldown_confidence = 0.0
            cooldown_topk: list[TopKItem] = []
            sim1 = 0.0
            sim2 = 0.0
            margin = 0.0
            uncertain = False
            hand_present = bool(detection.hand_present)
            bbox_norm = list(detection.bbox_norm) if hand_present else [0.0, 0.0, 0.0, 0.0]

            if detection.hand_present and detection.crop_bgr is not None:
                embedder = self.runtime.get_embedder()
                gallery_index = self.runtime.get_gallery_index()
                if embedder is not None and gallery_index is not None and gallery_index.size > 0:
                    query_rgb = cv2.cvtColor(detection.crop_bgr, cv2.COLOR_BGR2RGB)
                    query_vec = embedder.embed_rgb(query_rgb)
                    hits = gallery_index.search(query_vec, k=cfg.retrieval_k)
                    cooldown_topk = [
                        TopKItem(letter=hit.letter, score=hit.score, exemplar_path=hit.exemplar_path)
                        for hit in hits
                    ]
                    if hits:
                        sim1 = float(hits[0].score)
                        sim2 = float(hits[1].score) if len(hits) > 1 else -1.0
                        margin = float(sim1 - sim2)
                        uncertain = (sim1 < cfg.sim_vlm_th) or (margin < cfg.margin_th)
                        cooldown_letter = hits[0].letter
                        cooldown_score = sim1
                        cooldown_confidence = sim1

            return build_inference_message(
                status="COOLDOWN",
                letter=cooldown_letter,
                score=cooldown_score,
                confidence=cooldown_confidence,
                hand_present=hand_present,
                bbox_norm=bbox_norm,
                hold_elapsed_ms=0,
                hold_target_ms=cfg.hold_ms,
                text_value=self.state.text_value,
                committed_now=False,
                topk=cooldown_topk,
                vlm=VLMDecision(),
                sim1=sim1,
                sim2=sim2,
                margin=margin,
                uncertain=uncertain,
                cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
            )

        if not detection.hand_present or detection.crop_bgr is None:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms)

        embedder = self.runtime.get_embedder()
        gallery_index = self.runtime.get_gallery_index()
        if embedder is None or gallery_index is None or gallery_index.size == 0:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms)

        query_rgb = cv2.cvtColor(detection.crop_bgr, cv2.COLOR_BGR2RGB)
        query_vec = embedder.embed_rgb(query_rgb)
        hits = gallery_index.search(query_vec, k=cfg.retrieval_k)

        topk_items = [TopKItem(letter=hit.letter, score=hit.score, exemplar_path=hit.exemplar_path) for hit in hits]
        if not hits:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return self._none_message(detection=detection, now_ms=now_ms, topk_items=topk_items)

        sim1 = hits[0].score
        sim2 = hits[1].score if len(hits) > 1 else -1.0
        margin = sim1 - sim2

        if sim1 < cfg.sim_none:
            self.state.clear_candidate()
            self.cached_vlm_candidate_key = None
            self.cached_vlm_result = None
            return build_inference_message(
                status="NONE",
                letter="NONE",
                score=float(sim1),
                confidence=0.0,
                hand_present=False,
                bbox_norm=[0.0, 0.0, 0.0, 0.0],
                hold_elapsed_ms=0,
                hold_target_ms=cfg.hold_ms,
                text_value=self.state.text_value,
                committed_now=False,
                topk=topk_items,
                vlm=VLMDecision(),
                sim1=float(sim1),
                sim2=float(sim2),
                margin=float(margin),
                uncertain=False,
                cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
            )

        hold = self.state.update_candidate(hits[0].letter, now_ms)
        candidate_letter = self.state.candidate_letter or hits[0].letter
        candidate_key = self.state.candidate_key
        if candidate_key != self.cached_vlm_candidate_key:
            self.cached_vlm_candidate_key = candidate_key
            self.cached_vlm_result = None

        uncertain = (sim1 < cfg.sim_vlm_th) or (margin < cfg.margin_th)
        self.state.update_uncertain(uncertain)

        if now_ms - self.last_vlm_decision_ms <= 4000:
            vlm_decision = self.last_vlm_decision
        else:
            vlm_decision = VLMDecision()

        vlm_interval_ms = int(getattr(cfg, "vlm_min_interval_ms", 1800))
        vlm_allowed_now = (now_ms - self.last_vlm_call_ms) >= vlm_interval_ms

        ready_vlm = self._collect_pending_vlm(now_ms, candidate_key)
        if ready_vlm is not None:
            vlm_decision = ready_vlm
        elif self.pending_vlm_future is not None:
            vlm_decision = VLMDecision(
                used=False,
                letter="NONE",
                confidence=0.0,
                reason="pending",
                trigger=self.pending_vlm_trigger,
            )

        should_call_vlm, trigger = self.state.should_call_vlm(
            hold_elapsed_ms=hold.hold_elapsed_ms,
            is_uncertain=uncertain,
        )

        if (
            cfg.enable_vlm_judge
            and should_call_vlm
            and vlm_allowed_now
            and self.pending_vlm_future is None
            and detection.crop_bgr is not None
        ):
            started = self._schedule_vlm(
                now_ms=now_ms,
                query_rgb=query_rgb,
                query_crop_bgr=detection.crop_bgr,
                hits=hits,
                topk_items=topk_items,
                trigger=trigger,
                candidate_key=candidate_key,
                pre_vlm={
                    "candidate": candidate_letter,
                    "sim1": float(sim1),
                    "sim2": float(sim2),
                    "margin": float(margin),
                    "uncertain": bool(uncertain),
                    "hold_elapsed_ms": int(hold.hold_elapsed_ms),
                },
            )
            if started:
                vlm_decision = VLMDecision(
                    used=False,
                    letter="NONE",
                    confidence=0.0,
                    reason="pending",
                    trigger=trigger,
                )

        commit_now = False
        final_letter = candidate_letter
        final_conf = sim1

        if hold.hold_elapsed_ms >= cfg.hold_ms:
            if uncertain and cfg.enable_vlm_judge and self.runtime.get_vlm_judge() is not None:
                verdict = self.cached_vlm_result
                if verdict is None:
                    if vlm_allowed_now and self.pending_vlm_future is None and detection.crop_bgr is not None:
                        self._schedule_vlm(
                            now_ms=now_ms,
                            query_rgb=query_rgb,
                            query_crop_bgr=detection.crop_bgr,
                            hits=hits,
                            topk_items=topk_items,
                            trigger="commit_gate",
                            candidate_key=candidate_key,
                            pre_vlm={
                                "candidate": candidate_letter,
                                "sim1": float(sim1),
                                "sim2": float(sim2),
                                "margin": float(margin),
                                "uncertain": bool(uncertain),
                                "hold_elapsed_ms": int(hold.hold_elapsed_ms),
                            },
                        )
                        vlm_decision = VLMDecision(
                            used=False,
                            letter="NONE",
                            confidence=0.0,
                            reason="pending",
                            trigger="commit_gate",
                        )

                if verdict and verdict.letter != "NONE" and verdict.confidence >= cfg.vlm_min_confidence:
                    final_letter = verdict.letter
                    final_conf = verdict.confidence
                    commit_now = True
                else:
                    commit_now = False
            else:
                commit_now = True

        if commit_now:
            self.state.commit(final_letter, now_ms)
            status = "COMMITTED"
        else:
            status = "CANDIDATE"

        return build_inference_message(
            status=status,
            letter=final_letter if status == "COMMITTED" else candidate_letter,
            score=float(sim1),
            confidence=float(final_conf if status == "COMMITTED" else sim1),
            hand_present=True,
            bbox_norm=list(detection.bbox_norm),
            hold_elapsed_ms=hold.hold_elapsed_ms,
            hold_target_ms=cfg.hold_ms,
            text_value=self.state.text_value,
            committed_now=commit_now,
            topk=topk_items,
            vlm=vlm_decision,
            sim1=float(sim1),
            sim2=float(sim2),
            margin=float(margin),
            uncertain=bool(uncertain),
            cooldown_left_ms=self.state.cooldown_left_ms(now_ms),
        )


runtime = RuntimeContext()

app = FastAPI(title="RSL Static Dactyl MVP")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
app.mount("/gallery_files", StaticFiles(directory=str(GALLERY_DIR)), name="gallery_files")


@app.on_event("startup")
async def startup_event() -> None:
    runtime.reload_config()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(runtime.health())


@app.get("/api/gallery")
def api_gallery() -> JSONResponse:
    payload: dict[str, list[str]] = {}
    if not GALLERY_DIR.exists():
        return JSONResponse(payload)

    for label_dir in sorted(p for p in GALLERY_DIR.iterdir() if p.is_dir()):
        urls = []
        for img in sorted(label_dir.rglob("*")):
            if not img.is_file():
                continue
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel_path = img.relative_to(GALLERY_DIR).as_posix()
            urls.append(f"/gallery_files/{rel_path}")
        payload[label_dir.name] = urls
    return JSONResponse(payload)


@app.get("/gallery")
def gallery_inspector() -> HTMLResponse:
    html = """
<!doctype html>
<html lang=\"ru\">
<head>
  <meta charset=\"utf-8\" />
  <title>Gallery Inspector</title>
  <style>
    body { font-family: sans-serif; padding: 16px; background: #101318; color: #f3f5f7; }
    .topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .back-btn {
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      color: #eaf2f8;
      background: #27303d;
      border: 1px solid #41506a;
      padding: 8px 12px;
      border-radius: 8px;
      font-weight: 600;
    }
    .back-btn:hover { background: #2f3b4b; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
    .tile { background: #1c2129; border-radius: 10px; padding: 8px; }
    img { width: 100%; border-radius: 6px; display: block; }
    h1 { margin: 0; }
    h2 { margin-top: 24px; }
  </style>
</head>
<body>
  <div class=\"topbar\">
    <a class=\"back-btn\" href=\"/\" id=\"backBtn\">← Назад</a>
    <h1>Эталоны галереи</h1>
  </div>
  <div id=\"content\"></div>
  <script>
    document.getElementById('backBtn').addEventListener('click', (event) => {
      event.preventDefault();
      if (window.history.length > 1) {
        window.history.back();
      } else {
        window.location.href = '/';
      }
    });

    async function run() {
      const res = await fetch('/api/gallery');
      const data = await res.json();
      const root = document.getElementById('content');
      for (const [label, images] of Object.entries(data)) {
        const h = document.createElement('h2');
        h.textContent = `${label} (${images.length})`;
        root.appendChild(h);
        const grid = document.createElement('div');
        grid.className = 'grid';
        for (const url of images) {
          const tile = document.createElement('div');
          tile.className = 'tile';
          const img = document.createElement('img');
          img.src = url;
          tile.appendChild(img);
          grid.appendChild(tile);
        }
        root.appendChild(grid);
      }
    }
    run();
  </script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session = SessionProcessor(runtime)
    try:
        while True:
            try:
                packet = await websocket.receive()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            if packet["type"] == "websocket.disconnect":
                break

            if packet.get("text") is not None:
                try:
                    data = json.loads(packet["text"])
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "control" and data.get("action") == "clear_text":
                    session.clear_text()
                    await websocket.send_json({"type": "ack", "action": "clear_text"})
                continue

            frame_bytes = packet.get("bytes")
            if not frame_bytes:
                continue

            if cv2 is None:
                await websocket.send_json(
                    {
                        "status": "NONE",
                        "letter": "NONE",
                        "error": "opencv-python is not installed",
                    }
                )
                continue

            decode_started = time.perf_counter()
            np_buf = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            decode_jpeg_ms = float((time.perf_counter() - decode_started) * 1000.0)
            if frame_bgr is None:
                continue

            now_ms = int(time.monotonic() * 1000)
            if session.uses_pose_worker():
                session.enqueue_pose_frame(frame_bgr, now_ms, decode_jpeg_ms=decode_jpeg_ms)
                payload = await asyncio.to_thread(session.process_pose_latest, now_ms)
            else:
                payload = await asyncio.to_thread(
                    session.process_frame,
                    frame_bgr,
                    now_ms,
                    decode_jpeg_ms=decode_jpeg_ms,
                )
            send_started = time.perf_counter()
            try:
                await websocket.send_json(payload)
                session.set_last_ws_send_ms((time.perf_counter() - send_started) * 1000.0)
            except WebSocketDisconnect:
                break
            except RuntimeError:
                break
            except Exception:
                break
    finally:
        session.close()
