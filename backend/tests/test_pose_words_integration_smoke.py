from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.main import SessionProcessor
from app.pose.datatypes import PoseFrame, PoseLandmarksGroup


class _SyntheticBioModel:
    def infer(self, features_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        features = np.asarray(features_tf, dtype=np.float32)
        t_len = int(features.shape[0])

        sign = np.zeros((t_len, 3), dtype=np.float32)
        phrase = np.zeros((t_len, 3), dtype=np.float32)

        sign[:, 2] = 0.92
        sign[:, 0] = 0.04
        sign[:, 1] = 0.04

        phrase[:, 2] = 0.9
        phrase[:, 0] = 0.05
        phrase[:, 1] = 0.05

        for i in range(t_len):
            frame_idx = int(round(float(features[i, 0])))
            if frame_idx == 4:
                sign[i] = [0.94, 0.03, 0.03]
            elif frame_idx in {5, 6}:
                sign[i] = [0.06, 0.88, 0.06]

            if frame_idx == 3:
                phrase[i] = [0.9, 0.06, 0.04]
            elif frame_idx in {4, 5, 6, 7}:
                phrase[i] = [0.08, 0.84, 0.08]

        return sign, phrase, 2.0


class _StubPoseExtractor:
    def __init__(self) -> None:
        self._frame_idx = 0

    def process(self, rgb_frame: np.ndarray) -> PoseFrame:
        idx = float(self._frame_idx)
        self._frame_idx += 1

        body = np.zeros((33, 3), dtype=np.float32)
        body[0] = [idx, 0.0, 0.0]
        body[11] = [0.4, 0.5, 0.0]
        body[12] = [0.6, 0.5, 0.0]

        return PoseFrame(
            timestamp=idx,
            body=PoseLandmarksGroup(points=body, confidence=np.ones((33,), dtype=np.float32)),
            left_hand=None,
            right_hand=None,
            meta={},
        )


class _StubPoseWordModel:
    def __init__(self) -> None:
        self.labels = ["_no_event", "СЛОВО"]
        self.calls = 0

    def find_no_event_index(self, label_name: str) -> int | None:
        return 0

    def infer_probs(self, features_tf: np.ndarray) -> tuple[np.ndarray, float]:
        self.calls += 1
        probs = np.asarray([0.03, 0.97], dtype=np.float32)
        return probs, 1.0


class _StubRuntime:
    def __init__(self) -> None:
        self.errors: dict[str, str] = {}
        self._pose_extractor = _StubPoseExtractor()
        self._bio_model = _SyntheticBioModel()
        self._pose_word_model = _StubPoseWordModel()
        self.config = SimpleNamespace(
            recognition_mode="pose_words",
            hold_ms=700,
            cooldown_ms=500,
            precommit_ratio=0.8,
            uncertain_streak_frames=4,
            switch_min_frames=3,
            use_shoulder_norm=False,
            use_hands_3d_norm=False,
            segmentation_enabled=True,
            segmentation_thresholds_path="missing_thresholds.json",
            segmentation_window=8,
            segmentation_step=1,
            segmentation_min_len=2,
            segmentation_max_len=80,
            segmentation_merge_gap=0,
            segmentation_cool_off_frames=0,
            segmentation_max_buffer=64,
            segmentation_sign_th_b=0.5,
            segmentation_sign_th_o=0.5,
            segmentation_phrase_th_b=0.5,
            segmentation_phrase_th_o=0.5,
            pose_word_clip_frames=8,
            pose_word_topk=2,
            pose_word_hold_segments=1,
            pose_word_cooldown_segments=1,
            pose_word_ema_alpha=1.0,
            pose_word_dedup_same_word=False,
            pose_word_th_no_event=0.6,
            pose_word_th_unknown=0.5,
            pose_word_th_margin=0.1,
            pose_word_no_event_label="_no_event",
            word_hold_frames=6,
        )

    def get_pose_extractor(self) -> _StubPoseExtractor:
        return self._pose_extractor

    def get_bio_segmenter_model(self) -> _SyntheticBioModel:
        return self._bio_model

    def get_pose_word_model(self) -> _StubPoseWordModel:
        return self._pose_word_model



def test_pose_words_segment_completion_triggers_classifier() -> None:
    runtime = _StubRuntime()
    session = SessionProcessor(runtime)  # type: ignore[arg-type]

    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
    payload = None

    for i in range(16):
        payload = session.process_frame(frame_bgr, now_ms=1000 + i * 40)

    assert payload is not None
    assert payload["mode"] == "pose_words"
    assert runtime._pose_word_model.calls >= 1
    assert "segments" in payload
