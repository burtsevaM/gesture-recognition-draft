from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.main import SessionProcessor
from app.pose.datatypes import PoseFrame, PoseLandmarksGroup
from app.segmentation.decoder import decode_segments
from app.segmentation.streaming import StreamingBioSegmenter


class _SyntheticBioModel:
    def infer(self, features_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        features = np.asarray(features_tf, dtype=np.float32)
        t_len = int(features.shape[0])
        sign = np.zeros((t_len, 3), dtype=np.float32)
        phrase = np.zeros((t_len, 3), dtype=np.float32)
        sign[:, 2] = 0.9
        sign[:, 0] = 0.05
        sign[:, 1] = 0.05
        phrase[:, 2] = 0.9
        phrase[:, 0] = 0.05
        phrase[:, 1] = 0.05

        for i in range(t_len):
            frame_idx = int(round(float(features[i, 0])))
            if frame_idx in {3, 10}:
                sign[i] = [0.92, 0.05, 0.03]
            elif frame_idx in {4, 5, 11, 12}:
                sign[i] = [0.08, 0.84, 0.08]
            if frame_idx == 2:
                phrase[i] = [0.91, 0.05, 0.04]
            elif frame_idx in {3, 4, 5, 6, 7, 8}:
                phrase[i] = [0.08, 0.84, 0.08]
        return sign, phrase, 2.5


class _StubPoseExtractor:
    def __init__(self) -> None:
        self._counter = 0

    def process(self, rgb_frame: np.ndarray) -> PoseFrame:
        c = float(self._counter)
        self._counter += 1
        body = np.zeros((33, 3), dtype=np.float32)
        body[0] = [c, 0.0, 0.0]
        body[11] = [0.4, 0.5, 0.0]
        body[12] = [0.6, 0.5, 0.0]
        return PoseFrame(
            timestamp=1.0 + c,
            body=PoseLandmarksGroup(points=body, confidence=np.ones((33,), dtype=np.float32)),
            left_hand=None,
            right_hand=None,
            meta={},
        )


class _StubPoseWordModel:
    labels = ["_no_event", "ТЕСТ"]

    def find_no_event_index(self, label_name: str) -> int | None:
        return 0

    def infer_probs(self, features_tf: np.ndarray) -> tuple[np.ndarray, float]:
        probs = np.asarray([0.05, 0.95], dtype=np.float32)
        return probs, 1.5


class _StubRuntime:
    def __init__(self) -> None:
        self.errors: dict[str, str] = {}
        self._pose_extractor = _StubPoseExtractor()
        self._bio_model = _SyntheticBioModel()
        self._pose_word = _StubPoseWordModel()
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
            pose_word_th_unknown=0.55,
            pose_word_th_margin=0.1,
            pose_word_no_event_label="_no_event",
            word_hold_frames=6,
        )

    def get_pose_extractor(self) -> _StubPoseExtractor:
        return self._pose_extractor

    def get_bio_segmenter_model(self) -> _SyntheticBioModel:
        return self._bio_model

    def get_pose_word_model(self) -> _StubPoseWordModel:
        return self._pose_word


def test_decode_segments_and_streaming_output() -> None:
    model = _SyntheticBioModel()
    segmenter = StreamingBioSegmenter(
        model=model,  # type: ignore[arg-type]
        window=8,
        step=1,
        min_len=2,
        merge_gap=0,
        sign_th_b=0.5,
        sign_th_o=0.5,
        phrase_th_b=0.5,
        phrase_th_o=0.5,
        max_buffer=64,
    )

    all_sign: list[tuple[int, int]] = []
    for idx in range(15):
        feature = np.asarray([float(idx), 0.0, 0.0], dtype=np.float32)
        result = segmenter.update(feature)
        for seg in result.sign_segments:
            all_sign.append((seg.start, seg.end))

    assert (3, 5) in all_sign
    assert (10, 12) in all_sign

    synthetic_probs = np.zeros((8, 3), dtype=np.float32)
    synthetic_probs[:, 2] = 0.9
    synthetic_probs[2] = [0.92, 0.05, 0.03]
    synthetic_probs[3] = [0.05, 0.9, 0.05]
    synthetic_probs[4] = [0.05, 0.9, 0.05]
    synthetic_probs[5] = [0.02, 0.03, 0.95]
    segments = decode_segments(synthetic_probs, th_B=0.5, th_O=0.5, min_len=2, merge_gap=0)
    assert len(segments) == 1
    assert segments[0][0] == 2
    assert segments[0][1] == 4
    assert 0.0 <= float(segments[0][2]) <= 1.0


def test_pose_words_payload_contains_segments_when_enabled() -> None:
    runtime = _StubRuntime()
    session = SessionProcessor(runtime)  # type: ignore[arg-type]
    frame_bgr = np.zeros((64, 64, 3), dtype=np.uint8)

    payload = None
    for i in range(10):
        payload = session.process_frame(frame_bgr, now_ms=1000 + i * 33)

    assert payload is not None
    assert payload["mode"] == "pose_words"
    assert "segments" in payload
    assert "sign" in payload["segments"]
    assert "phrase" in payload["segments"]
    assert "skeleton" in payload
