from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time

import numpy as np

from ..pose_words.segment_utils import extract_segment
from .decoder import decode_segments
from .model_onnx import BioSegmenterOnnxModel


@dataclass(slots=True)
class BioSegment:
    start: int
    end: int
    score: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "score": float(self.score),
        }


@dataclass(slots=True)
class StreamingBioResult:
    ran_inference: bool
    sign_segments: list[BioSegment] = field(default_factory=list)
    phrase_segments: list[BioSegment] = field(default_factory=list)
    recent_sign_segments: list[BioSegment] = field(default_factory=list)
    recent_phrase_segments: list[BioSegment] = field(default_factory=list)
    active_sign: bool = False
    active_phrase: bool = False
    active_sign_progress: float = 0.0
    active_phrase_progress: float = 0.0
    latency_ms: float | None = None
    decode_latency_ms: float | None = None
    buffer_len: int = 0
    buffer_start: int = 0
    buffer_end: int = -1
    index_mode: str = "global"


class StreamingBioSegmenter:
    """Streaming BIO segmenter with global monotonic frame indices.

    Segment indices (`start`, `end`) in results are always global frame indices,
    not local buffer offsets.
    """

    def __init__(
        self,
        *,
        model: BioSegmenterOnnxModel,
        window: int = 256,
        step: int = 8,
        min_len: int = 6,
        max_len: int = 150,
        merge_gap: int = 2,
        cool_off_frames: int = 0,
        sign_th_b: float = 0.5,
        sign_th_o: float = 0.5,
        phrase_th_b: float = 0.5,
        phrase_th_o: float = 0.5,
        max_buffer: int | None = None,
    ) -> None:
        self.model = model
        self.window = max(8, int(window))
        self.step = max(1, int(step))
        self.min_len = max(1, int(min_len))
        self.max_len = max(self.min_len, int(max_len))
        self.merge_gap = max(0, int(merge_gap))
        self.cool_off_frames = max(0, int(cool_off_frames))
        self.sign_th_b = float(sign_th_b)
        self.sign_th_o = float(sign_th_o)
        self.phrase_th_b = float(phrase_th_b)
        self.phrase_th_o = float(phrase_th_o)

        self.max_buffer = max(self.window * 2, int(max_buffer) if max_buffer else self.window * 2)
        self._features: deque[np.ndarray] = deque(maxlen=self.max_buffer)
        self._frame_indices: deque[int] = deque(maxlen=self.max_buffer)
        self._next_frame_idx = 0
        self._frames_since_infer = 0

        self._sign_sum: dict[int, np.ndarray] = {}
        self._phrase_sum: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

        self._latest_sign_segments: list[BioSegment] = []
        self._latest_phrase_segments: list[BioSegment] = []
        self._last_emitted_sign_end = -1
        self._last_emitted_phrase_end = -1

    @property
    def has_enough_frames(self) -> bool:
        return len(self._features) >= self.window

    def _append_feature(self, feature: np.ndarray) -> None:
        feat = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(feat)):
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        idx = int(self._next_frame_idx)
        self._next_frame_idx += 1
        self._features.append(feat)
        self._frame_indices.append(idx)
        self._frames_since_infer += 1

    def _prune_aggregation(self) -> None:
        if not self._frame_indices:
            self._sign_sum.clear()
            self._phrase_sum.clear()
            self._counts.clear()
            return
        min_idx = int(self._frame_indices[0])
        drop_keys = [k for k in self._counts.keys() if k < min_idx]
        for key in drop_keys:
            self._counts.pop(key, None)
            self._sign_sum.pop(key, None)
            self._phrase_sum.pop(key, None)

    def _update_aggregation(self, frame_indices: list[int], sign_probs: np.ndarray, phrase_probs: np.ndarray) -> None:
        for offset, frame_idx in enumerate(frame_indices):
            s = np.asarray(sign_probs[offset], dtype=np.float32).reshape(3)
            p = np.asarray(phrase_probs[offset], dtype=np.float32).reshape(3)
            if frame_idx not in self._counts:
                self._counts[frame_idx] = 0
                self._sign_sum[frame_idx] = np.zeros((3,), dtype=np.float32)
                self._phrase_sum[frame_idx] = np.zeros((3,), dtype=np.float32)
            self._counts[frame_idx] += 1
            self._sign_sum[frame_idx] += s
            self._phrase_sum[frame_idx] += p

    def _averaged_probs(self) -> tuple[list[int], np.ndarray, np.ndarray]:
        if not self._frame_indices:
            return [], np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

        indices = list(self._frame_indices)
        sign = np.zeros((len(indices), 3), dtype=np.float32)
        phrase = np.zeros((len(indices), 3), dtype=np.float32)
        for i, idx in enumerate(indices):
            cnt = max(1, int(self._counts.get(idx, 1)))
            sign_sum = self._sign_sum.get(idx)
            phrase_sum = self._phrase_sum.get(idx)
            if sign_sum is None or phrase_sum is None:
                sign[i, 2] = 1.0
                phrase[i, 2] = 1.0
                continue
            sign[i] = (sign_sum / float(cnt)).astype(np.float32)
            phrase[i] = (phrase_sum / float(cnt)).astype(np.float32)
        return indices, sign, phrase

    def _decode_segments_for_buffer(self) -> tuple[list[BioSegment], list[BioSegment], bool, bool, float, float]:
        idx, sign_probs, phrase_probs = self._averaged_probs()
        if not idx:
            return [], [], False, False, 0.0, 0.0

        sign_local = decode_segments(
            sign_probs,
            th_B=self.sign_th_b,
            th_O=self.sign_th_o,
            min_len=self.min_len,
            max_len=self.max_len,
            merge_gap=self.merge_gap,
        )
        phrase_local = decode_segments(
            phrase_probs,
            th_B=self.phrase_th_b,
            th_O=self.phrase_th_o,
            min_len=self.min_len,
            max_len=self.max_len,
            merge_gap=self.merge_gap,
        )

        sign_segments = [BioSegment(start=idx[s], end=idx[e], score=float(score)) for s, e, score in sign_local]
        phrase_segments = [BioSegment(start=idx[s], end=idx[e], score=float(score)) for s, e, score in phrase_local]

        latest_idx = idx[-1]
        active_sign = bool(sign_segments and sign_segments[-1].end >= latest_idx)
        active_phrase = bool(phrase_segments and phrase_segments[-1].end >= latest_idx)

        sign_progress = 0.0
        if active_sign:
            seg_len = max(1, sign_segments[-1].end - sign_segments[-1].start + 1)
            sign_progress = min(1.0, seg_len / float(self.min_len))

        phrase_progress = 0.0
        if active_phrase:
            seg_len = max(1, phrase_segments[-1].end - phrase_segments[-1].start + 1)
            phrase_progress = min(1.0, seg_len / float(self.min_len))

        return sign_segments, phrase_segments, active_sign, active_phrase, sign_progress, phrase_progress

    def _apply_cool_off(self, segments: list[BioSegment], last_end: int) -> tuple[list[BioSegment], int]:
        if not segments:
            return [], last_end
        out: list[BioSegment] = []
        cursor_end = int(last_end)
        for seg in sorted(segments, key=lambda x: (x.end, x.start)):
            if seg.end <= cursor_end:
                continue
            if cursor_end >= 0 and (seg.start - cursor_end - 1) < self.cool_off_frames:
                continue
            out.append(seg)
            cursor_end = seg.end
        return out, cursor_end

    def get_feature_span(self, start: int, end: int) -> np.ndarray | None:
        if not self._frame_indices:
            return None
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            return None

        min_idx = int(self._frame_indices[0])
        max_idx = int(self._frame_indices[-1])
        if start_i < min_idx or end_i > max_idx:
            return None
        feat_mat = np.stack(list(self._features), axis=0).astype(np.float32)
        segment = extract_segment({"features": feat_mat, "start_idx": min_idx}, start_i, end_i)
        if segment.shape[0] == 0:
            return None
        return segment

    def update(self, feature: np.ndarray) -> StreamingBioResult:
        self._append_feature(feature)
        self._prune_aggregation()

        if not self.has_enough_frames:
            start = int(self._frame_indices[0]) if self._frame_indices else 0
            end = int(self._frame_indices[-1]) if self._frame_indices else -1
            return StreamingBioResult(
                ran_inference=False,
                recent_sign_segments=list(self._latest_sign_segments),
                recent_phrase_segments=list(self._latest_phrase_segments),
                buffer_len=len(self._frame_indices),
                buffer_start=start,
                buffer_end=end,
                index_mode="global",
            )

        should_run = self._frames_since_infer >= self.step
        if not should_run:
            start = int(self._frame_indices[0]) if self._frame_indices else 0
            end = int(self._frame_indices[-1]) if self._frame_indices else -1
            return StreamingBioResult(
                ran_inference=False,
                recent_sign_segments=list(self._latest_sign_segments),
                recent_phrase_segments=list(self._latest_phrase_segments),
                buffer_len=len(self._frame_indices),
                buffer_start=start,
                buffer_end=end,
                index_mode="global",
            )

        self._frames_since_infer = 0
        window_feats = np.stack(list(self._features)[-self.window :], axis=0).astype(np.float32)
        window_indices = list(self._frame_indices)[-self.window :]
        sign_probs, phrase_probs, latency_ms = self.model.infer(window_feats)
        self._update_aggregation(window_indices, sign_probs, phrase_probs)
        self._prune_aggregation()

        decode_started = time.perf_counter()
        all_sign, all_phrase, active_sign, active_phrase, sign_progress, phrase_progress = self._decode_segments_for_buffer()
        decode_latency_ms = (time.perf_counter() - decode_started) * 1000.0
        self._latest_sign_segments = list(all_sign)
        self._latest_phrase_segments = list(all_phrase)

        latest_idx = int(self._frame_indices[-1])
        completed_sign = [seg for seg in all_sign if seg.end < latest_idx]
        completed_phrase = [seg for seg in all_phrase if seg.end < latest_idx]

        new_sign = [seg for seg in completed_sign if seg.end > self._last_emitted_sign_end]
        new_phrase = [seg for seg in completed_phrase if seg.end > self._last_emitted_phrase_end]

        new_sign, self._last_emitted_sign_end = self._apply_cool_off(new_sign, self._last_emitted_sign_end)
        new_phrase, self._last_emitted_phrase_end = self._apply_cool_off(new_phrase, self._last_emitted_phrase_end)

        start = int(self._frame_indices[0]) if self._frame_indices else 0
        end = int(self._frame_indices[-1]) if self._frame_indices else -1
        return StreamingBioResult(
            ran_inference=True,
            sign_segments=new_sign,
            phrase_segments=new_phrase,
            recent_sign_segments=all_sign,
            recent_phrase_segments=all_phrase,
            active_sign=active_sign,
            active_phrase=active_phrase,
            active_sign_progress=float(sign_progress),
            active_phrase_progress=float(phrase_progress),
            latency_ms=float(latency_ms),
            decode_latency_ms=float(decode_latency_ms),
            buffer_len=len(self._frame_indices),
            buffer_start=start,
            buffer_end=end,
            index_mode="global",
        )
