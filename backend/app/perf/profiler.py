from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


PERF_STAGES: tuple[str, ...] = (
    "decode_jpeg_ms",
    "mediapipe_ms",
    "normalize_ms",
    "feature_ms",
    "bio_infer_ms",
    "bio_decode_ms",
    "word_infer_ms",
    "decoder_ms",
    "ws_send_ms",
    "total_ms",
)


@dataclass(slots=True)
class StageStats:
    ema: float = 0.0
    avg: float = 0.0
    p95: float = 0.0
    count: int = 0


class PerfAggregator:
    """Thread-safe rolling statistics with EMA for named stages."""

    def __init__(
        self,
        *,
        stage_names: tuple[str, ...] = PERF_STAGES,
        window_size: int = 120,
        ema_alpha: float = 0.2,
    ) -> None:
        self.stage_names = tuple(stage_names)
        self.window_size = max(1, int(window_size))
        self.ema_alpha = min(1.0, max(0.01, float(ema_alpha)))
        self._lock = threading.Lock()
        self._history: dict[str, deque[float]] = {name: deque(maxlen=self.window_size) for name in self.stage_names}
        self._ema: dict[str, float] = {name: 0.0 for name in self.stage_names}
        self._counts: dict[str, int] = {name: 0 for name in self.stage_names}

    def update(self, values_ms: dict[str, float]) -> None:
        with self._lock:
            for stage, value in values_ms.items():
                if stage not in self._history:
                    continue
                v = float(value)
                self._history[stage].append(v)
                self._counts[stage] += 1
                if self._counts[stage] == 1:
                    self._ema[stage] = v
                else:
                    prev = self._ema[stage]
                    self._ema[stage] = (1.0 - self.ema_alpha) * prev + self.ema_alpha * v

    def snapshot(self) -> dict[str, StageStats]:
        with self._lock:
            out: dict[str, StageStats] = {}
            for stage in self.stage_names:
                hist = self._history[stage]
                if hist:
                    values = list(hist)
                    values_sorted = sorted(values)
                    idx = int(round(0.95 * (len(values_sorted) - 1)))
                    p95 = values_sorted[max(0, min(idx, len(values_sorted) - 1))]
                    avg = float(sum(values) / len(values))
                else:
                    avg = 0.0
                    p95 = 0.0
                out[stage] = StageStats(
                    ema=float(self._ema[stage]),
                    avg=float(avg),
                    p95=float(p95),
                    count=int(self._counts[stage]),
                )
            return out

    def snapshot_flat(self) -> dict[str, float | int]:
        snap = self.snapshot()
        out: dict[str, float | int] = {}
        for stage, stats in snap.items():
            out[f"{stage}_ema"] = float(stats.ema)
            out[f"{stage}_avg"] = float(stats.avg)
            out[f"{stage}_p95"] = float(stats.p95)
            out[f"{stage}_count"] = int(stats.count)
        return out


class FrameProfiler:
    """Per-frame stage timer that can feed the shared aggregator."""

    def __init__(self, aggregator: PerfAggregator, *, enabled: bool = True) -> None:
        self.aggregator = aggregator
        self.enabled = bool(enabled)
        self._start_s = time.perf_counter()
        self._values: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._values[name] = float(max(0.0, elapsed_ms))

    def record(self, name: str, value_ms: float | int | None) -> None:
        if not self.enabled or value_ms is None:
            return
        self._values[name] = float(max(0.0, float(value_ms)))

    def finish(self) -> dict[str, float]:
        if self.enabled and "total_ms" not in self._values:
            self._values["total_ms"] = float(max(0.0, (time.perf_counter() - self._start_s) * 1000.0))
        if self.enabled:
            self.aggregator.update(self._values)
        return dict(self._values)


__all__ = [
    "PERF_STAGES",
    "StageStats",
    "PerfAggregator",
    "FrameProfiler",
]
