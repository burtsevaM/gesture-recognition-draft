from __future__ import annotations

import time

import numpy as np

from app.pose import PoseFrame, PoseLandmarksGroup, PosePipelineWorker


class _StubExtractor:
    def __init__(self) -> None:
        self._counter = 0

    def process(self, rgb_frame: np.ndarray) -> PoseFrame:
        self._counter += 1
        body = np.zeros((33, 3), dtype=np.float32)
        body[11] = [0.4, 0.5, 0.0]
        body[12] = [0.6, 0.5, 0.0]
        body[0] = [float(self._counter), 0.0, 0.0]
        return PoseFrame(
            timestamp=float(self._counter),
            body=PoseLandmarksGroup(points=body, confidence=np.ones((33,), dtype=np.float32)),
            left_hand=None,
            right_hand=None,
            meta={},
        )



def test_worker_latest_only_drop_policy_and_stop() -> None:
    worker = PosePipelineWorker(
        extractor=_StubExtractor(),  # type: ignore[arg-type]
        use_shoulder_norm=False,
        use_hands_3d_norm=False,
        queue_size=1,
        output_size=2,
        perf_enabled=True,
    )
    worker.start()

    latest_frame_id = 0
    for i in range(30):
        frame = np.full((48, 64, 3), i % 255, dtype=np.uint8)
        latest_frame_id = worker.submit_frame(frame, now_ms=1000 + i)

    deadline = time.time() + 2.0
    latest_result = None
    while time.time() < deadline:
        result = worker.get_latest_result()
        if result is not None:
            latest_result = result
        if worker.processed_frames > 0 and worker.dropped_frames_count > 0 and latest_result is not None:
            break
        time.sleep(0.01)

    assert worker.processed_frames > 0
    assert worker.dropped_frames_count > 0
    assert latest_result is not None
    assert latest_result.frame_id <= latest_frame_id
    assert latest_result.frame_id >= max(1, latest_frame_id - 10)

    worker.stop(timeout_sec=1.0)



def test_worker_no_deadlock_on_rapid_stop() -> None:
    worker = PosePipelineWorker(
        extractor=_StubExtractor(),  # type: ignore[arg-type]
        use_shoulder_norm=False,
        use_hands_3d_norm=False,
        queue_size=2,
        output_size=2,
        perf_enabled=False,
    )
    worker.start()
    worker.submit_frame(np.zeros((32, 32, 3), dtype=np.uint8), now_ms=1)
    worker.stop(timeout_sec=1.0)
    # second stop should be safe no-op
    worker.stop(timeout_sec=1.0)
