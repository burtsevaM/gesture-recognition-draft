from __future__ import annotations

from app.schemas import TopKItem, VLMDecision, build_inference_message


REQUIRED_TOP_LEVEL = {
    "mode",
    "status",
    "letter",
    "word",
    "score",
    "confidence",
    "hand_present",
    "bbox_norm",
    "hold",
    "text_state",
    "topk",
    "vlm",
    "debug",
}


def _assert_required_payload_fields(payload: dict) -> None:
    assert REQUIRED_TOP_LEVEL.issubset(payload.keys())
    assert isinstance(payload["bbox_norm"], list)
    assert isinstance(payload["hold"], dict)
    assert isinstance(payload["text_state"], dict)
    assert isinstance(payload["topk"], list)
    assert isinstance(payload["vlm"], dict)
    assert isinstance(payload["debug"], dict)

    hold = payload["hold"]
    assert {"elapsed_ms", "remaining_ms", "target_ms", "progress", "unit"}.issubset(hold.keys())

    text_state = payload["text_state"]
    assert {"value", "committed"}.issubset(text_state.keys())


def test_inference_message_shape() -> None:
    payload = build_inference_message(
        status='CANDIDATE',
        letter='А',
        score=0.77,
        confidence=0.77,
        hand_present=True,
        bbox_norm=[0.1, 0.2, 0.3, 0.4],
        hold_elapsed_ms=350,
        hold_target_ms=700,
        text_value='А',
        committed_now=False,
        topk=[TopKItem(letter='А', score=0.77, exemplar_path='/tmp/a.jpg')],
        vlm=VLMDecision(used=False),
        sim1=0.77,
        sim2=0.61,
        margin=0.16,
        uncertain=False,
        cooldown_left_ms=0,
    )

    _assert_required_payload_fields(payload)
    assert payload["status"] == "CANDIDATE"
    assert payload["hold"]["remaining_ms"] == 350
    assert payload["text_state"]["value"] == "А"
    assert "skeleton" not in payload
    assert "segments" not in payload
    assert "perf" not in payload
    assert "bio" not in payload


def test_ws_contract_words_mode_required_fields() -> None:
    payload = build_inference_message(
        status="HOLD",
        letter="HELLO",
        word="HELLO",
        score=0.91,
        confidence=0.91,
        hand_present=True,
        bbox_norm=[0.1, 0.2, 0.3, 0.4],
        hold_elapsed_ms=3,
        hold_target_ms=6,
        text_value="",
        committed_now=False,
        topk=[TopKItem(letter="HELLO", score=0.91, exemplar_path="")],
        vlm=VLMDecision(used=False),
        sim1=0.91,
        sim2=0.73,
        margin=0.18,
        uncertain=False,
        cooldown_left_ms=0,
        mode="words",
        hold_unit="frames",
        latency_ms=15.0,
    )

    _assert_required_payload_fields(payload)
    assert payload["mode"] == "words"
    assert payload["word"] == "HELLO"
    assert "skeleton" not in payload
    assert "segments" not in payload
    assert "perf" not in payload
    assert "bio" not in payload


def test_ws_contract_optional_pose_fields() -> None:
    payload = build_inference_message(
        status="POSE",
        letter="NONE",
        word="NONE",
        score=0.0,
        confidence=0.0,
        hand_present=True,
        bbox_norm=[0.0, 0.0, 0.0, 0.0],
        hold_elapsed_ms=0,
        hold_target_ms=1,
        text_value="",
        committed_now=False,
        topk=[],
        vlm=VLMDecision(used=False),
        sim1=0.0,
        sim2=0.0,
        margin=0.0,
        uncertain=False,
        cooldown_left_ms=0,
        mode="pose_words",
        hold_unit="segments",
    )
    payload["skeleton"] = {
        "raw": {"body": None, "lh": None, "rh": None},
        "norm": {"body": None, "lh": None, "rh": None},
    }
    payload["segments"] = {"sign": [], "phrase": []}
    payload["perf"] = {"latency_ms": 20.0, "fps_in": 12.0, "fps_pose": 10.0, "fps_total": 9.0}
    payload["bio"] = {"enabled": True, "th_B": 0.5, "th_O": 0.5, "window": 256, "step": 8}

    _assert_required_payload_fields(payload)
    assert isinstance(payload["skeleton"], dict)
    assert isinstance(payload["segments"], dict)
    assert isinstance(payload["perf"], dict)
    assert isinstance(payload["bio"], dict)
