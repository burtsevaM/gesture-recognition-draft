from __future__ import annotations

from pathlib import Path

from app.config import AppConfig, load_config, merge_config_values


def test_load_defaults_when_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / 'missing.yaml')
    assert isinstance(cfg, AppConfig)
    assert cfg.hold_ms == 700


def test_merge_config_values(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text('hold_ms: 100\nsim_none: 0.2\n', encoding='utf-8')

    merged = merge_config_values(path, {'sim_none': 0.45})
    assert merged.hold_ms == 100
    assert merged.sim_none == 0.45


def test_nested_words_config_parsing(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        """
recognition_mode: words
word_model:
  path: backend/artifacts/slovo_word_model.onnx
  labels_path: backend/artifacts/labels.txt
  input_size: 224
  window_frames: 32
thresholds:
  no_event_label: no_event
  th_no_event: 0.61
  th_unknown: 0.57
  th_margin: 0.12
""".strip(),
        encoding='utf-8',
    )
    cfg = load_config(path)
    assert cfg.recognition_mode == 'words'
    assert cfg.word_input_size == 224
    assert cfg.word_window_frames == 32
    assert cfg.word_th_no_event == 0.61
    assert cfg.word_th_unknown == 0.57
    assert cfg.word_th_margin == 0.12


def test_pose_words_flags_parsing(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        """
recognition_mode: pose_words
use_shoulder_norm: false
use_hands_3d_norm: true
""".strip(),
        encoding='utf-8',
    )
    cfg = load_config(path)
    assert cfg.recognition_mode == 'pose_words'
    assert cfg.use_shoulder_norm is False
    assert cfg.use_hands_3d_norm is True


def test_pose_words_segmentation_parsing(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        """
recognition_mode: pose_words
segmentation:
  enabled: true
  model_path: backend/artifacts/bio_segmenter.onnx
  thresholds_path: backend/artifacts/bio_thresholds.json
  window: 192
  step: 6
  min_len: 5
  max_len: 120
  merge_gap: 1
  cool_off_frames: 3
  max_buffer: 384
pose_word_model:
  path: backend/artifacts/pose_word_model.onnx
  labels_path: backend/artifacts/pose_word_labels.txt
  clip_frames: 24
  topk: 7
pose_word_thresholds:
  no_event_label: "---"
  th_no_event: 0.62
  th_unknown: 0.59
  th_margin: 0.11
pose_word_commit_logic:
  ema_alpha: 0.25
  hold_segments: 3
  cooldown_segments: 4
  dedup_same_word: false
""".strip(),
        encoding='utf-8',
    )
    cfg = load_config(path)
    assert cfg.recognition_mode == 'pose_words'
    assert cfg.segmentation_enabled is True
    assert cfg.segmentation_window == 192
    assert cfg.segmentation_step == 6
    assert cfg.segmentation_min_len == 5
    assert cfg.segmentation_max_len == 120
    assert cfg.segmentation_merge_gap == 1
    assert cfg.segmentation_cool_off_frames == 3
    assert cfg.segmentation_max_buffer == 384
    assert cfg.pose_word_clip_frames == 24
    assert cfg.pose_word_topk == 7
    assert cfg.pose_word_no_event_label == '---'
    assert cfg.pose_word_th_no_event == 0.62
    assert cfg.pose_word_th_unknown == 0.59
    assert cfg.pose_word_th_margin == 0.11
    assert cfg.pose_word_ema_alpha == 0.25
    assert cfg.pose_word_hold_segments == 3
    assert cfg.pose_word_cooldown_segments == 4
    assert cfg.pose_word_dedup_same_word is False
