# Perf Tuning for `pose_words`

Документ про ускорение `pose_words` режима с включенной сегментацией BIO.

## Что влияет на скорость

Основные параметры в `backend/config.yaml`:

- `recognition_mode: pose_words`
- `segmentation.enabled`
- `segmentation.window`
- `segmentation.step`
- `segmentation.ort_num_threads`
- `pose_word_model.ort_num_threads`
- `pose_worker.enabled`
- `pose_worker.queue_size`
- `pose_worker.output_size`
- `perf.enabled`

## Рекомендованные настройки для realtime

```yaml
perf:
  enabled: true
  window_size: 120
  ema_alpha: 0.2

pose_worker:
  enabled: true
  queue_size: 3
  output_size: 4

segmentation:
  enabled: true
  window: 192
  step: 8
  min_len: 6
  merge_gap: 2
  ort_num_threads: 1

pose_word_model:
  ort_num_threads: 1
```

## Практические советы

1. Отключите face landmarks (в проекте это уже дефолт).
2. Уменьшайте `segmentation.window` (например 256 -> 192), если растет задержка.
3. Увеличивайте `segmentation.step` (например 4 -> 8), чтобы реже дергать ONNX.
4. Держите `ort_num_threads` в диапазоне `1..2` для стабильной latency на CPU.
5. Оставляйте `pose_worker.enabled=true` для вынесения MediaPipe из WS-цикла.
6. Следите за `dropped_frames_count`: если слишком высокий, повышайте `step` или снижайте входной FPS.

## Что смотреть в payload

При `perf.enabled=true` в WS payload появляется опциональный блок:

- `perf.latency_ms`
- `perf.fps_in`
- `perf.fps_pose`
- `perf.fps_total`
- `perf.dropped_frames_count`
- stage-метрики (EMA/avg/p95/count), например:
  - `decode_jpeg_ms_ema`
  - `mediapipe_ms_ema`
  - `bio_infer_ms_ema`
  - `word_infer_ms_ema`
  - `ws_send_ms_ema`
  - `total_ms_ema`

Также в `payload.debug.bio` есть технические поля сегментации, включая:

- `buffer_len`
- `active_sign_progress`
- `dropped_frames_count`

## Как проверить изменения

```bash
python backend/scripts/smoke_pose_words.py \
  --base-url http://127.0.0.1:8000 \
  --duration-sec 30 \
  --fps 12 \
  --require-pose-words true \
  --require-segmentation true
```

Сравнивайте до/после:

- `latency_avg_ms`
- `latency_p95_ms`
- `stream_fps`
- `unique_segment_events`
- `dropped_frames_count`
