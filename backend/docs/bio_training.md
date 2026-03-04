# BIO Training (Pose BiLSTM) + ONNX Export

Документ описывает этап обучения BIO-сегментера на synthetic continuous датасете:
- вход: `features [T,F]`, `sign_bio [T]`, `phrase_bio [T]`,
- модель: BiLSTM + две головы (`sign`, `phrase`),
- выход: ONNX + thresholds + config.

## 1) Предусловия

1. Сгенерирован датасет через:
   - `backend/scripts/build_bio_continuous_dataset.py`
2. Структура датасета:

```text
backend/data/bio_continuous/
  train/*.npz
  val/*.npz
  test/*.npz
  dataset_stats.json
```

## 2) Обучение BIO-сегментера

Скрипт:
- `backend/train/train_pose_bio_segmenter.py`

Пример запуска:

```bash
./.venv/bin/python backend/train/train_pose_bio_segmenter.py \
  --data-dir backend/data/bio_continuous \
  --save-dir backend/artifacts/bio_training \
  --thresholds-out backend/artifacts/bio_thresholds.json \
  --epochs 20 \
  --batch-size 32 \
  --hidden-size 256 \
  --num-layers 2 \
  --dropout 0.2 \
  --bidirectional true \
  --alpha-phrase 1.0 \
  --threshold-grid "0.3,0.4,0.5,0.6,0.7" \
  --boundary-tolerance 0 \
  --segment-iou-threshold 0.5 \
  --use-shoulder-norm true \
  --use-hands-3d-norm false
```

Что делает обучение:

1. Загружает `train/val/test` `.npz`.
2. Считает class weights (авто по train или дефолт `B:5, I:1, O:1`).
3. Учит BiLSTM:
   - `sign_head -> [B,T,3]`,
   - `phrase_head -> [B,T,3]`.
4. На каждом epoch подбирает пороги (`th_B`, `th_O`) grid-search на `val`:
   - `th_B ∈ {0.3,0.4,0.5,0.6,0.7}`
   - `th_O ∈ {0.3,0.4,0.5,0.6,0.7}`
5. Сохраняет лучший чекпоинт и `bio_thresholds.json`.

Метрики:
- frame accuracy (`sign_frame_acc`, `phrase_frame_acc`)
- boundary F1 по событиям `B`
- segment F1 и mean IoU после порогового декодирования

## 3) Экспорт в ONNX

Скрипт:
- `backend/train/export_bio_segmenter_onnx.py`

Пример:

```bash
./.venv/bin/python backend/train/export_bio_segmenter_onnx.py \
  --checkpoint backend/artifacts/bio_training/best_model.pt \
  --onnx-out backend/artifacts/bio_segmenter.onnx \
  --thresholds-in backend/artifacts/bio_thresholds.json \
  --thresholds-out backend/artifacts/bio_thresholds.json \
  --bio-config-out backend/artifacts/bio_config.json \
  --window-size 256 \
  --dynamic-time true
```

Результат:
- `backend/artifacts/bio_segmenter.onnx`
- `backend/artifacts/bio_thresholds.json`
- `backend/artifacts/bio_config.json`

## 4) Быстрый pipeline-скрипт

Есть make-like скрипт:
- `backend/scripts/run_bio_training_pipeline.sh`

Запуск:

```bash
backend/scripts/run_bio_training_pipeline.sh \
  backend/data/bio_continuous \
  backend/artifacts
```

Скрипт:
1. запускает обучение;
2. экспортирует ONNX;
3. обновляет thresholds/config в `backend/artifacts`.

## 5) Проверка на test

После обучения смотрите:
- `backend/artifacts/bio_training/train_summary.json`

Разделы:
- `best` — лучшие val-метрики и пороги;
- `test` — test-метрики с лучшими порогами;
- `history` — история по эпохам.

## 6) Артефакты

Обязательные артефакты runtime:
- `backend/artifacts/bio_segmenter.onnx`
- `backend/artifacts/bio_thresholds.json`
- `backend/artifacts/bio_config.json`

Mapping классов зафиксирован:
- `B = 0`, `I = 1`, `O = 2`.

