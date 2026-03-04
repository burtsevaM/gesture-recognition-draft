# Pose Model + BIO Plan (Scaffolding)

Этот документ фиксирует следующий этап после текущего MVP (pose extraction + normalization), но без реализации обучения в этом PR.

## 1) Slovo -> pose dataset

Цель: получить датасет последовательностей ключевых точек (body + hands), чтобы учить модели на позах, а не на RGB.

Рекомендуемый поток:

1. Скачать/подготовить Slovo локально.
2. Для каждого видео извлечь landmarks через MediaPipe Holistic.
3. Применить нормализацию:
   - центр по midpoint плеч,
   - масштаб по расстоянию плеч,
   - опционально 3D-нормализация кистей.
4. Сохранить последовательности в pose-формат (или jsonl/npy + manifest).
5. Сделать split строго по signer (без утечки между train/val/test).

Полезный reference по формату поз и инструментам:
- pose-format: [https://github.com/sign-language-processing/pose](https://github.com/sign-language-processing/pose)

Минимальные поля в manifest для каждого sample:
- `sample_id`
- `video_path` (или `pose_path`)
- `label` (слово)
- `signer_id`
- `num_frames`
- `fps`
- `split` (`train`/`val`/`test`)

## 2) Обучение pose word classifier (isolated words)

Задача: классификатор по последовательности поз на словарь слов (например 50/100/200 или весь набор).

Вход модели:
- `X`: `[T, F]` (T кадров, F признаков из body+hands + опционально velocity)
- `mask`: `[T]` или `[T,1]` (валидные кадры)

Выход:
- вероятности классов слов + `no_event`/`unknown` на этапе декодирования.

Минимальный план:

1. Loader читает pose-sequence + labels.
2. Модель (базовый вариант):
   - Temporal Conv / BiGRU / Transformer-encoder (легкий CPU-friendly).
3. Loss: cross-entropy.
4. Метрики: top-1, top-5, per-class recall, confusion matrix.
5. Экспорт лучшего чекпоинта + labels.

Каркас под это:
- `backend/train/train_pose_word_model.py`
- `backend/train/export_pose_word_onnx.py`

## 3) BIO сегментация (B/I/O) и декодирование

Задача: онлайн-находить границы жеста в потоке:
- `B` — начало жеста,
- `I` — продолжение,
- `O` — вне жеста.

Пайплайн:

1. Для каждого кадра/окна модель сегментации дает `p(B), p(I), p(O)`.
2. Сглаживание вероятностей (EMA/temporal smoothing).
3. Пороговый декодер:
   - старт сегмента, если `p(B) >= th_B`,
   - продолжаем, пока `p(I) >= th_I`,
   - закрываем, когда `p(O) >= th_O` N кадров подряд.
4. На выделенный сегмент запускается classifier слова.

Что важно хранить в порогах:
- `th_B`, `th_I`, `th_O`
- `min_segment_frames`
- `max_segment_frames`
- `close_on_o_streak`
- `cooldown_frames`

Каркас под это:
- `backend/train/train_pose_bio_segmenter.py`

## 4) Артефакты и структура хранения

Рекомендуемая структура `backend/artifacts/`:

- `pose_word_model.onnx` — классификатор слов по позам
- `pose_word_labels.txt` — список слов (порядок = индекс выхода модели)
- `pose_bio_model.onnx` — сегментатор BIO
- `pose_bio_thresholds.yaml` — пороги и параметры декодера
- `pose_norm_config.yaml` — конфиг нормализации/фичей
- `pose_metrics.json` — итоговые метрики val/test

Минимальные требования к reproducibility:
- фиксированный `seed`,
- сохранение версии датасета/split,
- сохранение конфигов тренировки и инференса рядом с моделью.

## 5) Что уже добавлено в код как scaffolding

- `backend/train/train_pose_word_model.py` — entrypoint-каркас для обучения classifier.
- `backend/train/export_pose_word_onnx.py` — entrypoint-каркас для экспорта в ONNX.
- `backend/train/train_pose_bio_segmenter.py` — entrypoint-каркас для обучения BIO segmenter.

Все три скрипта пока не обучают модель, а фиксируют CLI и ожидаемые входы/выходы следующего этапа.
