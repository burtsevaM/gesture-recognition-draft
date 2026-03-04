# Synthetic Continuous BIO Dataset

Скрипт: `backend/scripts/build_bio_continuous_dataset.py`

Назначение: из isolated pose-клипов (например Slovo) собрать длинные synthetic continuous последовательности и автоматически проставить:
- `sign_bio` (B/I/O на уровне слова),
- `phrase_bio` (B/I/O на уровне фразы).

## Входные данные

`--pose_index` принимает:

1. `index.jsonl` (рекомендуется), где каждая строка содержит минимум:
   - `path`/`features_path`/`pose_path`
   - `label`
   - `signer_id` (желательно)
   - `fps` (опционально)
2. Каталог с файлами `.npz/.npy/.json/.pose` (скрипт попробует собрать индекс автоматически).

Поддерживаемые форматы клипов:
- `.npz` с `features: [T,F]` (основной вариант),
- `.npz` с raw landmarks `body:[T,33,3]`, `left_hand:[T,21,3]`, `right_hand:[T,21,3]` (скрипт сам соберет features через `compose_features_sequence`),
- `.npy`/`.json` с `features`,
- `.pose` при установленном `pose-format`.

## Выход

Структура:

```text
out_dir/
  train/*.npz
  val/*.npz
  test/*.npz
  dataset_stats.json
```

Каждый `.npz` содержит:
- `features: float32 [T,F]`
- `sign_bio: int64 [T]` (`B=0, I=1, O=2`)
- `phrase_bio: int64 [T]`
- `meta: json string` (слова, границы, фразы, signer, seed)

## Базовый запуск

```bash
./.venv/bin/python backend/scripts/build_bio_continuous_dataset.py \
  --pose_index backend/data/slovo_pose/index.jsonl \
  --out_dir backend/data/bio_continuous \
  --min_words 3 --max_words 12 \
  --pause_min 3 --pause_max 12 \
  --phrase_min_words 2 --phrase_max_words 6 \
  --seed 42 \
  --splits_by_signer true \
  --resample_fps 25 \
  --use_shoulder_norm true \
  --use_hands_3d_norm false
```

## Пример с фиксированным числом сэмплов на split

```bash
./.venv/bin/python backend/scripts/build_bio_continuous_dataset.py \
  --pose_index backend/data/slovo_pose/index.jsonl \
  --out_dir backend/data/bio_continuous_small \
  --min_words 4 --max_words 8 \
  --pause_min 2 --pause_max 6 \
  --phrase_min_words 2 --phrase_max_words 4 \
  --splits_by_signer false \
  --train_samples 300 --val_samples 60 --test_samples 60 \
  --seed 7
```

## Ключевые параметры

- `--min_words / --max_words`: сколько слов склеиваем в одну последовательность.
- `--pause_min / --pause_max`: длина пауз между словами (в кадрах), паузы маркируются `O`.
- `--phrase_min_words / --phrase_max_words`: размер группировки слов во фразы.
- `--pause_inside_phrase_as_i`: если `true`, паузы внутри фразы помечаются как `I` (по умолчанию `false`, то есть `O`).
- `--splits_by_signer`: делать split по signer (без утечки) или случайный.
- `--resample_fps` и `--window_fps`: целевая частота кадров для всех клипов.
- `--use_shoulder_norm`, `--use_hands_3d_norm`: включают нормализацию при сборке features из raw landmarks.

## Что проверять после генерации

1. `dataset_stats.json`:
   - доли `B/I/O`,
   - длины последовательностей,
   - средняя длина паузы.
2. Наличие сплитов `train/val/test`.
3. Баланс слов и signer leakage (если `splits_by_signer=true`, signer не должен пересекаться между split).

## Референсы

- BIO segmentation paper: [https://aclanthology.org/2023.findings-emnlp.846.pdf](https://aclanthology.org/2023.findings-emnlp.846.pdf)
- segmentation repo: [https://github.com/sign-language-processing/segmentation](https://github.com/sign-language-processing/segmentation)
- pose-format: [https://github.com/sign-language-processing/pose](https://github.com/sign-language-processing/pose)

