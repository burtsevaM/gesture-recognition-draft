# BIO Live Calibration

## Что делает калибратор

`backend/tools/calibrate_bio_thresholds_live.py` подбирает пороги для BIO-сегментации на **живом источнике** (камера / видео / websocket-кадры):

- `th_B`, `th_O`
- `min_len`, `merge_gap`

Метрики для сравнения конфигураций:

- `segments/min` — сколько сегментов в минуту;
- `fp/min` — ложные сегменты в минуту (`--quiet-mode` = все сегменты считаются FP);
- `avg_len(fr)`, `avg_len(s)` — средняя длина сегмента;
- `jitter` — дребезг границ между соседними итерациями декодера;
- `stability` — стабильность (`1 / (1 + jitter)`).

Калибратор сохраняет лучший набор в `backend/artifacts/bio_thresholds.json` и делает бэкап предыдущего файла.

## Запуск

Из корня репозитория:

```bash
cd "/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2"
```

### 1) Камера (рекомендуется для практической доводки)

```bash
./.venv/bin/python backend/tools/calibrate_bio_thresholds_live.py \
  --source camera \
  --camera-index 0 \
  --max-frames 1200 \
  --quiet-mode
```

### 2) Локальный видеофайл

```bash
./.venv/bin/python backend/tools/calibrate_bio_thresholds_live.py \
  --source video \
  --video-path /absolute/path/to/video.mp4 \
  --max-frames 1800
```

### 3) WebSocket source (если поток отдает JPEG-кадры)

```bash
./.venv/bin/python backend/tools/calibrate_bio_thresholds_live.py \
  --source ws \
  --ws-url ws://127.0.0.1:9000/frames \
  --max-frames 1200
```

Ожидается, что WS присылает либо бинарный JPEG, либо JSON с `frame_jpeg_b64`/`jpeg_b64`.

## Полезные параметры

- `--grid-th-b`, `--grid-th-o` — сетка порогов.
- `--grid-min-len`, `--grid-merge-gap` — сетка постобработки.
- `--target-segments-per-min` — “комфортная” плотность сегментов для ранжирования.
- `--motion-threshold` — порог для эвристики FP в не-quiet режиме.
- `--window`, `--step` — окно/шаг BIO ONNX.

Пример расширенной калибровки:

```bash
./.venv/bin/python backend/tools/calibrate_bio_thresholds_live.py \
  --source camera \
  --max-frames 1600 \
  --grid-th-b "0.35,0.45,0.55,0.65" \
  --grid-th-o "0.35,0.45,0.55,0.65" \
  --grid-min-len "4,6,8,10" \
  --grid-merge-gap "0,1,2,3" \
  --target-segments-per-min 8 \
  --top-k 12
```

## Как интерпретировать метрики

- Если `fp/min` высокий: поднимать `th_B`, `th_O`, `min_len`, `cool_off_frames`.
- Если система “пропускает” жесты: снижать `th_B` или `min_len`, уменьшать `merge_gap`.
- Если много дробления одного жеста: увеличивать `merge_gap`, `min_len`, иногда `th_O`.
- Если `jitter` высокий: поднимать `th_B`, `min_len`, шагать в сторону более “консервативных” порогов.

## Нормальные ориентиры для демо

Для спокойного демо-потока (1 человек, статичный фон):

- `fp/min <= 0.5` (лучше `<= 0.2` в режиме тишины),
- `segments/min` обычно в диапазоне `4..12` (зависит от темпа),
- `avg_len(s)` около `0.2..1.2`,
- `stability >= 0.7` (чем ближе к 1 — тем лучше).

Эти значения не универсальны; итог подбирается под конкретную камеру/свет/дистанцию.
