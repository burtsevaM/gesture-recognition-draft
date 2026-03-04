# Pose-First Plan: режим `pose_words`

## 1. Что добавляем

Добавляем новый режим распознавания: `pose_words` (pose-first), не ломая текущие режимы:

- `letters` (DINOv2 + FAISS retrieval) — без изменений,
- `words` (RGB sliding-window ONNX) — без изменений,
- `pose_words` (новый): извлечение позы на каждом кадре через MediaPipe Holistic и нормализация скелета.

Цель MVP в этом этапе: **не обучение модели**, а подготовка pose-пайплайна и визуализации:

1. Извлечение body+hands landmarks из кадра (face выключено по умолчанию).
2. Нормализация позы:
   - центр по midpoint плеч,
   - масштаб по расстоянию между плечами,
   - опционально отдельная 3D-нормализация кистей.
3. Расширение WS-ответа опциональным полем `skeleton` (raw + norm).
4. Рендер скелета на фронте (canvas overlay) с переключением `raw/norm`.

---

## 2. Текущее состояние контракта (до изменений)

### 2.1. WebSocket вход

Точка: `backend/app/main.py`, endpoint `/ws/stream`.

- Бинарный кадр JPEG (`bytes`) от фронта.
- Текстовый control-пакет (`{"type":"control","action":"clear_text"}`) -> `{"type":"ack","action":"clear_text"}`.

### 2.2. WebSocket выход

Сейчас backend отправляет payload из `build_inference_message(...)` (`backend/app/schemas.py`), ключевые поля:

- `mode`, `status`, `letter`, `word`,
- `score`, `confidence`,
- `hand_present`, `bbox_norm`,
- `hold`, `text_state`,
- `topk`, `vlm`, `debug`.

Фронт (`frontend/app.js`) уже использует этот контракт и рисует bbox на `#overlay` canvas.

---

## 3. Поток данных для `pose_words`

Планируемый поток (MVP):

1. **JPEG frame** приходит по WS (`/ws/stream`).
2. Backend декодирует JPEG в `frame_bgr` (как и сейчас).
3. В режиме `pose_words` кадр передается в **MediaPipe Holistic**:
   - pose landmarks (body),
   - left/right hand landmarks,
   - face landmarks по умолчанию отключены.
4. Формируем два представления:
   - `raw`: координаты landmarks в исходной системе (нормированные координаты MediaPipe),
   - `norm`: координаты после нормализации:
     - центрирование по midpoint плеч,
     - масштабирование по shoulder distance (=1),
     - опционально hand 3D-нормализация.
5. Возвращаем стандартный WS payload + **опционально** `skeleton`.
6. Фронт, если `skeleton` есть, рисует скелет поверх видео.

---

## 4. Новые поля WS payload (строго опциональные)

Чтобы не ломать текущий контракт, добавляем только необязательный блок `skeleton`.

### 4.1. Предлагаемая структура

```json
{
  "...": "existing fields",
  "skeleton": {
    "version": 1,
    "source": "mediapipe_holistic",
    "raw": {
      "pose": [[x,y,z,vis], ...],
      "left_hand": [[x,y,z,vis], ...],
      "right_hand": [[x,y,z,vis], ...]
    },
    "norm": {
      "pose": [[x,y,z,vis], ...],
      "left_hand": [[x,y,z,vis], ...],
      "right_hand": [[x,y,z,vis], ...],
      "meta": {
        "center": [cx, cy],
        "shoulder_dist": d,
        "hand_3d_norm": false
      }
    },
    "edges": {
      "pose": [[i,j], ...],
      "left_hand": [[i,j], ...],
      "right_hand": [[i,j], ...]
    }
  }
}
```

### 4.2. Принцип совместимости

- Если `skeleton` отсутствует, фронт работает как сейчас.
- Все текущие поля ответа сохраняются без переименования.
- В `letters`/`words` можно не отправлять `skeleton` (или отправлять только при включенном флаге).

---

## 5. Флаги, которые добавим в `backend/config.yaml`

План расширения конфига (минимально достаточный):

```yaml
recognition_mode: letters | words | pose_words

pose_words:
  enabled: true
  include_skeleton_in_ws: true
  include_raw: true
  include_norm: true
  include_face: false

  holistic:
    static_image_mode: false
    model_complexity: 1
    smooth_landmarks: true
    min_detection_confidence: 0.5
    min_tracking_confidence: 0.5

  normalize:
    enabled: true
    center_by_shoulders: true
    scale_by_shoulder_distance: true
    shoulder_eps: 1e-6
    hand_3d_enabled: false
    hand_3d_wrist_index: 0
    hand_3d_scale_eps: 1e-6

  performance:
    process_every_n_frame: 1
```

Плюс фронтовые настройки (через `/health` и UI-переключатели):

```yaml
frontend:
  skeleton_overlay_default: true
  skeleton_space_default: raw   # raw | norm
```

---

## 6. Что будем менять в коде (план файлов)

### 6.1. Создать

- `backend/app/pose_words/__init__.py`
- `backend/app/pose_words/holistic_extractor.py` — вызов MediaPipe Holistic и сбор landmarks.
- `backend/app/pose_words/normalize.py` — shoulder-нормализация + optional hand 3D-нормализация.
- `backend/app/pose_words/service.py` — сервис режима `pose_words`, формирование `skeleton` блока.
- `backend/tests/test_pose_normalize.py` — `shoulder_normalize`, `hand_normalize_3d`.
- `backend/tests/test_pose_payload.py` — smoke на опциональный `skeleton` в payload.

### 6.2. Изменить

- `backend/app/config.py` — новые поля конфига и парсинг.
- `backend/config.yaml` — добавление секции `pose_words`.
- `backend/app/schemas.py` — опциональный `skeleton` в `build_inference_message`.
- `backend/app/main.py` — ветка `if recognition_mode == "pose_words"`.
- `frontend/index.html` — UI-переключатели overlay/raw-norm.
- `frontend/app.js` — рендер скелета на canvas overlay.
- `frontend/style.css` — стили для переключателей.

---

## 7. Ограничения MVP (осознанно)

- На этом шаге нет обучения/классификатора по позе.
- `pose_words` в MVP — это extraction + normalization + визуализация и доставка позы в WS.
- Классификацию слов по позе можно добавить следующим этапом (например, sequence model по landmarks).

---

## 8. Критерии готовности этапа

1. В конфиге доступен режим `pose_words`.
2. WS вход не изменился (JPEG bytes), выход совместим с текущим фронтом.
3. В WS ответе появляется опциональный `skeleton` (raw+norm).
4. На фронте отображается скелет с переключением `raw/norm`.
5. Тесты:
   - shoulder normalization (центр ~0, плечи ~1),
   - hand 3D normalization (инвариантность),
   - smoke payload (`skeleton` optional, без поломки старого формата).

