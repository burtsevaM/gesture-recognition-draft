# Текущее состояние проекта

## 1. Назначение документа

Этот документ фиксирует текущее состояние репозитория на момент анализа кода и структуры файлов.

Документ описывает:
- текущее устройство репозитория;
- реально существующие runtime-пайплайны и утилиты;
- найденные артефакты, конфиги, точки входа и тесты;
- подтвержденные по коду ограничения и риски.

Документ **не** является описанием целевой архитектуры и **не** описывает желаемое будущее состояние проекта.

## 2. Краткое резюме текущего состояния

- Репозиторий представляет собой локальный MVP для распознавания РЖЯ, в котором одновременно сосуществуют несколько разных пайплайнов: `letters`, `words` и `pose_words`.
- Основной runtime реализован как FastAPI-сервис с WebSocket-стримингом JPEG-кадров из браузера.
- В коде подтверждены три отдельных режима распознавания: буквенный retrieval-пайплайн, RGB word-classifier и pose-first пайплайн со скелетом и BIO-сегментацией.
- В проекте уже есть фронтенд для локального демо, healthcheck, gallery inspector, debug-панели, оверлей скелета и базовый offline/client-side scaffold.
- Для `letters` реализован законченный локальный контур: сбор эталонов, построение FAISS-индекса, retrieval, hold/cooldown, optional VLM-judge.
- Для `words` реализован отдельный ONNX inference runtime со sliding window, декодером состояний и runtime-логированием.
- Для `pose_words` реализован отдельный pose-first runtime: MediaPipe Holistic, нормализация позы, feature composition, BIO-сегментация, pose-word classifier, skeleton payload, perf/debug payload.
- В проекте присутствует крупная training/data-подсистема: split’ы Slovo, подготовка pose dataset, генерация synthetic BIO dataset, обучение и экспорт моделей. Часть этой подсистемы выглядит рабочей, часть — как scaffold или исследовательский код.
- Текущая конфигурация `backend/config.yaml` выставлена в `recognition_mode: pose_words` и `segmentation.enabled: true`.
- Основной риск текущего состояния: репозиторий смешивает продуктовый runtime, датасетные утилиты, тренировочные скрипты, экспериментальные ветки архитектуры и локальные артефакты в одном рабочем дереве.

## 3. Структура репозитория

### Корень репозитория

Основные директории и файлы:
- `README.md` — верхнеуровневое описание проекта, режимов и команд запуска.
- `docs/` — общие документы в корне. На момент анализа здесь находятся `design.md`, а также документы фиксации текущего состояния и целевой архитектуры.
- `frontend/` — браузерный интерфейс, работающий поверх backend WebSocket.
- `backend/` — вся серверная, ML runtime, training/data/scripts/tests.
- `agents.md` — служебный файл для агентной работы, не относится к runtime.

### `backend/`

Ключевая директория проекта. Внутри нее собраны runtime, конфигурация, артефакты, данные, скрипты и тесты.

Основные поддиректории:
- `backend/app/` — основной runtime-код FastAPI и ML пайплайнов.
- `backend/artifacts/` — рабочие артефакты инференса: индексы, ONNX, labels, config/threshold JSON, runtime-логи.
- `backend/data/` — локальные датасеты и внешние исходники (`slovo.zip`, `slovo_repo`).
- `backend/docs/` — локальные runbook/планы/документы по подпроцессам.
- `backend/gallery/` — галерея эталонов для буквенного режима.
- `backend/scripts/` — датасетные и инфраструктурные скрипты.
- `backend/tools/` — пользовательские утилиты для галереи, индекса и оценки.
- `backend/train/` — тренировочные и export-скрипты для моделей.
- `backend/tests/` — unit/smoke/regression tests.
- `backend/config.yaml` — главный конфиг runtime.
- `backend/requirements.txt` — зависимости.
- `backend/README.md` — runbook по `pose_words + segmentation`.

### `backend/app/`

Основные файлы runtime:
- `backend/app/main.py` — единая точка входа FastAPI, WebSocket, маршрутизация режимов, healthcheck, gallery API, orchestration всех пайплайнов.
- `backend/app/config.py` — dataclass-конфиг, загрузка `yaml`, merge nested sections.
- `backend/app/schemas.py` — helper для сборки JSON payload inference-сообщения.
- `backend/app/hand_detector.py` — MediaPipe Hand Landmarker wrapper и crop для буквенного пайплайна.
- `backend/app/embedding.py` — DINOv2 embedding extractor.
- `backend/app/retrieval.py` — FAISS-индекс и поиск по галерее.
- `backend/app/state_machine.py` — hold-to-commit state machine для буквенного режима.
- `backend/app/vlm_judge.py` — optional VLM judge через LM Studio/OpenAI-compatible API.
- `backend/app/words/` — отдельный runtime для RGB `words` mode.
- `backend/app/pose/` — pose extraction, datatypes, normalization, feature composition, worker.
- `backend/app/segmentation/` — BIO model wrapper, decoder, streaming segmenter, метрики.
- `backend/app/pose_words/` — pose-word ONNX wrapper и утилиты сегментов.
- `backend/app/perf/` — профилирование и агрегация runtime-показателей.

### `backend/artifacts/`

По рабочему дереву найдены:
- `faiss.index`, `meta.json` — артефакты буквенного retrieval.
- `models/hand_landmarker.task` — модель MediaPipe Hand Landmarker.
- `slovo_word_model.onnx` — RGB word model.
- `slovo_word_model_mvit16-4.onnx` — еще одна RGB word model; происхождение по коду полностью не подтверждено.
- `labels.txt` — labels для `words` mode.
- `pose_word_model.onnx`, `pose_word_labels.txt`, `pose_word_config.json` — артефакты `pose_words`.
- `bio_segmenter.onnx`, `bio_thresholds.json`, `bio_config.json` — артефакты сегментации.
- `words_runtime.jsonl` — runtime-лог.

### `backend/data/`

По рабочему дереву найдены:
- `backend/data/slovo/slovo.zip` — локальный архив датасета Slovo.
- `backend/data/slovo_repo/` — отдельная вложенная копия внешнего репозитория Slovo с собственным `.git`.

### `backend/gallery/`

Содержит структуру эталонов для букв:
- директории по меткам (`А`, `B`, `Г`, `И`, `М` и т.д.);
- внутри — session-based подкаталоги (`A_mirror_01`, `G_protocol_v1`, `M_manual_single` и т.п.).

### `frontend/`

Ключевые файлы:
- `frontend/index.html` — UI приложения.
- `frontend/app.js` — логика камеры, WebSocket, overlay, debug, sentence splitting.
- `frontend/style.css` — стили.
- `frontend/offline/` — experimental scaffold для browser/client-side inference.
- `frontend/README_offline.md` — пояснение по experimental browser mode.

## 4. Текущие режимы и сценарии распознавания

В коде подтверждены три runtime-режима, выбираемые через `recognition_mode` в `backend/config.yaml` и `AppConfig`.

### 4.1. Режим `letters`

**Где найден в коде**
- `backend/app/config.py`
- `backend/app/main.py`
- `backend/app/hand_detector.py`
- `backend/app/embedding.py`
- `backend/app/retrieval.py`
- `backend/app/state_machine.py`
- `backend/app/vlm_judge.py`

**Назначение**
- Локальное распознавание статических букв дактиля по crop руки и эталонной галерее.

**Входные данные**
- JPEG-кадры из браузера через WebSocket.
- Галерея эталонов в `backend/gallery/`.
- FAISS-индекс и metadata в `backend/artifacts/`.

**Основные этапы обработки**
- Детекция руки через MediaPipe Hand Landmarker.
- Вырезание crop руки.
- Извлечение DINOv2 embedding.
- Поиск ближайших эталонов через FAISS cosine/IP.
- Применение порогов `sim_none`, `sim_vlm_th`, `margin_th`.
- Hold/cooldown логика.
- Опционально VLM judge для спорных случаев.

**Какие модели/файлы/конфиги использует**
- `backend/artifacts/models/hand_landmarker.task`
- DINOv2 из Hugging Face (`facebook/dinov2-small`) загружается в runtime.
- `backend/artifacts/faiss.index`
- `backend/artifacts/meta.json`
- `backend/config.yaml`
- Опционально LM Studio/OpenAI-compatible VLM endpoint.

**Какой результат выдает**
- JSON payload со статусом, текущей буквой, top-k, hold-прогрессом, VLM verdict, debug similarity.

**Текущий статус**
- Реализован и выглядит рабочим по коду.
- Качество распознавания и устойчивость на конкретном датасете требуют ручной проверки.

**Замечания и ограничения**
- Пайплайн опирается на локальную галерею и качество ее наполнения.
- Это retrieval-подход, а не полноценный обученный буквенный классификатор.
- Режим сильно завязан на локальные эталоны и FAISS-индекс.

### 4.2. Режим `words`

**Где найден в коде**
- `backend/app/config.py`
- `backend/app/main.py`
- `backend/app/words/service.py`
- `backend/app/words/model_onnx.py`
- `backend/app/words/decoder.py`
- `backend/app/words/buffer.py`
- `backend/app/words/metrics.py`

**Назначение**
- Распознавание isolated words по RGB-видеоклипу через sliding-window ONNX classifier.

**Входные данные**
- Последовательность JPEG-кадров из браузера.
- ONNX модель слова и labels.

**Основные этапы обработки**
- Буферизация кадров в ring buffer.
- Отбор clip окна `window_frames` с шагом `frame_interval`.
- ONNX inference по RGB clip.
- EMA сглаживание вероятностей.
- Threshold-based decoding: `NONE | UNKNOWN | HOLD | COMMIT | COOLDOWN`.
- Runtime log и latency/FP metrics.

**Какие модели/файлы/конфиги использует**
- `backend/artifacts/slovo_word_model.onnx` или другой путь из конфига.
- Текущий конфиг указывает на `backend/artifacts/slovo_word_model_mvit16-4.onnx`.
- `backend/artifacts/labels.txt`.
- `backend/config.yaml`.
- Опционально hand presence gate через `HandDetector`.

**Какой результат выдает**
- JSON payload со статусом, top1/topk, hold/cooldown, latency и runtime debug.

**Текущий статус**
- Реализован и выглядит рабочим по коду при наличии ONNX и labels.
- Тренировочная часть внутри репозитория для RGB words выглядит неполной: `backend/train/train_word_model.py` и `backend/train/export_onnx.py` являются scaffold/stub.

**Замечания и ограничения**
- В репозитории присутствует runtime для `words`, но не подтвержден полный training/export pipeline для актуальной RGB модели.
- В `backend/artifacts` лежат два больших ONNX-файла слова; происхождение версии `slovo_word_model_mvit16-4.onnx` не удалось полностью подтвердить по коду.
- Контур работает как отдельная RGB-ветка и не использует pose extraction.

### 4.3. Режим `pose_words`

**Где найден в коде**
- `backend/app/config.py`
- `backend/app/main.py`
- `backend/app/pose/`
- `backend/app/segmentation/`
- `backend/app/pose_words/`
- `backend/app/perf/`

**Назначение**
- Pose-first пайплайн для распознавания слов: извлечение скелета, нормализация, feature composition, сегментация, классификация слова.

**Входные данные**
- JPEG-кадры из браузера.
- MediaPipe Holistic pose extraction.
- ONNX для BIO segmenter и pose-word classifier.

**Основные этапы обработки**
- Получение body/hand landmarks через `PoseExtractor`.
- Опциональная нормализация плеч и 3D-нормализация кистей.
- `compose_features` -> вектор признаков по кадру.
- Буферизация и streaming BIO segmentation.
- Извлечение завершенных sign segments.
- Ресэмплинг сегмента до фиксированного `T`.
- ONNX inference pose-word classifier.
- Декодирование через `WordDecisionDecoder`.
- Возврат `skeleton.raw`, `skeleton.norm`, `segments`, `bio`, `perf`.

**Какие модели/файлы/конфиги использует**
- `backend/artifacts/pose_word_model.onnx`
- `backend/artifacts/pose_word_labels.txt`
- `backend/artifacts/pose_word_config.json`
- `backend/artifacts/bio_segmenter.onnx`
- `backend/artifacts/bio_thresholds.json`
- `backend/artifacts/bio_config.json`
- `backend/config.yaml`

**Какой результат выдает**
- JSON payload с обычным inference-сообщением плюс optional поля:
  - `skeleton`
  - `segments`
  - `bio`
  - `perf`
  - `segment_event`

**Текущий статус**
- Runtime-пайплайн реализован и health/readiness в коде предусмотрены.
- Текущие локальные артефакты `pose_word_model.onnx` и `bio_segmenter.onnx` в рабочем дереве сгенерированы dummy/bootstrap-скриптами, что подтверждается полем `generated_by` в JSON-конфигах.
- Поэтому режим можно считать **экспериментальным**: pipeline и интеграция есть, но реальное качество pose-распознавания слов текущими локальными артефактами не подтверждено.

**Замечания и ограничения**
- При `segmentation.enabled: false` режим фактически работает как pose viewer / feature extraction path без полноценного word recognition.
- При `segmentation.enabled: true` readiness зависит от полного набора pose/BIO артефактов.
- Наличие runtime не означает наличие обученной целевой модели.

### 4.4. Дополнительный сценарий `inference_location=browser`

**Где найден в коде**
- `frontend/index.html`
- `frontend/app.js`
- `frontend/offline/backend_browser_ort.js`

**Назначение**
- Experimental scaffold под будущий client-side/browser inference.

**Статус**
- Не является отдельным backend `recognition_mode`.
- По фронтенд-коду это scaffold загрузки ORT Web и моделей с fallback на backend WS.
- Реальный browser runtime inference по коду не доведен: `predictBio()` и `predictWord()` в `backend_browser_ort.js` пока выбрасывают ошибки-заглушки.

## 5. Текущий пайплайн обработки данных

### 5.1. Общий транспортный контур

Для всех режимов подтвержден общий поток:
- браузер получает видеопоток через `getUserMedia`;
- фронтенд рисует кадр на canvas;
- кадр кодируется в JPEG;
- кадр отправляется binary-пакетом в WebSocket `/ws/stream`;
- backend декодирует JPEG в `frame_bgr`;
- дальше `SessionProcessor` маршрутизирует кадр в выбранный runtime pipeline.

Также по WebSocket поддержан text-control пакет:
- `{"type":"control","action":"clear_text"}`.

### 5.2. Пайплайн `letters`

- `frame_bgr` -> `HandDetector.detect()`.
- Если руки нет или bbox слишком мал, возвращается `NONE`.
- Если рука есть, crop -> `DinoEmbedder.embed_rgb()`.
- embedding -> `GalleryIndex.search()`.
- top1/top2 similarity -> threshold logic.
- `HoldToCommitStateMachine` формирует `CANDIDATE/COMMITTED/COOLDOWN/NONE`.
- Optional VLM judge может вмешаться на precommit/uncertain.
- Payload возвращается сразу в WebSocket.

### 5.3. Пайплайн `words`

- `frame_bgr` -> append в `FrameRingBuffer`.
- По достижении достаточного окна кадров сервис делает `sample_clip()`.
- `WordOnnxModel` нормализует RGB clip и запускает ONNXRuntime.
- `WordDecisionDecoder` сглаживает вероятности и применяет gating.
- `WordRuntimeMetrics` считает latency/FP.
- Payload + runtime log отправляются наружу.

### 5.4. Пайплайн `pose_words`

- `frame_bgr` либо сразу обрабатывается, либо уходит в `PosePipelineWorker`.
- В worker или синхронно:
  - BGR -> RGB;
  - MediaPipe Holistic -> `PoseFrame`;
  - нормализация плеч/кистей;
  - `compose_features()` -> feature vector.
- Далее два варианта:
  - если `segmentation.enabled = false`, runtime возвращает `POSE` payload со скелетом без word classification;
  - если `segmentation.enabled = true`, feature vector идет в `StreamingBioSegmenter`.
- `StreamingBioSegmenter` поддерживает ring buffer, окно, step, aggregation и возвращает completed sign segments.
- По завершенному сегменту вызывается `get_feature_span()` -> `resample_to_fixed_T()` -> `PoseWordOnnxModel.infer_probs()`.
- `WordDecisionDecoder` применяет hold/cooldown уже по сегментам.
- Payload дополняется скелетом, сегментами, perf-метриками и BIO debug.

## 6. Модели, веса, ONNX и артефакты

### 6.1. Что найдено в рабочем дереве

Найдены следующие runtime-артефакты:

**Для `letters`**
- `backend/artifacts/models/hand_landmarker.task`
- `backend/artifacts/faiss.index`
- `backend/artifacts/meta.json`
- локальная галерея `backend/gallery/`

**Для `words`**
- `backend/artifacts/slovo_word_model.onnx`
- `backend/artifacts/slovo_word_model_mvit16-4.onnx`
- `backend/artifacts/labels.txt`
- `backend/artifacts/words_runtime.jsonl`

**Для `pose_words`**
- `backend/artifacts/pose_word_model.onnx`
- `backend/artifacts/pose_word_labels.txt`
- `backend/artifacts/pose_word_config.json`
- `backend/artifacts/bio_segmenter.onnx`
- `backend/artifacts/bio_thresholds.json`
- `backend/artifacts/bio_config.json`

### 6.2. Обязательные зависимости между кодом и артефактами

- `letters` зависит от построенного FAISS-индекса и hand landmarker task file.
- `words` зависит от ONNX word model + labels.
- `pose_words` при включенной сегментации зависит от полного набора pose/BIO артефактов.
- `DinoEmbedder` требует `torch + transformers`, но модель хранится не в репозитории, а загружается через Hugging Face runtime.
- Optional VLM judge требует внешний сервис LM Studio/OpenAI-compatible endpoint.

### 6.3. Что важно зафиксировать отдельно

- `backend/artifacts/pose_word_config.json` содержит `generated_by: backend/train/make_dummy_pose_word_model.py`.
- `backend/artifacts/bio_config.json` и `backend/artifacts/bio_thresholds.json` содержат `generated_by: backend/train/make_dummy_bio_segmenter.py`.
- Следовательно, текущие pose/BIO артефакты в рабочем дереве — bootstrap baseline, а не подтвержденные production-модели.
- Для `words` runtime в конфиге используется `slovo_word_model_mvit16-4.onnx`, но скрипт `backend/scripts/download_slovo_assets.sh` скачивает `slovo_word_model.onnx` (`mvit32-2.onnx`). Связь между этими двумя файлами требует ручной проверки.

### 6.4. Чего может не хватать на чистой машине

На чистой машине могут отсутствовать:
- `faiss.index` и `meta.json` — если индекс еще не собран;
- `hand_landmarker.task` — хотя `HandDetector` умеет скачать его автоматически;
- `slovo_word_model*.onnx` и `labels.txt` — если не скачаны/не экспортированы;
- pose/BIO ONNX и JSON — если не выполнен bootstrap или export;
- локальный архив Slovo и вложенный `slovo_repo`.

## 7. API / сервисная часть / интеграция

### 7.1. Что найдено

- Backend реализован на FastAPI.
- Есть WebSocket API и несколько HTTP endpoints.
- Frontend отдается этим же backend-сервисом.

### 7.2. Точки входа

Подтвержденные маршруты:
- `GET /` — отдает `frontend/index.html`.
- `GET /health` — healthcheck и readiness summary.
- `GET /api/gallery` — JSON-список изображений галереи.
- `GET /gallery` — gallery inspector.
- `WS /ws/stream` — основной streaming inference контракт.
- Static mounts:
  - `/static`
  - `/gallery_files`

### 7.3. Контракт WebSocket

**Вход**
- binary JPEG кадр;
- text JSON control packet (`clear_text`).

**Выход**
- базовое inference-сообщение строится через `build_inference_message()`;
- обязательный базовый каркас payload общий для всех режимов;
- для `pose_words` добавляются optional поля `skeleton`, `segments`, `perf`, `bio`, `segment_event`.

Контракт не версионируется и не описан отдельной схемой/версией API; он собирается вручную через dict helper.

### 7.4. Связь API и ML

- `RuntimeContext` выступает как service-locator для runtime-объектов и readiness.
- `SessionProcessor` связывает WebSocket-сессию с выбранным ML пайплайном.
- В `backend/app/main.py` одновременно находятся:
  - FastAPI routes;
  - runtime initialization;
  - orchestration режимов;
  - payload assembly;
  - часть debug/perf логики.

Явное разделение между transport/API слоем и ML-оркестрацией присутствует частично, но неполное.

## 8. Конфигурация и запуск

### 8.1. Основной конфиг

Главный runtime-конфиг — `backend/config.yaml`.

Через него управляются:
- `recognition_mode`;
- `use_shoulder_norm`, `use_hands_3d_norm`;
- `perf` и `pose_worker`;
- retrieval thresholds для `letters`;
- пути и параметры `word_model`;
- thresholds/smoothing/commit logic для `words`;
- `segmentation` и `pose_word_model` для `pose_words`.

### 8.2. Выбор режима

Режим выбирается значением `recognition_mode`:
- `letters`
- `words`
- `pose_words`

Нормализация в `AppConfig.from_dict()` приводит неизвестное значение обратно к `letters`.

### 8.3. Что критично для запуска

**Минимально критично по режимам:**

Для `letters`:
- `backend/config.yaml`
- `backend/artifacts/models/hand_landmarker.task`
- `backend/artifacts/faiss.index`
- `backend/artifacts/meta.json`
- наполненная `backend/gallery/`

Для `words`:
- `backend/config.yaml`
- корректный путь `word_model.path`
- `labels.txt`

Для `pose_words`:
- `backend/config.yaml`
- `PoseExtractor` (MediaPipe)
- при `segmentation.enabled: true` весь набор pose/BIO артефактов

### 8.4. Команды и скрипты запуска

Подтверждены команды и скрипты:
- `uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- `python backend/tools/capture_gallery.py ...`
- `python backend/tools/build_index.py ...`
- `python backend/tools/calibrate_thresholds.py ...`
- `python -m backend.scripts.bootstrap_pose_words_artifacts`
- `python backend/scripts/smoke_pose_words.py ...`
- `pytest backend/tests`

### 8.5. Особенно критичные места конфига

- `recognition_mode`
- `word_model.path` и `word_model.labels_path`
- `segmentation.enabled` и соответствующие пути
- `pose_word_model.path` / `labels_path` / `config_path`
- `hand_landmarker_model_path`
- `sim_none`, `sim_vlm_th`, `margin_th`

## 9. Тесты и проверяемость

### 9.1. Что найдено

Тесты есть и покрывают значительную часть инфраструктуры.

Подтвержденные группы тестов:
- retrieval helpers (`test_retrieval.py`)
- буквенный state machine (`test_state_machine.py`)
- config parsing (`test_config.py`)
- `words` buffer/decoder (`test_words_buffer.py`, `test_words_decoder.py`)
- WS contract (`test_ws_contract.py`)
- pose normalization (`test_shoulder_normalization.py`, `test_hand_normalize_3d.py`)
- pose extractor shape/input validation (`test_pose_extractor_shapes.py`)
- segment resampling and ring indexing (`test_segment_resample.py`, `test_ring_indexing.py`)
- BIO decoder and streaming segmenter (`test_bio_decoder.py`, `test_streaming_segmenter.py`)
- bootstrap/readiness (`test_artifacts_bootstrap.py`)
- pose_words payload and smoke integration (`test_pose_words_payload.py`, `test_pose_words_integration_smoke.py`)
- perf payload optionality (`test_perf_payload_optional.py`)
- synthetic BIO dataset builder (`test_bio_dataset_builder.py`)

### 9.2. Что тестами покрыто хорошо

- математика нормализации;
- формат payload;
- локальная логика декодеров;
- разбор конфигов;
- readiness при отсутствии/наличии артефактов;
- streaming segmenter и segment utilities.

### 9.3. Что покрыто слабо или не подтверждено

- Нет полноценного автоматического end-to-end теста с реальной камерой и реальными production-моделями.
- Не подтверждена точность распознавания по тестам.
- Нет CI-подтверждения, что все локальные артефакты воспроизводимы из нуля без ручных шагов кроме bootstrap dummy.
- Для `words` training/export pipeline в репозитории не выглядит завершенным.
- Для `pose_words` текущий smoke опирается на synthetic/dummy сценарии, а не на обученную модель.

### 9.4. Быстрая ручная проверка

По коду и документации для быстрой проверки доступны:
- `GET /health`
- открытие UI на `http://127.0.0.1:8000`
- `backend/scripts/smoke_pose_words.py`
- `backend/tools/eval_realtime_log.py`
- `backend/tools/eval_sanity_split.py`

## 10. Проблемы и технический долг

- `backend/app/main.py` одновременно содержит API, orchestration режимов, payload assembly, readiness, часть perf/debug и часть runtime-логики. Это делает файл перегруженным.
- В одном репозитории смешаны:
  - продуктовый runtime;
  - сбор датасета;
  - подготовка Slovo;
  - обучение;
  - экспорт;
  - experimental browser scaffold;
  - runtime-логи и локальные артефакты.
- В проекте одновременно существуют несколько параллельных путей распознавания слов (`words` и `pose_words`) с частичным пересечением ответственности.
- Верхний `README.md`, `backend/README.md` и текущий конфиг уже фокусируются на разных центрах тяжести; единая “истина” по текущему основному сценарию отсутствует.
- Для `pose_words` локальные артефакты являются dummy baseline. Это означает, что наличие working runtime не равно наличию реального рабочего pose-word recognition.
- RGB `words` runtime выглядит более “настоящим” по runtime-части, чем по training/export части: inference реализован, а training/export внутри репозитория остаются неполными.
- Внутри `backend/data/slovo_repo/` хранится вложенный внешний git-репозиторий. Это осложняет переносимость и границы ответственности.
- В репозитории уже находятся runtime-логи (`backend/artifacts/words_runtime.jsonl`) и локальные рабочие артефакты, что плохо для чистого основного репозитория.
- WebSocket контракт не версионируется и не вынесен в отдельный формализованный слой схем/контрактов.
- Статус `browser` inference по фронтенду присутствует в UI, но является scaffold и может создавать ложное ощущение готовой offline-функциональности.

## 11. Что точно нельзя потерять при переносе

Критично сохранить:
- `backend/app/main.py` как текущую фактическую реализацию маршрутизации режимов и WS-потока.
- `backend/app/config.py` и структуру `backend/config.yaml`.
- Модули `backend/app/hand_detector.py`, `backend/app/embedding.py`, `backend/app/retrieval.py`, `backend/app/state_machine.py`.
- Весь пакет `backend/app/words/`.
- Весь пакет `backend/app/pose/`.
- Весь пакет `backend/app/segmentation/`.
- Весь пакет `backend/app/pose_words/`.
- `backend/app/schemas.py` и фактический payload contract.
- `frontend/index.html`, `frontend/app.js`, `frontend/style.css`.
- `backend/tools/capture_gallery.py`, `backend/tools/build_index.py`, `backend/tools/calibrate_thresholds.py`.
- `backend/scripts/prepare_slovo_splits.py`, `backend/scripts/slovo_to_pose_dataset.py`, `backend/scripts/build_bio_continuous_dataset.py`.
- `backend/scripts/bootstrap_pose_words_artifacts.py` как механизм восстановления pose_words runtime без падения.
- Тесты, которые фиксируют текущий контракт и математику: `test_ws_contract.py`, `test_shoulder_normalization.py`, `test_hand_normalize_3d.py`, `test_streaming_segmenter.py`, `test_artifacts_bootstrap.py`, `test_pose_words_integration_smoke.py`.

Для воспроизводимости также критичны:
- структура `backend/artifacts/` и ожидаемые имена файлов;
- структура `backend/gallery/` для букв;
- структура labels/threshold/config JSON рядом с моделями;
- команды запуска из `README.md` и `backend/README.md`.

## 12. Итог

Проект находится в состоянии функционального, но архитектурно смешанного MVP.

Что можно считать базой:
- FastAPI + WebSocket transport;
- буквенный retrieval pipeline;
- RGB words runtime;
- pose-first runtime с extraction/normalization/segmentation scaffold;
- фронтенд для локальной проверки;
- заметное покрытие unit/smoke тестами.

Что нельзя считать подтвержденным без дополнительной проверки:
- качество `words` и `pose_words` на целевых данных;
- происхождение и статус всех крупных ONNX-артефактов;
- готовность `pose_words` как продуктового режима при текущих dummy pose/BIO моделях.

Что уже очевидно требует реорганизации:
- разделение runtime и research/training зон;
- отделение контракта API от оркестрации и ML-логики;
- явное разведение продуктовых и экспериментальных режимов;
- очистка основного репозитория от локальных данных, nested repo и bootstrap/draft артефактов.
