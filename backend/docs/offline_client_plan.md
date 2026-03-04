# Offline / Client-Side Plan (Scaffolding)

## Цель текущего этапа
Сделать безопасный задел под client-side инференс без миграции действующего production-пайплайна.

Сейчас рабочий путь остается прежним:
- камера -> JPEG -> WebSocket -> backend inference (`letters` / `words` / `pose_words`) -> UI.

Новый код в этом этапе:
- добавляет UI-переключатель `inference_location` (`backend` | `browser`);
- добавляет frontend-структуру offline backend-адаптеров;
- добавляет backend-скрипт экспорта ONNX-артефактов для фронта;
- не меняет WS-контракт и не ломает текущий runtime.

## Вариант A: onnxruntime-web (WASM/WebGPU)

### Идея
Держать текущие модели в ONNX:
- `pose_word_model.onnx`;
- `bio_segmenter.onnx`;
и исполнять их в браузере через ONNX Runtime Web.

### Плюсы
- минимальная дистанция до текущего backend-стека (ONNX уже используется на сервере);
- можно начать с `wasm`, затем экспериментально включать `webgpu`;
- проще поддерживать единые артефакты для backend/frontend.

### Риски
- размеры моделей и холодный старт (скачивание + компиляция);
- разные профили производительности на Chrome/Edge/Safari;
- ограничения мобильных браузеров и нестабильность WebGPU.

### Что нужно сделать позже
1. Реально подключить `predictBio()` и `predictWord()` в browser backend.
2. Настроить ring buffer и decoding сегментов прямо во фронте.
3. Добавить version pinning/хэш-контроль моделей в `manifest.json`.
4. Добавить feature flag для автопадения в backend path при перегрузке браузера.

## Вариант B: TFJS (WebGPU/WebGL/WASM)

### Идея
Переехать на стек TensorFlow.js для клиентского инференса (или для части моделей), как альтернативу ORT Web.

### Плюсы
- mature JS API для браузерных пайплайнов;
- гибкая интеграция с WebGPU/WebGL.

### Минусы
- потребуется конвертация моделей и проверка эквивалентности;
- усложняется единый артефактный pipeline с backend ONNX;
- выше риск дрейфа качества между backend и frontend версиями.

### Когда рассматривать
- если ORT Web не даст приемлемую latency/стабильность;
- если решим обучать модель сразу с экспортом в tfjs target.

## Что остается на клиенте в обеих стратегиях
Минимум, который уже есть:
- захват камеры;
- canvas overlay;
- skeleton viewer;
- debug/payload UI.

При полном client-side режиме два подварианта:
1. `MediaPipe Holistic` в браузере: камера -> landmarks/features -> BIO + word classifier (все локально).
2. Extraction на backend, а клиент делает только инференс/рендер (гибридный режим, обычно как промежуточный этап).

## Что нужно решить на следующем этапе
1. Размер моделей и стратегия загрузки:
   - eager preload или lazy load по режиму;
   - chunking/компрессия.
2. Кэширование:
   - HTTP cache headers;
   - service worker (опционально).
3. Безопасность:
   - какие артефакты можно безопасно отдавать в браузер;
   - защита от подмены артефактов (hash/version check).
4. Производительность:
   - baseline на WASM;
   - сравнение WebGPU vs WASM на целевых устройствах;
   - warmup и memory pressure.
5. Совместимость браузеров:
   - fallback policy для Safari/старых Chrome;
   - явные требования к платформе в README.

## Технический ориентир на будущее
- Референс архитектуры sign.mt: [sign/translate](https://github.com/sign/translate)
- Альтернатива inference runtime: [TensorFlow.js](https://github.com/tensorflow/tfjs)
- Документация ONNX Runtime Web: [onnxruntime.ai/docs/get-started/with-javascript/web.html](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)

## Статус после этого этапа
- `backend` inference остается default и production-path.
- `browser` mode в UI помечен как experimental.
- Есть готовый scaffold, в который можно поэтапно вставлять реальный client-side inference без слома существующего WS пути.
