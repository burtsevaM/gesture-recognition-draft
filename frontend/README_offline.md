# Frontend Offline / Client-Side Notes

## Что добавлено
В UI добавлен переключатель `inference_location`:
- `backend` (по умолчанию): действующий WebSocket inference через FastAPI.
- `browser (experimental)`: пробует инициализировать ONNX Runtime Web и загрузить модели, но рабочий inference пока остается на backend.

Это сделано как задел под полный client-side режим без ломки текущего MVP.

## Как включить experimental browser mode
1. Открой приложение.
2. В блоке камеры выбери `inference_location = browser (experimental)`.
3. Нажми `Включить камеру`.
4. В статусе рядом с переключателем появится результат инициализации:
   - если ORT/models загрузились, увидишь сообщение про experimental режим;
   - если не загрузились, автоматически будет fallback на backend WebSocket.

## Где находится scaffold-код
- `/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2/frontend/offline/inference_backend.js` — базовый интерфейс backend.
- `/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2/frontend/offline/backend_ws.js` — текущий backend path.
- `/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2/frontend/offline/backend_browser_ort.js` — ORT Web experimental loader (init + fallback).

## Подготовка артефактов для фронта
Чтобы сложить модели в фронтовую директорию:

```bash
cd "/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2"
./.venv/bin/python backend/scripts/export_artifacts_for_frontend.py
```

После запуска файлы копируются в:
- `/Users/mariaburtseva/Documents/проект грант/mvp1/SuperLuchito--SimpleGesture2Letter-Model-Version-2/frontend/assets/models`

И создается:
- `manifest.json` (имена файлов, размеры, хэши, T/F если доступны из config-файлов).

## Ограничения текущего этапа
- Browser mode пока не выполняет реальный инференс в runtime.
- Нет полной оптимизации под WebGPU/WebGL/WASM.
- Нет гарантированной производительности на всех браузерах.
- Для production пока использовать `backend` режим.
