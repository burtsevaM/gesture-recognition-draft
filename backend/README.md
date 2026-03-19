# Backend QA Runbook

Единый сценарий запуска `pose_words + segmentation` для локальной проверки.

## 1) Установка зависимостей

```bash
cd <repo-root>
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

## 2) Где должны лежать артефакты

Минимальный набор для `pose_words + segmentation`:

- `backend/artifacts/runtime/active/pose_words/bio_segmenter.onnx`
- `backend/artifacts/runtime/active/pose_words/bio_thresholds.json`
- `backend/artifacts/runtime/active/pose_words/pose_word_model.onnx`
- `backend/artifacts/runtime/active/pose_words/pose_word_labels.txt`

Рекомендуемый доп. артефакт:

- `backend/artifacts/runtime/active/pose_words/pose_word_config.json`

Если артефактов нет, сгенерируй baseline локально одной командой:

```bash
python -m backend.scripts.bootstrap_pose_words_artifacts
```

Документация по артефактам: `backend/docs/pose_words_artifacts.md`.

Для честной non-dummy технической валидации есть отдельный workflow:

```bash
source .venv/bin/activate
python backend/scripts/run_pose_words_validation.py
```

Он обучает минимальные validation-модели, экспортирует ONNX, временно поднимает backend в `pose_words` и пишет локальный report в `backend/artifacts/validation/pose_words/technical_validation_report.json`. Подробности и ограничения зафиксированы в `docs/pose_words_technical_validation.md`.

Чтобы сделать эти non-dummy artifacts активным runtime-набором для backend, выполни:

```bash
source .venv/bin/activate
python backend/scripts/install_pose_words_runtime_artifacts.py --force
```

После этого `backend/artifacts/runtime/active/pose_words/*` становится active runtime set, а installer пишет manifest `backend/artifacts/runtime/active/pose_words/pose_words_active_manifest.json`.

## 3) Включение режима pose_words + segmentation

Из корня репозитория:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import yaml

cfg_path = Path("backend/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

cfg["recognition_mode"] = "pose_words"
seg = cfg.setdefault("segmentation", {})
seg["enabled"] = True

cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("updated:", cfg_path)
PY
```

Проверка текущего конфига:

```bash
python - <<'PY'
from app.config import load_config
cfg = load_config("backend/config.yaml")
print("mode=", cfg.recognition_mode)
print("segmentation.enabled=", cfg.segmentation_enabled)
PY
```

## 4) Запуск backend

```bash
cd backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Проверка health:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

## 5) Запуск frontend

Отдельно поднимать frontend-сервер не нужно: UI отдается backend-ом.

Открой в браузере:

- `http://127.0.0.1:8000`

## 6) Smoke QA (30 секунд)

Из корня репозитория:

```bash
source .venv/bin/activate
python backend/scripts/smoke_pose_words.py \
  --base-url http://127.0.0.1:8000 \
  --duration-sec 30 \
  --fps 12 \
  --require-pose-words true \
  --require-segmentation true \
  --require-non-dummy-active true \
  --log-path backend/artifacts/smoke_pose_words.jsonl
```

Скрипт делает:

1. Проверку `/health`, включая active artifact profile.
2. Вывод активного `mode` и `segmentation.enabled`.
3. При `--require-non-dummy-active true` проверку, что backend использует non-dummy active artifacts.
4. 30-секундный WS прогон с JPEG-кадрами (mock по умолчанию).
5. Сводку по `fps/latency/segments` и JSONL лог.

Если нужен прогон с видео вместо mock:

```bash
python backend/scripts/smoke_pose_words.py \
  --base-url http://127.0.0.1:8000 \
  --video /absolute/path/to/video.mp4 \
  --duration-sec 30
```

Важно: preferred supported path для `pose_words` теперь такой:

1. `python backend/scripts/run_pose_words_validation.py`
2. `python backend/scripts/install_pose_words_runtime_artifacts.py --force`
3. запуск backend
4. `python backend/scripts/smoke_pose_words.py --require-non-dummy-active true ...`

Bootstrap path сохраняется, но рассматривается как fallback/dev recovery, а не как основной подтвержденный runtime path.

## 7) Regression tests

Из корня репозитория:

```bash
source .venv/bin/activate
pytest backend/tests
```

Важные блоки регрессии:

- `letters` (существующие тесты retrieval/state-machine)
- `words` RGB (существующие тесты words buffer/decoder)
- WS contract (новые проверки обязательных/optional полей payload)
