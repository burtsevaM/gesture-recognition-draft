# Pose Words Artifacts

## Обязательные файлы

Для режима `recognition_mode: pose_words` + `segmentation.enabled: true` backend ожидает:

- `backend/artifacts/runtime/active/pose_words/pose_word_model.onnx`
- `backend/artifacts/runtime/active/pose_words/pose_word_labels.txt`
- `backend/artifacts/runtime/active/pose_words/pose_word_config.json`
- `backend/artifacts/runtime/active/pose_words/bio_segmenter.onnx`
- `backend/artifacts/runtime/active/pose_words/bio_thresholds.json`
- `backend/artifacts/runtime/active/pose_words/bio_config.json`

Эти пути трактуются как **active runtime artifact set** для `pose_words`.

Если файлов нет, сервер не падает, но в `GET /health` вернется:

- `pose_words_ready: false`
- `missing_artifacts: [...]`

## Проверка наличия файлов

Из корня репозитория:

```bash
find backend/artifacts -type f \
  \( -path 'backend/artifacts/runtime/active/pose_words/pose_word_model.onnx' \
     -o -path 'backend/artifacts/runtime/active/pose_words/pose_word_labels.txt' \
     -o -path 'backend/artifacts/runtime/active/pose_words/pose_word_config.json' \
     -o -path 'backend/artifacts/runtime/active/pose_words/bio_segmenter.onnx' \
     -o -path 'backend/artifacts/runtime/active/pose_words/bio_thresholds.json' \
     -o -path 'backend/artifacts/runtime/active/pose_words/bio_config.json' \) \
  | sort
```

## Генерация baseline-артефактов одной командой

Из корня репозитория:

```bash
python -m backend.scripts.bootstrap_pose_words_artifacts
```

Принудительно пересобрать:

```bash
python -m backend.scripts.bootstrap_pose_words_artifacts --force
```

## Проверка readiness после bootstrap

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

Ожидаемо:

- `"ok": true`
- `"recognition_mode": "pose_words"`
- `"pose_words_ready": true`
- `"pose_words_validated_runtime_ready": false`
- `"active_artifact_profile": "dummy_fallback"`
- `"pose_words_non_dummy_active": false`
- `"missing_artifacts": []`

## Ручной запуск генераторов

```bash
python backend/train/make_dummy_pose_word_model.py
python backend/train/make_dummy_bio_segmenter.py
```

`make_dummy_*` создают минимальные baseline ONNX для восстановления пайплайна.  
Качество распознавания у них нецелевое: они нужны для того, чтобы `pose_words` работал end-to-end без падения.

## Preferred non-dummy runtime path

Поддерживаемый путь для активного non-dummy runtime теперь состоит из двух шагов.

### Шаг 1. Получить non-dummy validation artifacts

```bash
source .venv/bin/activate
python backend/scripts/run_pose_words_validation.py
```

Этот runner не перезаписывает active runtime files в `backend/artifacts/`, а генерирует отдельные локальные outputs:

- `backend/data/pose_words_validation/generated/`
- `backend/artifacts/validation/pose_words/`

### Шаг 2. Установить их как active runtime artifacts

```bash
source .venv/bin/activate
python backend/scripts/install_pose_words_runtime_artifacts.py --force
```

Installer:

- проверяет, что source artifacts не dummy и имеют `trained: true`;
- делает backup текущего active набора в `backend/artifacts/runtime/backups/pose_words/...`;
- копирует validated artifacts в active runtime paths `backend/artifacts/runtime/active/pose_words/*`;
- пишет `backend/artifacts/runtime/active/pose_words/pose_words_active_manifest.json`.

### Как понять, что backend работает на active non-dummy artifacts

После установки и запуска backend проверь `/health`.

Ожидаемые признаки:

- `active_artifact_profile: validation_active`
- `pose_words_artifact_kind: validation`
- `bio_artifact_kind: validation`
- `pose_words_non_dummy_active: true`
- `pose_words_validated_runtime_ready: true`
- `missing_artifacts: []`

Также smoke script теперь поддерживает явную проверку active non-dummy profile:

```bash
source .venv/bin/activate
python backend/scripts/smoke_pose_words.py \
  --base-url http://127.0.0.1:8000 \
  --require-pose-words true \
  --require-segmentation true \
  --require-non-dummy-active true
```

## Metadata и manifest

Внутри validation/runtime artifacts и manifest используются markers:

- `artifact_kind`
- `dataset_kind`
- `trained`
- `source_pipeline`
- `generated_by`
- `active_artifact_profile`

Это нужно, чтобы не смешивать:

- `dummy/bootstrap` artifacts для fallback и быстрого dev recovery;
- `validation_active` artifacts для подтвержденного technical runtime path.

## Статус bootstrap path

Bootstrap path сохраняется, но считается fallback/dev recovery сценарием.

Если в active runtime установлен bootstrap-набор, `/health` должен показывать профиль `dummy_fallback`, а не `validation_active`.

Подробный итог validation run описан в `docs/pose_words_technical_validation.md`.
