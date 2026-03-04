# Pose Words Artifacts

## Обязательные файлы

Для режима `recognition_mode: pose_words` + `segmentation.enabled: true` backend ожидает:

- `backend/artifacts/pose_word_model.onnx`
- `backend/artifacts/pose_word_labels.txt`
- `backend/artifacts/pose_word_config.json`
- `backend/artifacts/bio_segmenter.onnx`
- `backend/artifacts/bio_thresholds.json`
- `backend/artifacts/bio_config.json`

Если файлов нет, сервер не падает, но в `GET /health` вернется:

- `pose_words_ready: false`
- `missing_artifacts: [...]`

## Проверка наличия файлов

Из корня репозитория:

```bash
find backend/artifacts -type f \
  \( -name 'pose_word_model.onnx' -o -name 'pose_word_labels.txt' -o -name 'pose_word_config.json' \
     -o -name 'bio_segmenter.onnx' -o -name 'bio_thresholds.json' -o -name 'bio_config.json' \) \
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
- `"missing_artifacts": []`

## Ручной запуск генераторов

```bash
python backend/train/make_dummy_pose_word_model.py
python backend/train/make_dummy_bio_segmenter.py
```

`make_dummy_*` создают минимальные baseline ONNX для восстановления пайплайна.  
Качество распознавания у них нецелевое: они нужны для того, чтобы `pose_words` работал end-to-end без падения.
