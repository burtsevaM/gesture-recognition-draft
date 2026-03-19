# Техническая валидация `pose_words`

## 1. Цель

Этот документ фиксирует честную техническую валидацию `pose_words` в текущем репозитории.

Подтверждался не продуктовый quality level, а воспроизводимый технический путь:

`fixtures -> train -> export -> install active runtime -> backend smoke -> validation report`

Целью было проверить, что `pose_words` можно довести до non-dummy validation artifacts и запустить на них полный локальный validation workflow без скрытой ручной магии.

## 2. Scope validation

В validation scope входили:

- генерация маленького детерминированного validation dataset;
- обучение минимального `pose_word` classifier;
- обучение минимального BIO segmenter;
- экспорт обеих моделей в ONNX;
- явное разделение dummy/bootstrap и validation artifacts через metadata markers;
- direct runtime replay на non-dummy validation artifacts;
- установка non-dummy artifacts как active runtime set для backend;
- запуск backend в режиме `pose_words` на active non-dummy artifacts;
- smoke-прогон и сбор machine-readable validation report.

В validation scope не входили:

- качество на широком реальном датасете;
- сравнение `pose_words` с `words`;
- подтверждение production-ready поведения;
- полноформатный camera/video end-to-end прогон с положительными распознаваниями через backend WebSocket.

## 3. Validation path

В репозитории теперь есть поддерживаемый двухшаговый workflow.

### Шаг 1. Подготовить non-dummy validation artifacts

```bash
source .venv/bin/activate
python backend/scripts/run_pose_words_validation.py
```

Runner выполняет следующий путь:

1. Генерирует локальные validation fixtures в `backend/data/pose_words_validation/generated/`.
2. Готовит classifier index и BIO source index.
3. Строит synthetic continuous BIO dataset.
4. Обучает `pose_word` classifier.
5. Обучает BIO segmenter.
6. Экспортирует обе модели в ONNX.
7. Поднимает backend c временной подменой конфига на validation artifacts.
8. Выполняет smoke check и пишет итоговый report в `backend/artifacts/validation/pose_words/technical_validation_report.json`.

### Шаг 2. Сделать их active runtime artifacts

```bash
source .venv/bin/activate
python backend/scripts/install_pose_words_runtime_artifacts.py --force
```

Installer проверяет metadata source artifacts, делает backup текущего active набора, копирует validated files в `backend/artifacts/runtime/active/pose_words/` и пишет `backend/artifacts/runtime/active/pose_words/pose_words_active_manifest.json`.

Выходные директории validation path intentionally не коммитятся:

- `backend/data/pose_words_validation/generated/`
- `backend/artifacts/validation/pose_words/`

Это локальные воспроизводимые outputs, а не продуктовые runtime artifacts.

## 4. Какие артефакты использовались

Validation path генерирует следующие non-dummy artifacts:

- `backend/artifacts/validation/pose_words/pose_word_model.onnx`
- `backend/artifacts/validation/pose_words/pose_word_labels.txt`
- `backend/artifacts/validation/pose_words/pose_word_config.json`
- `backend/artifacts/validation/pose_words/bio_segmenter.onnx`
- `backend/artifacts/validation/pose_words/bio_thresholds.json`
- `backend/artifacts/validation/pose_words/bio_config.json`

Для этих артефактов в metadata фиксируются маркеры:

- `artifact_kind: validation`
- `dataset_kind: synthetic_fixture`
- `trained: true`
- `source_pipeline: run_pose_words_validation`

Dummy/bootstrap artifacts при этом не удаляются, но остаются отдельным fallback-путем через `backend/scripts/bootstrap_pose_words_artifacts.py` и `backend/train/make_dummy_*`.

После шага install active runtime backend использует:

- `backend/artifacts/runtime/active/pose_words/pose_word_model.onnx`
- `backend/artifacts/runtime/active/pose_words/pose_word_labels.txt`
- `backend/artifacts/runtime/active/pose_words/pose_word_config.json`
- `backend/artifacts/runtime/active/pose_words/bio_segmenter.onnx`
- `backend/artifacts/runtime/active/pose_words/bio_thresholds.json`
- `backend/artifacts/runtime/active/pose_words/bio_config.json`
- `backend/artifacts/runtime/active/pose_words/pose_words_active_manifest.json`

## 5. Какие команды и проверки запускались

Основной validation run:

```bash
source .venv/bin/activate
python backend/scripts/run_pose_words_validation.py
```

Активация non-dummy runtime artifacts:

```bash
source .venv/bin/activate
python backend/scripts/install_pose_words_runtime_artifacts.py --force
```

Проверка active runtime:

```bash
source .venv/bin/activate
python backend/scripts/smoke_pose_words.py \
  --base-url http://127.0.0.1:8012 \
  --duration-sec 5 \
  --fps 8 \
  --require-pose-words true \
  --require-segmentation true \
  --require-non-dummy-active true \
  --log-path backend/artifacts/validation/pose_words/smoke_pose_words_active_runtime.jsonl
```

Дополнительно были повторно запущены релевантные regression checks:

```bash
source .venv/bin/activate
pytest backend/tests/test_artifacts_bootstrap.py \
  backend/tests/test_pose_words_integration_smoke.py \
  backend/tests/test_ws_contract.py \
  backend/tests/test_bio_dataset_builder.py \
  backend/tests/test_pose_word_model_io.py
```

## 6. Результаты

По итогам фактического прогона подтверждено следующее:

- train/export path для `pose_word` classifier отработал успешно; в локальном validation run получены `best_val_top1=1.0` и `test_top1=1.0` на tiny deterministic fixtures;
- train/export path для BIO segmenter отработал успешно; в локальном validation run получены `test_sign_boundary_f1=0.5455` и `test_phrase_boundary_f1=0.4`;
- экспорт обеих моделей в ONNX завершился успешно;
- generated artifacts промаркированы как `validation`, а не `dummy`;
- direct runtime replay на non-dummy validation artifacts завершился успешно: `segments_detected=26`, `committed_total_words=26`;
- installer успешно переводит validation artifacts в active runtime set `backend/artifacts/runtime/active/pose_words/*` и пишет `pose_words_active_manifest.json` с профилем `validation_active`;
- backend стартует на active non-dummy artifacts и через `/health` показывает `pose_words_ready=true`, `pose_words_validated_runtime_ready=true`, `pose_words_non_dummy_active=true`, `active_artifact_profile=validation_active`, `missing_artifacts=[]`;
- smoke script завершился без падения и оставил JSONL лог.

Одновременно зафиксированы важные ограничения самого прогона:

- direct runtime replay сработал на validation artifacts, но дал `expected_total_words=6` против `committed_total_words=26`; это подтверждает техническую работоспособность контура, но также показывает пере-сегментацию и отсутствие quality-stability guarantees;
- backend WebSocket smoke даже на active non-dummy artifacts использовал mock JPEG frames и не дал положительных segment/commit событий: `positive_segment_frames=0`, `committed_frames=0`;
- поэтому backend smoke подтверждает readiness и отсутствие падения сервиса, но не подтверждает положительное end-to-end распознавание на camera/video входе.

## 7. Ограничения

Эта валидация не доказывает:

- продуктового качества `pose_words`;
- качества на реальном широком датасете;
- устойчивости на демонстрационном или интеграционном контуре с камерой;
- отсутствия регрессий относительно `words`;
- готовности удалять, замораживать или деактивировать `words`.

Также важно:

- validation artifacts и active runtime manifest создаются локально и не должны трактоваться как production release artifacts;
- bootstrap path сохраняется как fallback/dev recovery и по-прежнему может быть активирован отдельно;
- наличие supported active non-dummy runtime path не подменяет собой baseline comparison и ручную stability validation.

## 8. Вывод

В текущем репозитории теперь подтвержден воспроизводимый technical validation path для `pose_words` на уровне `synthetic_fixture -> train -> export -> install active runtime -> backend smoke -> report`.

Это означает, что `pose_words` больше не опирается только на dummy/bootstrap path ни для технического подтверждения train/export/runtime связки, ни для supported active runtime workflow.

Это не означает, что `pose_words` уже готов заменить `words` как единственный активный pipeline. Для этого по-прежнему нужны baseline comparison, более сильная integration validation и отдельное подтверждение стабильности на целевом контуре.
