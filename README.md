# Черновой репозиторий контуров распознавания РЖЯ

Это рабочий draft-репозиторий по распознаванию русского жестового языка. В нем одновременно сосуществуют runtime, training, validation и experimental части, поэтому это не чистовой продуктовый repo, а рабочий контур для фиксации текущего состояния, архитектурных решений и перехода между pipeline.

Для словарного распознавания в репозитории сейчас сосуществуют два режима: `words` и `pose_words`. Архитектурно целевым основным pipeline уже выбран `pose_words`, а `words` сохраняется как baseline/reference до прохождения validation gates. Отдельно существует `letters` как режим буквенного распознавания.

## Краткое описание

- Репозиторий содержит backend, frontend demo, training/export скрипты, локальные runtime artifacts, validation workflow и архитектурную документацию.
- Основной текущий архитектурный вектор для словарного распознавания направлен в `pose_words`.
- `words` пока не удаляется и не считается закрытым: он нужен как baseline/reference pipeline.
- `letters` остается отдельным режимом распознавания статических букв.

## Что есть в репозитории

### `letters`

Буквенный retrieval-пайплайн на базе MediaPipe, DINOv2 и FAISS. Это отдельный локально рабочий режим для статических букв, но не основной вектор развития словарного распознавания.

### `words`

RGB sliding-window pipeline для isolated words. Сейчас он сохраняется как baseline/reference для сравнения runtime-поведения, качества и интеграционных сценариев, пока `pose_words` проходит validation gates.

### `pose_words`

Pose-first pipeline со skeleton extraction, BIO segmentation и word classifier. Именно он выбран как целевой основной pipeline для распознавания слов, но это не означает production-ready зрелость: в репозитории подтвержден technical validation/runtime path, а не полное качество на целевом датасете.

## Текущее архитектурное направление

- `pose_words` зафиксирован как целевой основной pipeline для распознавания слов.
- `words` остается baseline/reference до прохождения обязательных validation gates.
- Наличие active non-dummy runtime path для `pose_words` подтверждает техническую воспроизводимость, но не означает, что `words` уже можно удалить или заморозить.

Ключевые документы:

- [ADR-001: pose_words как целевой pipeline](docs/adr/ADR-001-pose-words-target.md)
- [Validation gates для перехода от words к pose_words](docs/validation-gates.md)
- [Техническая валидация pose_words](docs/pose_words_technical_validation.md)

## Статус репозитория

Это draft/sandbox working repository, в котором смешаны:

- runtime backend;
- frontend demo;
- training/export scripts;
- validation workflows;
- локальные артефакты и recovery/bootstrap path;
- experimental и research части.

Из этого рабочего контура позже должен выделяться более чистый основной репозиторий. Текущий репозиторий нужен для фиксации состояния, проверки гипотез, технической валидации и управляемого архитектурного перехода.

## Что здесь уже есть

- `backend/` — FastAPI runtime, WebSocket-контур, ML pipelines, train/export scripts, tests и artifact wiring.
- `frontend/` — локальный demo UI поверх backend WebSocket.
- `docs/` — описание текущего состояния, целевой архитектуры, ADR, validation gates и technical validation.
- `backend/train/` и `backend/scripts/` — обучение, экспорт, bootstrap, validation и runtime utility workflows.
- `backend/artifacts/` — retrieval, words, pose_words и runtime artifact directories, включая active runtime path.
- `backend/tests/` — unit, smoke и contract tests.

## Рекомендуемый entrypoint

Этот README намеренно не дублирует внутренние runbook'и. Для практической работы удобнее входить в проект через специализированные документы:

- Для backend runtime и локального запуска: [backend/README.md](backend/README.md)
- Для структуры pose_words artifacts и active runtime path: [backend/docs/pose_words_artifacts.md](backend/docs/pose_words_artifacts.md)
- Для reproducible technical validation path: [docs/pose_words_technical_validation.md](docs/pose_words_technical_validation.md)
- Для снимка текущего состояния репозитория: [docs/current-state.md](docs/current-state.md)
- Для целевой архитектуры и migration context: [docs/target-architecture.md](docs/target-architecture.md)

## Ограничения и честные оговорки

- Не все pipeline в репозитории одинаково зрелые по runtime, training/export и integration coverage.
- Наличие technical validation path и active non-dummy runtime path для `pose_words` не означает production-ready quality.
- Наличие `pose_words` как target pipeline не означает, что `words` уже можно удалить, отключить или окончательно вывести из основной линии.
- Репозиторий по-прежнему остается рабочим draft-контуром, а не чистовым продуктовым репозиторием.

## Лицензия

MIT, см. `LICENSE`.
