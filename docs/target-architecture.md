# Целевая архитектура проекта

## 1. Цель документа

Этот документ описывает **рекомендуемое будущее состояние** проекта.

Документ нужен для:
- подготовки чистого основного репозитория;
- переноса в него только необходимых и воспроизводимых компонентов;
- отделения продуктового runtime от экспериментального кода и черновых исследований.

Важно:
- документ опирается на анализ текущего состояния проекта;
- рекомендации ниже **не означают**, что такое устройство уже реализовано в текущем коде;
- это ориентир для migration path, а не описание текущей реализации.

## 2. Основные архитектурные принципы

Новая версия репозитория должна строиться на следующих принципах:

- **Явное разделение ответственности.** API, контракты, инференс, preprocessing, segmentation, decoding, training и tools не должны быть перемешаны в одном модуле.
- **Один продуктовый путь, остальные — явно вторичны.** Основной runtime pipeline должен быть один; альтернативы должны быть помечены как baseline или experimental.
- **Воспроизводимость.** Любой обязательный runtime-артефакт должен иметь понятный источник: download, export или training pipeline.
- **Отделение API от ML-логики.** WebSocket/REST слой должен быть тонким, а не содержать половину бизнес-логики пайплайна.
- **Контракт как отдельный слой.** Формат сообщений между frontend и backend должен быть формализован, минимально стабилен и версионируем.
- **Проверяемость.** Для каждого ключевого слоя должны существовать локальные тесты и smoke-проверки.
- **Минимизация хаоса экспериментального кода.** Эксперименты не должны лежать в том же слое, что и основной runtime.
- **Документируемость.** Стартовый сценарий, required artifacts, health/readiness и dev workflow должны быть описаны отдельно и кратко.
- **Готовность к веб-интеграции.** Frontend не должен угадывать поведение backend по случайным полям payload.
- **Поддержка issue-based разработки.** Архитектура должна позволять развивать независимые части без каскадного редактирования монолитного файла.

## 3. Рекомендуемый основной pipeline

### 3.1. Найдено сейчас

По текущему коду в репозитории сосуществуют три независимых режима:
- `letters` — статические буквы через MediaPipe + DINOv2 + FAISS retrieval;
- `words` — isolated words по RGB sliding-window ONNX model;
- `pose_words` — pose-first runtime со скелетом, feature extraction, BIO-сегментацией и pose-word classifier.

При этом:
- `letters` выглядит как локально завершенный MVP для букв;
- `words` выглядит как более зрелый runtime для слов, чем его training/export часть;
- `pose_words` архитектурно наиболее перспективен для дальнейшего развития; в текущем рабочем дереве для него уже подтверждены technical validation path и supported active non-dummy runtime path, но не закрыты quality/stability gates.

### 3.2. Рекомендуется сделать

Рекомендуется зафиксировать **один основной продуктовый pipeline для слов**:

- **Основной product-oriented pipeline:** `pose_words`.
- **Baseline / reference pipeline:** `words`.
- **Отдельный буквенный baseline:** `letters`.

### 3.3. Почему рекомендуется `pose_words` как целевая основа

Не потому, что он уже является самым рабочим режимом сейчас, а потому что он лучше соответствует будущей архитектуре распознавания слов и последовательностей:
- позволяет явно работать с позой и руками, а не только с пикселями;
- лучше масштабируется к сегментации и temporal modeling;
- уже содержит skeleton payload и BIO segmentation слой;
- архитектурно ближе к будущему continuous sign pipeline.

### 3.4. Что делать с `words`

Рекомендуется:
- **не делать `words` главным продуктовым путем** в новой архитектуре;
- оставить его как baseline/reference pipeline;
- использовать для сравнения качества, latency и fallback-экспериментов;
- если он не нужен продуктово, перевести его в `experimental/` или отдельный baseline-модуль.

Практические критерии, по которым `words` можно переводить в `frozen`/`deprecated`/`baseline-only` статус, зафиксированы в [docs/validation-gates.md](validation-gates.md).

### 3.5. Что делать с `letters`

Рекомендуется:
- не тащить буквенный retrieval-MVP как равноправный runtime в основной word-oriented репозиторий;
- сохранить его как отдельный baseline или как отдельный подрепозиторий/архивный эксперимент;
- переносить его в чистовой word repo только если он нужен продуктово и поддерживается осознанно.

### 3.6. Нужен ли один режим или несколько

Рекомендуемая схема:
- **внешне** иметь один продуктовый контур “word recognition”;
- **внутри** держать один основной backend (`pose_words`) и, при необходимости, альтернативный baseline (`words`).

То есть в чистовой архитектуре не стоит держать три полностью равноправных product режима.

## 4. Рекомендация по границам репозитория

### 4.1. Найдено сейчас

Сейчас один репозиторий содержит одновременно:
- runtime backend;
- frontend;
- буквенный MVP;
- RGB words runtime;
- pose-first runtime;
- training/export;
- датасетные утилиты;
- nested Slovo repo;
- экспериментальный client-side scaffold;
- локальные артефакты и runtime-логи.

### 4.2. Рекомендуется сделать

Рекомендуется разделить пространство ответственности на два уровня:

**1. Основной чистовой репозиторий**
- только то, что нужно для воспроизводимого продуктового runtime и его поддержки;
- основной pipeline распознавания слов;
- минимально необходимый frontend и API-контракт;
- только те tools, которые нужны для runtime/оценки/операционного использования.

**2. Draft / sandbox / research repo**
- тренировочные эксперименты;
- альтернативные пайплайны;
- датасетная сборка и черновые export-пайплайны;
- вложенные внешние репозитории;
- временные модели, dummy bootstrap, runtime logs, exploratory docs.

Верхний `README.md` такого draft-репозитория должен явно отражать этот sandbox/migration context и направлять к специализированным runbook/architecture docs, а не описывать проект как уже стабилизированный продуктовый runtime.

### 4.3. Четкий вывод

- Для основной продуктовой линии нужен **отдельный чистовой основной репозиторий**.
- Для текущего состояния и продолжающихся экспериментов нужен **отдельный draft/sandbox repo**.
- Буквы и слова **не стоит** держать как одинаково важные product-модули в одном чистовом word-oriented репозитории.
- Если буквенный режим остается самостоятельным направлением — лучше вынести его отдельно.

## 5. Целевая структура основного репозитория

Рекомендуемая структура:

```text
docs/
configs/
src/
  api/
  contracts/
  runtime/
  pipelines/
  features/
  segmentation/
  inference/
  decoding/
  assets/
tests/
tools/
artifacts/
experiments/   # опционально, минимально или отдельно
```

### `docs/`

**Зачем нужна**
- архитектурные документы;
- runbook;
- описание контрактов;
- описание required artifacts.

**Что должно лежать**
- current architecture overview;
- target architecture;
- run/start guide;
- artifact policy;
- integration docs.

**Что не должно лежать**
- временные заметки про отдельные эксперименты;
- несогласованные черновики без статуса.

### `configs/`

**Зачем нужна**
- хранение runtime-конфигов и профилей окружений.

**Что должно лежать**
- базовый config;
- dev/demo configs;
- pose/segmentation thresholds defaults.

**Что не должно лежать**
- runtime-логи;
- артефакты моделей;
- секреты и локальные machine-specific пути.

### `src/api/`

**Зачем нужна**
- FastAPI/WS transport слой.

**Что должно лежать**
- роуты;
- startup/shutdown;
- dependency wiring;
- health/readiness handlers.

**Что не должно лежать**
- математика нормализации;
- сегментация;
- inference orchestration на сотни строк.

### `src/contracts/`

**Зачем нужна**
- единое описание входных/выходных схем.

**Что должно лежать**
- схемы WS payload;
- versioned message contracts;
- serialизация/валидация.

**Что не должно лежать**
- UI-специфические импровизации;
- ad hoc dict-конструирование по месту.

### `src/runtime/`

**Зачем нужна**
- orchestration конкретного product pipeline.

**Что должно лежать**
- session processor;
- runtime context;
- readiness logic;
- lifecycle сервисов.

**Что не должно лежать**
- HTTP routes;
- training/export код.

### `src/pipelines/`

**Зачем нужна**
- изоляция отдельных пайплайнов.

**Что должно лежать**
- `pose_words/` как основной pipeline;
- `words_rgb_baseline/` только если baseline действительно нужен;
- возможно `letters_baseline/` только если буквы остаются внутри чистового репо.

**Что не должно лежать**
- случайные mixed-mode if/else в одном файле.

### `src/features/`

**Зачем нужна**
- preprocessing, pose extraction, normalization, feature composition.

**Что должно лежать**
- `pose/extractor`;
- `pose/datatypes`;
- `pose/normalization`;
- feature schemas.

**Что не должно лежать**
- WebSocket логика;
- JSON payload assembly.

### `src/segmentation/`

**Зачем нужна**
- BIO inference, decoding и streaming logic.

**Что должно лежать**
- ONNX wrapper;
- decoder;
- streaming segmenter;
- segmentation metrics.

**Что не должно лежать**
- pose extractor;
- frontend debug-специфика.

### `src/inference/`

**Зачем нужна**
- обертки вокруг моделей и артефактов.

**Что должно лежать**
- ONNX model wrappers;
- labels/config loaders;
- artifact validation.

**Что не должно лежать**
- training;
- dataset preparation.

### `src/decoding/`

**Зачем нужна**
- hold/cooldown/commit/unknown/no_event логика.

**Что должно лежать**
- decision decoders;
- post-processing state machines.

**Что не должно лежать**
- inference моделей;
- FastAPI routes.

### `src/assets/` или `artifacts/`

**Зачем нужна**
- хранение runtime-required артефактов по стабильной схеме.

**Что должно лежать**
- только те артефакты, которые реально нужны для запуска сервиса.

**Что не должно лежать**
- runtime logs;
- временные датасеты;
- вложенные внешние репозитории;
- dummy bootstrap артефакты, если они не используются как test fixtures.

### `tests/`

**Зачем нужна**
- unit/integration/contract/smoke tests.

**Что должно лежать**
- tests по слоям: contracts, normalization, segmentation, runtime, smoke.

**Что не должно лежать**
- ad hoc manual notebooks или разовые локальные проверки.

### `tools/`

**Зачем нужна**
- операционные утилиты вокруг runtime.

**Что должно лежать**
- только то, что реально нужно команде для эксплуатации и оценки.

**Что не должно лежать**
- полигон обучения и большие исследовательские конвейеры.

## 6. Границы модулей и ответственность

Рекомендуемое разделение по слоям:

### Transport / API layer

Должен отвечать только за:
- HTTP/WS маршруты;
- прием и отправку сообщений;
- startup/shutdown;
- health/readiness.

### Contract layer

Должен отвечать только за:
- схемы входных/выходных сообщений;
- сериализацию;
- versioning payload.

### Inference layer

Должен отвечать только за:
- загрузку ONNX/labels/config;
- вызов моделей;
- проверку совместимости артефактов.

### Preprocessing / features layer

Должен отвечать только за:
- pose extraction;
- normalization;
- feature composition;
- resampling.

### Segmentation layer

Должен отвечать только за:
- BIO inference;
- streaming decode;
- segment extraction rules.

### Temporal model layer

Должен отвечать за:
- классификацию сегмента слова;
- RGB baseline classifier, если он сохраняется.

### Decoding / post-processing layer

Должен отвечать за:
- `NONE/UNKNOWN/HOLD/COMMIT/COOLDOWN`;
- dedup;
- commit text assembly.

### Evaluation / testing layer

Должен отвечать за:
- smoke scripts;
- contract tests;
- calibration;
- metrics.

### Config / assets layer

Должен отвечать за:
- стабильные пути к артефактам;
- env-specific настройки;
- отсутствие скрытых machine-local зависимостей.

## 7. Контракты и интеграция с веб-платформой

### 7.1. Найдено сейчас

Сейчас frontend и backend связаны через один фактический WS payload без явной версии контракта.

### 7.2. Рекомендуется сделать

Нужно выделить отдельный contract layer и зафиксировать:
- входной формат кадров и control-сообщений;
- базовый обязательный payload;
- optional debug/pose/perf/segments блоки;
- версию контракта.

### 7.3. Минимальные требования к контракту

Рекомендуется ввести:
- `contract_version` в WS payload;
- отдельный документ `docs/ws-contract.md`;
- типизированные схемы/модели в коде;
- отдельные integration tests на совместимость frontend/backend.

### 7.4. Как не допустить расхождения frontend и backend

Рекомендуется:
- держать contract tests в backend;
- держать mock payload fixtures для frontend;
- любые изменения optional/required полей проводить только через PR с обновлением документации и smoke-проверки.

### 7.5. Нужен ли mock protocol

Да, рекомендуется.

Минимально нужен:
- один фиксированный набор JSON payload fixtures;
- local integration test, который гоняет frontend против mock payload или backend smoke endpoint.

## 8. Что переносить в новый чистовой репозиторий в первую очередь

### Переносить первым

- FastAPI/WS каркас и health/readiness слой.
- Базовый frontend viewer.
- Pose extraction + normalization + feature composition.
- BIO segmentation runtime.
- Pose-word classifier runtime wrapper.
- Contract tests и smoke tests.

### Переносить вторым

- Perf/profiler слой.
- Structured debug payload (`skeleton`, `segments`, `perf`, `bio`).
- Artifact validation/readiness logic.
- Runbook и artifact docs.

### Переносить только после рефакторинга

- `SessionProcessor` и orchestration логику из монолитного `main.py`.
- Training/export scripts, если они действительно станут частью основного цикла.
- Frontend experimental browser scaffold — только если есть план на client-side inference.

### Временно не переносить

- `letters` retrieval pipeline как равноправный продуктовый режим.
- nested `backend/data/slovo_repo`.
- локальные gallery sessions.
- runtime logs (`words_runtime.jsonl`).
- dummy bootstrap артефакты как рабочие модели.

### Оставить в draft repo

- RGB `words` как baseline/reference, если он нужен для экспериментов;
- training sandbox;
- synthetic BIO dataset generation;
- альтернативные модели и export scaffolds;
- client-side experimental код до стабилизации.

### Можно удалить или заморозить при переносе

- локальные служебные и machine-specific файлы;
- лишние датасетные копии, если они не используются runtime;
- дублирующие docs-планы после переноса в единые архитектурные документы.

## 9. Что оставить как experimental

Рекомендуется явно пометить как experimental:
- `words` RGB pipeline, если основной product path становится pose-first;
- буквенный retrieval pipeline, если продуктовая цель — слова;
- browser/client-side inference scaffold;
- synthetic BIO dataset builder и связанные исследовательские скрипты;
- bootstrap dummy generators для pose/BIO моделей.

Как хранить experimental-часть:
- либо в отдельном draft/sandbox repo;
- либо в отдельной `experiments/` зоне с запретом тащить ее в основной runtime-контур.

Почему не стоит включать их сразу в основную продуктовую архитектуру:
- они увеличивают поверхность поддержки;
- создают несколько конкурирующих путей внутри одного сервиса;
- размывают границу между “рабочим runtime” и “исследованием”.

## 10. Рекомендации по процессу разработки

Компактный рекомендуемый процесс для чистового репозитория:

- Основная ветка: одна стабильная trunk/main ветка.
- Feature branches: короткоживущие ветки под одну задачу.
- PR discipline: один PR — одна архитектурная цель.
- Коммиты: короткие, технические, с единым стилем (`feat`, `fix`, `docs`, `test`, `refactor`).
- Issue structure: отдельные issue для contracts, runtime, segmentation, artifacts, frontend integration.
- Labels: `runtime`, `contract`, `frontend`, `segmentation`, `artifacts`, `research`, `infra`, `good-first-task`.
- Definition of Done: код, тесты, обновленный контракт/документация, smoke-check.
- Quality gates: lint/type checks по возможности, unit tests, contract tests, `/health` readiness, smoke run.
- Templates: PR template, issue template, artifact checklist, runtime change checklist.

## 11. Риски и архитектурные компромиссы

### Основные риски переноса

- Потерять фактический WS контракт, на который уже завязан фронтенд.
- Потерять неочевидные runtime зависимости на имена файлов артефактов.
- Случайно перенести dummy pose/BIO модели как будто это целевые production-модели.
- Смешать в новом репозитории продуктовый код и training/research код, повторив текущее состояние.
- Потерять важные helper-утилиты для датасета и воспроизводимости.

### Где решение зависит от дальнейших экспериментов

- Финальный статус `words` RGB pipeline: нужен ли он как baseline в основном репозитории или только в draft.
- Станет ли `letters` частью продукта или отдельным архивным baseline.
- Нужен ли browser/client-side inference в основном контуре.
- Когда `pose_words` пройдет не только technical validation на synthetic fixtures, но и более сильную quality/stability validation на целевом контуре и перестанет быть research-first runtime.

### Компромисс “быстро собрать рабочее” vs “сделать чисто”

Если цель — быстро получить работающий сервис:
- можно временно сохранить больше кода внутри одного репозитория.

Если цель — получить поддерживаемую архитектуру:
- нужно сразу отделить основной runtime от draft/research зоны, даже если это замедлит перенос.

Рекомендуемый компромисс:
- в новый чистовой репозиторий переносить только minimum viable runtime;
- экспериментальные и тренировочные части держать отдельно, но не удалять.

## 12. Итоговая рекомендация

- Основным будущим pipeline для распознавания слов рекомендуется сделать `pose_words`.
- Режим `words` рекомендуется оставить как baseline/reference, а не как главный продуктовый путь.
- Режим `letters` не рекомендуется переносить как равноправный runtime в основной word-oriented репозиторий.
- Основной чистовой репозиторий должен содержать только воспроизводимый runtime, контракт, минимальный frontend и обязательные tools/tests.
- Отдельный draft/sandbox repo нужен для training, export, экспериментов, альтернативных пайплайнов и локальных артефактов.
- Контракт frontend/backend нужно формализовать и версионировать отдельно от runtime-логики.
- `main.py` в текущем виде не должен быть перенесен в новый репозиторий без декомпозиции по слоям.
- Dummy pose/BIO артефакты нельзя трактовать как финальные runtime-модели; они должны быть либо test/bootstrap fixtures, либо остаться в draft.
- Перенос следует делать поэтапно: сначала transport+contracts+pose runtime, затем segmentation/classifier, затем tests/runbooks, и только после этого рассматривать перенос baseline/experimental частей.
- Архитектурное решение, которое стоит принять уже сейчас: новый основной репозиторий должен быть сфокусирован на одном product pipeline, а не на одновременной поддержке всех исторических режимов.
