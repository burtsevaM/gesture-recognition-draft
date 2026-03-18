# Pose Words Validation Data

Этот каталог предназначен для **локально воспроизводимой technical validation** пайплайна `pose_words`.

Что важно:

- validation fixtures и derived datasets генерируются локально, а не считаются продуктовым датасетом;
- generated outputs складываются в `backend/data/pose_words_validation/generated/`;
- этот каталог не version-control-ится, чтобы не тащить в git лишние бинарники и производные артефакты;
- верхнеуровневый сценарий подготовки и прогона находится в `backend/scripts/run_pose_words_validation.py`.

Validation data из этого workflow предназначены только для подтверждения пути:

`fixtures -> train -> export -> runtime -> smoke/integration`

Они **не** подтверждают продуктового качества распознавания и **не** заменяют реальный датасет.
