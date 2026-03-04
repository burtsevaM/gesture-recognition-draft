# Pose Module (MVP)

Минимальный pose-first модуль для:

- извлечения позы из RGB-кадра (`PoseExtractor`),
- нормализации плеч (`shoulder_normalize`),
- канонизации кистей (`hand_normalize_3d`),
- сборки признаков для будущей модели (`compose_features*`).

## Пример: один кадр

```python
import cv2
from app.pose import PoseExtractor, shoulder_normalize, hand_normalize_3d, compose_features

bgr = cv2.imread("frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

with PoseExtractor(include_face=False) as extractor:
    frame = extractor.process(rgb)

if frame is not None:
    seq_norm, info = shoulder_normalize([frame])
    feature, aux = compose_features(seq_norm[0])
    print(feature.shape, info.normalized, aux["point_mask"].shape)
```

## Пример: последовательность

```python
from app.pose import compose_features_sequence

features, aux = compose_features_sequence(frames, include_velocity=True)
print(features.shape)
```

