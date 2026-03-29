# HW10-11 — компьютерное зрение: CNN, transfer learning и сегментация

## 1. Кратко: что сделано

- Построены пайплайны для задачи классификации на `STL10` и для сегментации на `OxfordIIITPet`.
- Для классификации сравнивались варианты обучения: простой CNN без аугментаций/с аугментациями и transfer learning на `ResNet18` (head-only и частичный fine-tune).
- Для сегментации использовалась предобученная модель (семейство DeepLab), показана визуализация предсказаний и базовые метрики.
- Все ключевые артефакты (графики, лучшая модель, конфиг, таблица прогонов) сохраняются в `./artifacts/`.

## 2. Среда и воспроизводимость

- Seed: `42` (см. `HW10-11.ipynb`)
- Устройство: CPU/GPU определяется автоматически (`cuda` при наличии)
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All

## 3. Данные

- Классификация: `STL10` (train/test из torchvision + val split 80/20 из train)
- Сегментация: `OxfordIIITPet` (из torchvision)
- Папка для данных: `./data/` (скачивание автоматически при первом запуске)

## 4. Часть A: модели и обучение (C1-C4)

Задача: классификация STL10. Для выбора лучшего подхода используется train/val split 80/20, **test** применяется только один раз — для финальной оценки лучшей модели по `val_accuracy`.

- C1: простой CNN без аугментаций
- C2: простой CNN + аугментации
- C3: `ResNet18` (замороженный backbone, обучается только голова)
- C4: `ResNet18` (частичный fine-tune верхних слоёв)

Превью аугментаций (C2): `artifacts/figures/augmentations_preview.png`

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

Задача: сегментация OxfordIIITPet предобученной моделью DeepLabV3.

- V1: threshold=0.5
- V2: threshold=0.7 + min-area filtering

Визуализация примеров: `artifacts/figures/segmentation_examples.png`  
Метрики: `artifacts/figures/segmentation_metrics.png`

## 6. Результаты

- Таблица прогонов: `artifacts/runs.csv`
- Лучшая модель классификатора: `artifacts/best_classifier.pt`
- Конфиг лучшего прогона: `artifacts/best_classifier_config.json`

Графики/артефакты:
- `artifacts/figures/classification_curves_best.png`
- `artifacts/figures/classification_compare.png`
- `artifacts/figures/augmentations_preview.png`
- `artifacts/figures/segmentation_examples.png`
- `artifacts/figures/segmentation_metrics.png`

## 7. Анализ

Transfer learning (ResNet18) даёт заметный прирост качества на STL10 по сравнению с SimpleCNN, а аугментации помогают улучшить обобщающую способность. В сегментации качество чувствительно к порогу и пост-обработке.

## 8. Итоговый вывод

Transfer learning на `ResNet18` обычно даёт более высокое качество на `STL10`, чем обучение простого CNN “с нуля”, особенно при аккуратном fine-tune. Для сегментации предобученные модели дают разумный baseline без сложной настройки.
