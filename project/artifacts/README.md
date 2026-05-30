# Артефакты

Эта папка содержит результаты запуска `python -m src.train`:

- `ticket_classifier.joblib` - сериализованный финальный пайплайн `TF-IDF + SGDClassifier`.
- `metrics.json` - размер данных, сравнение моделей, метрики baseline/final model и матрица ошибок.
- `sample_predictions.csv` - тестовые примеры с предсказанными категориями.

Если артефакты нужно пересоздать, запустите обучение из корня проекта:

```bash
python -m src.train
```

