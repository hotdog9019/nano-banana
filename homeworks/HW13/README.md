# HW13

Требования: `TS/S13-homework.md`.

См. `HW13.ipynb`:

- загружает датасет `emotion` через `datasets`;
- демонстрирует токенизацию и инференс предобученной моделью;
- выполняет fine-tuning `DistilBERT` под классификацию эмоций;
- сохраняет артефакты в `./artifacts/` (предсказания, матрица ошибок, результаты обучения).

Зависимости: `torch`, `transformers`, `datasets`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
