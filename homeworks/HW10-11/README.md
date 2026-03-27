# HW10-11

Требования: `TS/S10-homework.md`.

См. `HW10-11.ipynb`:

- загружает датасеты `STL10` и `OxfordIIITPet` через `torchvision` в `./data/`;
- делает воспроизводимый train/val split (seed=42);
- запускает эксперименты по классификации (CNN / transfer learning) и сегментации;
- сохраняет артефакты в `./artifacts/` (таблица прогонов, конфиги, модель, графики).

Зависимости: `torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`.
