# Воспроизведение обучения

Это описание фактической цепочки обучения, а не итоговый выбор сдаваемой модели. После фиксации checkpoint сюда добавляются окончательные этапы и число шагов. Все команды выполняются из корня проекта после установки requirements.txt. Для нового запуска отключается календарный cutoff через end_utc=null; каталог каждого запуска должен быть новым.

## 1. Данные

```bash
python ops/download_data.py --parts train-clean-100 dev-clean dev-other test-clean test-other
mkdir -p data/datasets
for archive in data/downloads/*.tar.gz; do tar -xzf "$archive" -C data/datasets; done
python ops/build_staging.py
```

build_staging.py выбирает первые 1024 FLAC-записи в исходном официальном архиве train-clean-100, сохраняет тексты и список идентификаторов. Это обучающие записи. Dev используется для выбора; test не участвует ни в обучении, ни в подборе декодера.

## 2. Собственная начальная модель

Фактический staging-запуск обучался 100 эпох (3200 шагов), AdamW 3e-4, warmup 50, cosine horizon 3200, seed 1, четыре аугментации. Основные запуски получили собственные веса лучшего dev-clean checkpoint: шаг 2784, эпоха 86 при нумерации с нуля. Его recorded dev-clean WER равен 0.8242848803269118; это только инициализация, а не качество основной модели.

Полный запуск:

```bash
python train.py custom_train=data/downloads/staging data_root=data/datasets run_dir=runs/repro-staging epochs=100 warmup_steps=50 end_utc=null
```

Чтобы воспроизвести именно использованный префикс обучения, можно остановиться на шаге 2784. epochs=100 сохраняет исходный горизонт cosine scheduler; уменьшать epochs до 87 вместо этого нельзя:

```bash
python train.py custom_train=data/downloads/staging data_root=data/datasets run_dir=runs/repro-staging-prefix epochs=100 warmup_steps=50 max_steps=2784 end_utc=null
```

В следующем этапе использовать runs/repro-staging-prefix/last_model.pt. В исходной кампании тот же шаг был экспортирован как staging/best_model.pt. Веса из интернета и готовые акустические модели не использовались.

## 3. Обучение на train-clean-100

```bash
python train.py data_root=data/datasets train_parts=[train-clean-100] init_checkpoint=runs/repro-staging-prefix/last_model.pt run_dir=runs/repro-main epochs=60 lr=0.0003 warmup_steps=500 augment=true end_utc=null
```

Контрольный вариант без аугментаций начинался с того же checkpoint, seed, размера batch 32 и горизонта scheduler 60 эпох:

```bash
python train.py data_root=data/datasets train_parts=[train-clean-100] init_checkpoint=runs/repro-staging-prefix/last_model.pt run_dir=runs/repro-noaug epochs=60 lr=0.0003 warmup_steps=500 augment=false end_utc=null
```

Для сравнения на десяти эпохах добавить max_steps=8920, сохранив epochs=60. Сохранённые исходные main10/no_aug10 и их dev-сравнения входят в логи. Не сравнивать разные шаги как одинаковый бюджет.

## 4. Последующие этапы

Кампания сначала использует оставшееся время до 06:00 для refinement на 100 часах: новый optimizer/scheduler, LR 1e-4, warmup 100, горизонт 120 эпох. После 06:00 используются 460 часов только при полной проверке и распаковке train-clean-360 до cutoff; иначе остаются 100 часов. Начальная модель каждого этапа выбирается по измеренному dev-clean WER среди завершённых кандидатов. Дальнейшее обучение ограничено 12:30 МСК.

Для 460 часов заранее загрузить и распаковать официальный train-clean-360.tar.gz. Его проверенный SHA-256: 146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf.

Фактический выбранный родитель, конфигурация, количество шагов и SHA-256 финальных весов будут внесены после завершения кампании. Состояния optimizer, scheduler, RNG и накопленных градиентов сохраняются в доверенных full checkpoint; model-only checkpoint предназначен для инференса.

## 5. Проверка и пределы воспроизводимости

Каждый run содержит config.json, train_manifest.json, metrics.jsonl, offline W&B и media. В metrics.jsonl событие initialize_from_model содержит путь и source_state родительского checkpoint. Эту запись нужно использовать для проверки цепочки, а не угадывать родителя по названию файла.

Точные параметры и данные позволяют воспроизвести процедуру. Побитовое совпадение весов между разными GPU, версиями CUDA и недетерминированными CUDA kernels не обещается. Resume проверен отдельным регрессионным тестом со сравнением следующего шага, включая dropout, частичное накопление и RNG.

## Дополнительный кандидат по learning rate

После 06:00 и завершения `no_aug` отдельный контроллер `python ops/supervisor.py lr_candidate` использует GPU 1. Конфигурация данных и архитектура совпадают с основной кампанией, начальный LR нового этапа — 3e-4 вместо 1e-4, warmup — 100. Имена запусков начинаются с `lr_candidate_`, cutoff — 12:30 МСК. Это кандидат для выбора по dev, а не строго контролируемая абляция LR: фактический родительский checkpoint каждого этапа записывается в логи и может различаться. Проверка аугментаций с одинаковой инициализацией отдельно сохранена на 8920 шагах.
