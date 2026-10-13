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

После 06:00 и завершения `no_aug` отдельный контроллер `python ops/supervisor.py lr_candidate` использует GPU 1. Конфигурация данных и архитектура совпадают с основной кампанией, начальный LR нового этапа — 3e-4 вместо 1e-4, batch — 128 вместо 32, warmup — 100. Имена запусков начинаются с `lr_candidate_`, cutoff — 12:30 МСК. Это кандидат для выбора по dev, а не строго контролируемая абляция LR или batch size: фактический родительский checkpoint каждого этапа записывается в логи и может различаться. Проверка аугментаций с одинаковой инициализацией отдельно сохранена на 8920 шагах.

## Финальная оценка

`python ops/final_evaluation.py` ожидает 12:30 МСК и завершения собственных trainer-процессов. Контроллер фиксирует best-модель каждого завершённого этапа, сравнивает их по greedy WER на полном dev-clean (dev-other — tie-break), затем на выбранной неизменной модели сравнивает beam widths 10/20/50 и LM weights 0.1/0.2/0.3 с word bonus 1.0. Декодер выбирается по сумме ошибок, делённой на сумму слов обоих полных dev-наборов. Только после записи `frozen_decoder.json` оцениваются test-clean/test-other. Итоговые файлы: `/work/runs/final-evaluation/selected_model.pt`, совпадающий `selected_resume.pt`, `selection.json`, `frozen_decoder.json`, отдельные dev/test транскрипции и `complete.json`.

Snapshot и кеши записываются атомарно. Повреждённые промежуточные файлы пересоздаются; проверка resume сравнивает состояние и все тензоры с фиксированной моделью. Запуск не останавливает процессы и не меняет сервисы.

## Готовность train-clean-360 и измерение batch

Архив полностью передан, SHA-256 проверен, распаковка завершена в 05:43:45 МСК до cutoff06:00. `scripts/audit_librispeech.py` проверил 104014 FLAC-заголовков и транскрипций: все mono16kHz, пропусков/дубликатов/невозможных CTC-выравниваний нет, минимальный запас2шага. Train100:28539 записей, запас6шагов.

`scripts/benchmark_batches.py` измеряет только forward/backward/optimizer, без загрузки аудио, на фиксированном реальном train-фрагменте и временной модели, которая не используется для качества. Batch32/64/128/256:310/495/579/621 записей в секунду; peak allocated1.83/3.28/6.22/12.12GiB. Для второго кандидата выбран batch128; основной сохраняет32. При OOM явный batch128 корректно уменьшается до64 с accumulation2. Предварительный прогноз полного прохода по объединённым132553 записям — около15мин для основного и8мин для второго с запасом1.5на I/O; это прогноз, фактическое время следующего этапа будет записано отдельно.


## Неизменные исходные веса каждого этапа

Новые этапы сохраняют `initial_model-<sha256>.pt`: модель загружается из единственного чтения parent-файла, те же байты атомарно записываются в snapshot. Событие `initialize_from_model` содержит `source_sha256`, `snapshot` и фактический `source_state`. Resume восстанавливает собственный full checkpoint и не зависит от наличия исходного parent-пути. Реальная CPU-проверка прошла обучение до шага 1 и resume до шага 2 при намеренно недоступном `init_checkpoint`.

Для уже начатых main/no_aug/refine100/finetune460/lr_candidate_finetune460 исходные события не менялись. Отдельная ретроспективная проверка подтвердила точное совпадение записанного `source_state`, завершение parent-запуска и mtime его модели до инициализации дочернего этапа. Копии лежат в `initialization-evidence`; доказательства — `legacy-parents.json`.

- Staging best: шаг 2784, эпоха 86, SHA-256 `0513d0edd18ce740f0994946a76cdc8a4a93b5367c4d054869736dd5fbc94da6`.
- Main best, использованный при старте обоих этапов на 464 часах: шаг 52628, эпоха 58, SHA-256 `4f0d97726e4116286040ff8095d1c6030a1281469962c5a6544b4b16277f16be`.

Для воспроизведения второго parent-префикса сохранить `epochs=60`, добавить `max_steps=52628` к команде основного обучения и использовать полученный `last_model.pt`. Это сохраняет исходный горизонт scheduler, в отличие от уменьшения числа эпох. В итоговый комплект следует включить неизменные исходные веса выбранной цепочки отдельно от логов: обычный logs.zip намеренно исключает PT-файлы.


## Сборка архивов логов

`scripts/package_delivery.py` принимает либо `--runs <directory>`, либо `--logs-archive <captured.tar.gz>`. Второй вариант преобразует проверенный TAR в ZIP потоком, не распаковывая десятки тысяч файлов. PT/NPZ/временные файлы исключаются из logs.zip; исходные и итоговые веса сохраняются отдельно. Текстовые и медиафайлы сохраняются побайтно. Даты служебных файлов до1980 допускаются, логические UTC-времена внутри журналов не изменяются. ZIP завершается атомарно; повреждённый или небезопасный TAR не заменяет существующий готовый logs.zip.

## Chronological training curves

For phases with separate optimizer counters, render on the actual shared UTC timeline:

```sh
python scripts/build_report_pdf.py --source REPORT.md --output REPORT.pdf --metrics runs/main/metrics.jsonl runs/finetune460/metrics.jsonl runs/lr_candidate_finetune460/metrics.jsonl --time-axis elapsed
```

The origin is the earliest recorded UTC timestamp across the supplied logs. Counter resets split lines rather than connecting unrelated steps. Solid dev curves denote dev-clean; dashed curves denote dev-other. These training-monitoring curves use the logged subsets and do not replace full dev/test evaluation. The default `--time-axis step` remains available for comparable schedules such as the augmentation ablation. Report rendering uses the separate `requirements-report.txt` dependencies.

## Exact initialization archive

After training stops, collect all campaign initialization events and their immutable parent weights, deduplicated by SHA-256:

```sh
python scripts/collect_initializations.py --runs /work/runs --legacy-evidence /work/runs/initialization-evidence/legacy-parents-expanded.json --output /work/incoming/initializations-final.zip
```

The collector preserves original events, labels retrospective evidence explicitly and checks every parent hash and training state. Historical retries have separate evidence records for each actual initialization timestamp. A retry initialization event may precede loading a full resume checkpoint; it is not presented as a fresh training start. New continuation phases record their own immutable parent snapshot at initialization. Re-run collection after the final phase exits; the intermediate archive cannot contain future phases.
