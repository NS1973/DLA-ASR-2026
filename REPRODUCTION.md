# Воспроизведение обучения

Окончательная модель выбрана по полным dev-clean/dev-other: lr_candidate_finetune460_continue2__last, шаг 31717 текущего этапа. Ниже приведены её собственная цепочка родителей и фактические горизонты scheduler. Все команды выполняются из корня проекта после установки requirements.txt. Для нового запуска отключается календарный cutoff через end_utc=null; каталог каждого запуска должен быть новым.

## 1. Данные

```bash
python ops/download_data.py --parts train-clean-100 train-clean-360 dev-clean dev-other test-clean test-other
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

## 4. Выбранная цепочка на 464 часах

train-clean-360 готов и проверен в 05:43:45 МСК, до cutoff 06:00. Вместе с train-clean-100 он даёт 132553 записи, 464,20 часа. SHA-256 официального train-clean-360.tar.gz: 146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf.

В окончательный архив включён initializations.zip с пятью неизменяемыми собственными родительскими моделями и исходными событиями инициализации. Распаковка:

```bash
python -m zipfile -e initializations.zip initializations
```

Выбранная цепочка:

| Родитель | Шаг | SHA-256 |
|---|---:|---|
| staging, собственная инициализация из случайных весов | 2784 | 0513d0edd18ce740f0994946a76cdc8a4a93b5367c4d054869736dd5fbc94da6 |
| main на train-clean-100 | 52628 | 4f0d97726e4116286040ff8095d1c6030a1281469962c5a6544b4b16277f16be |
| первый batch-128 этап на train100+360 | 41440 | a51d1268802190790766736a6e97f8f23a17de78787dec8e7d349f8a21204fa5 |
| продолжение 1, лучший сохранённый родитель | 39368 | a475b472868aca39556102640143a5278ab1cd9d2fb65efd59c2bd81f6fcc553 |

Пятый родитель 6066bd6b382a8d1391a4669f188f64b623266c4b7806742f82dc34d2db3e7678, шаг 36260 продолжения 1, использовался только отдельным кандидатом effective_v2; он тоже сохранён.

Для воспроизведения условий каждого этапа используйте соответствующего исходного родителя. Это исключает накопление различий предыдущего повторного обучения:

```bash
python train.py data_root=data/datasets train_parts=[train-clean-100,train-clean-360] init_checkpoint=initializations/4f0d97726e4116286040ff8095d1c6030a1281469962c5a6544b4b16277f16be.pt run_dir=runs/repro-460-first batch_size=128 epochs=40 lr=0.0003 warmup_steps=100 augmentation_profile=legacy data_epoch_offset=0 max_steps=41440 end_utc=null
python train.py data_root=data/datasets train_parts=[train-clean-100,train-clean-360] init_checkpoint=initializations/a51d1268802190790766736a6e97f8f23a17de78787dec8e7d349f8a21204fa5.pt run_dir=runs/repro-460-continue1 batch_size=128 epochs=40 lr=0.00005 warmup_steps=100 augmentation_profile=legacy data_epoch_offset=0 max_steps=39368 end_utc=null
python train.py data_root=data/datasets train_parts=[train-clean-100,train-clean-360] init_checkpoint=initializations/a475b472868aca39556102640143a5278ab1cd9d2fb65efd59c2bd81f6fcc553.pt run_dir=runs/repro-selected-prefix batch_size=128 epochs=40 lr=0.00005 warmup_steps=100 augmentation_profile=legacy data_epoch_offset=0 max_steps=31717 end_utc=null
```

epochs=40 сохраняет фактический cosine horizon 41440 шагов; max_steps задаёт префикс, а не новый горизонт scheduler. Продолжение 1 в исходной кампании завершило все 41440 шагов; следующий этап был инициализирован его best на шаге 39368. Последний выбранный этап штатно остановлен по времени на шаге 31717, epoch 30, batch_index 637. Это 30 полных эпох и часть 31-й.

Для непрерывной новой цепочки от случайной инициализации выполните staging-префикс из раздела 2, затем main-префикс max_steps=52628 при epochs=60; в следующих командах заменяйте ссылки initializations/*.pt на last_model.pt предыдущего воспроизведённого префикса. Побитная идентичность исторического обучения на другой GPU/версии библиотек не обещается. Предоставленные исходные родители и финальные веса позволяют точно проверить сохранённые состояния и повторить инференс.

Возобновление выбранного доверенного full checkpoint не зависит от наличия исходного пути родителя:

```bash
python train.py resume=resume.pt data_root=data/datasets train_parts=[train-clean-100,train-clean-360] run_dir=runs/resume-selected batch_size=128 accumulation=1 epochs=40 lr=0.00005 warmup_steps=100 augmentation_profile=legacy data_epoch_offset=0 end_utc=null
```

Проверка именно сдаваемого full checkpoint подтвердила точное восстановление параметров, AdamW, scheduler, Python/NumPy/Torch/CUDA RNG и градиентов. Диагностический шаг на четырёх коротких train-записях выполнен в FP32 на CPU только в памяти: loss 0,00465095, норма градиента 0,350478. Сдаваемые файлы не изменялись. Это проверка восстановления, не новое обучение модели для выбора.

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

## Подтверждённое продолжение кандидата после 40 эпох

Первый этап `lr_candidate_finetune460` завершил 40 полных эпох по 132553 записям (41440 optimizer steps) 13 октября в08:19:38МСК. Его последний checkpoint оказался лучшим по контрольному dev-clean: WER14.9056%, dev-other40.8360%, по256 seed1 записям. Это мониторинговые поднаборы, а не итоговая оценка.

Следующий этап автоматически стартовал с точных собственных весов SHA-256 `a51d1268802190790766736a6e97f8f23a17de78787dec8e7d349f8a21204fa5` (parent step41440/epoch39). Batch128, новый AdamW/LR5e-5, warmup100, cosine horizon40эпох. Пример отдельного воспроизведения после распаковки архива родителей:

```bash
python train.py data_root=data/datasets train_parts=[train-clean-100,train-clean-360] init_checkpoint=initializations/a51d1268802190790766736a6e97f8f23a17de78787dec8e7d349f8a21204fa5.pt run_dir=runs/repro-candidate-continue1 batch_size=128 epochs=40 lr=0.00005 warmup_steps=100 end_utc=null
```

Продолжение 1 завершило все 40 эпох; выбранным родителем продолжения 2 стал его best на шаге 39368. Для воспроизведения этого префикса сохраняйте epochs=40 и задавайте max_steps=39368. Локальный резерв завершённого предыдущего этапа содержит model-only и совпадающий по всем тензорам/state full resume, SHA-256 full resume `8a3f536cfaba13528144f9ebf92dbfe6cf0da98edb52ce731e47b8405d53bf88`.

## Исправление аугментаций после аудита

Исторические запуски используют `augmentation_profile=legacy`, значение по умолчанию сохранено. Постоянный gain почти полностью исчезал после нормализации log-Mel; прежняя reverb имела разностный SNR40–46dB. Эти ограничения не скрываются и не переименовываются в сильные аугментации.

Отдельный кандидат `lr_candidate_augmented460` использует `augmentation_profile=effective_v2`: плавную огибающую громкости с5опорными точками0.65–1.35 и реверберацию с RMS отражённой компоненты10–20% от исходного сигнала. Noise/speed и CTC-защита длины сохранены. На4реальных trainFLAC×3seed измерено изменение нормализованных признаков: gain0.0307–0.0988RMS, reverb0.0744–0.1617RMS. Это проверка преобразований, а не доказательство улучшенияWER.

Новый кандидат: batch128, LR5e-5, warmup100,40эпоховый cosine horizon, те же132553 train-записи и dev256seed1. `data_epoch_offset=80` меняет порядок train и seed аугментаций, сохраняя фиксированный dev-поднабор. Точный parentSHA/state сохраняется автоматически. `python ops/augmentation_proof.py --profile effective_v2` создаёт отдельный offlineW&B run с аудио/спектрограммами. `python ops/augmented_experiment.py` ограничивает запуск12:30МСК и использует GPU0 параллельно сохранённому baseline; не считать такое сравнение изолированным влиянием одного gain или reverb. Итоговый выбор остаётся по dev.

## Финальное правило выбора после аудита

До первого запуска test-оценки правило уточнено: среди замороженных best и last checkpoint каждого основного эксперимента выбрать минимальный **pooled greedy WER полного dev-clean+dev-other**: сумма word_errors обоих наборов / сумма reference words. При равенстве — dev-clean, затем dev-other, затем имя запуска. Это заменяет прежний приоритет только dev-clean; фиксированное правило не использует test и лучше учитывает целевой test-other. Сначала выбирается checkpoint, затем настройки декодера по обоим полным dev, затем единственная итоговая оценка test. Исторические мониторинговые best внутри отдельных запусков по-прежнему сохранялись по dev-clean256; утверждения о глобальном поиске среди всех промежуточных эпох не делаются.
