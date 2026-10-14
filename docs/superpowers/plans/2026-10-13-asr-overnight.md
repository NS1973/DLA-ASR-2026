# ASR Overnight Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans for inline implementation. Do not spawn additional agents without explicit authorization.

**Goal:** завершить обучение14октября2026 в16:00МСК, максимально улучшив полный dev-other и полезный throughput без потери проверенных результатов; затем провести полную приёмку нового комплекта.

**Architecture:** сохранить совместимую собственную DS2-CTC модель и начинать новые этапы с проверенного e05c. Увеличить уникальные other-данные до доступной проверенной части полного train-other-500; сравнить other-heavy смеси с clean replay. Оптимизации включаются только после измерения скорости и проверки корректности.

**Tech Stack:** существующие PyTorch2.11/CUDA12.9, Hydra, две H200, собственный CTC/beam/ARPA decoder, offline W&B/JSONL, SSH с проверкой SHA, проектные контейнеры, E-backups.

**Spec:** ../specs/2026-10-13-asr-overnight.md

## Global Constraints

- Обучение заканчивается14октября16:00МСК, cutoffUTC2026-10-14T13:00:00Z. Проверки и сдача после этого времени.
- Выпуск56c3158, его архив/SHA, e05c/fullresume и все старые комплекты сохраняются неизменными.
- Новая ветка codex/hw1-asr-overnight-20261014; новые run_dir, snapshot и evaluation root. Старые evaluator roots не менять.
- Только собственные модельные родители; один активный trainer на GPU. Новые конфигурации не менять посреди run.
- Test не используется для выбора. Полный dev-other primary, полный dev-clean<=20% gate.
- /home>=20GiB, root>=1GiB, MemAvailable>=24GiB; изменения чужих сервисов/Xorg запрещены. Apple/прочие E-пути не трогать.
- Не обещать ускорение, качество или отсутствие сбоев до измерения; наблюдательный timeout не означает остановку процесса.

## Review Focus

1. Доля other считается по исходной длительности, а не числу файлов; повторы не выдаются за уникальные часы.
2. Packed GRU, BF16 и SpecAugment не нарушают CTC-длины, padding или совместимость сохранённых весов.
3. Fixed manifest, sampler order, augmentation RNG и scheduler воспроизводятся после resume.
4. Данные/weights/checkpoints в RAM имеют внешние резервы; новый tmpfs не вытесняет память чужих сервисов.
5. Новые метрики имеют полный набор, SHA модели, frontend/decoder/LM signature; прежние test результаты не влияют на выбор.

## Расписание МСК

| Время | Результат |
|---|---|
| Сейчас–22:20 | Сохранение резерва, профиль, подготовка other-heavy manifest и первые обучающие этапы; сразу начинается догрузка данных |
| 22:20–23:45 | Короткое сравнение mild SpecAugment против того же75/25 без него; одинаковый parent/seed/data и общий проверенный fast path |
| 23:45–06:00 | Два основных кандидата75/25 и85/15; новые уникальные other-шарды подключаются только отдельным проверенным этапом |
| 06:00–12:30 | Длительное обучение, полные dev каждые60–90мин; неудачный вариант заменяется ограниченным опытом без потери его результатов |
| 12:30–15:00 | Подтверждение лучших кандидатов/стабильности, подготовка отчёта и приёмки; прекратить новые инженерные эксперименты |
| 15:00–16:00 | Обучение фиксированных оставшихся этапов, резервирование/подготовка точного завершения; код/данные/гиперпараметры не менять |
| После16:00 | Actualtrain_exit → закрытые резервы → полныйdev/decoderfreeze → полныйtest → новыйfreshColab → отчёт/bundle/archive/SHA |

Время подготовки ограничено: не тратить ночь на инфраструктуру или переписывание архитектуры. Если fast path не готов, запускать проверенный legacy pipeline; если500h ещё передаются, учить на уже проверенных100h. Не выдавать это за полный500h опыт.

### Task1: Сохранность, ресурсы и baseline профилирования

**Files:** ops/profile_training.py (new), logs/new campaign evidence; old delivery files read-only.

- [ ] Подтвердить E-копии e05c/model-full, старого release archive и соответствующие SHA; завершить уже начатую privateDrive передачу. Новые исходники держать приватными до новой сдачи.
- [ ] Проверить реальные GPU/process/CPU/memory/disk ресурсы, Kafka register/poll/claims; создать только собственное рабочее пространство.
- [ ] Профиль:20warmup+100–200representative batches с одинаковым parent; optimizer probes только в RAM. Измерить audio_seconds/wall_second, records/s, median/p95 step, data wait, feature extraction, transfer, forward/backward/optimizer, padding и peakVRAM. Учитывать реальный augmentation и logging/checkpoint overhead.
- [ ] Проверить фактические dtype/cuDNN kernels. Сравнить B128/256/512, workers4/8/12 и prefetch в доступном CPU бюджете, без вытеснения соседей. Выбирать самый быстрый устойчивый вариант по аудиочасам, не steps/s.
- [ ] Зафиксировать baseline и итог benchmark JSON с командами/конфигом. VRAM reserve15–20% для dev/атомарных операций.

### Task2: Больше уникальных other и фиксированные смеси

**Files:** ops/build_mixed_manifest.py (new); src/asr/runtime.py (opt-in training_manifest_path); tests/test_mixed_manifest.py (new); source/data manifests on E and project scratch.

- [ ] Сначала тесты: duration-ratio, intentional repeat UID versus уникальные часы, deterministic generation, validation/test UID exclusion и восстановление того же manifest SHA.
- [ ] Реализовать минимальный opt-in input manifest; существующий train_parts/default/LengthBatchSampler не менять. Saved cfg содержит manifest identity и полный рецепт. Пути нового manifest разрешать относительно data_root; тест переноса данных/возобновления не должен зависеть от /scratch конкретного сервера.
- [ ] GPU0/GPU1 получают75%other/25%clean и85%other/15%clean по длительности (точность до одной записи). Использовать все доступные чистые данные как replay-источник; уменьшить их долю экспозиции. Заморозить manifest до запуска.
- [ ] Практичный builder: повторить весьother минимальное целое число раз, чтобы целевая clean-экспозиция включала все464h хотя бы раз; остаток clean добрать детерминированным replay. Не вводить новый сложный online sampler, пока обычный фиксированный manifest решает задачу.
- [ ] На нынешних100h это примерно14other-cycles для75% и27для85%; фактические длительности/повторы считать из manifest. На полномother500 потребуются другие количества. Эти повторы не добавляют уникальные часы.
- [ ] Скачать официальный train-other-500 на E, сохранив прежние prefix/repair файлы; resume только при совпадении идентичности источника. Проверить официальный checksum полного исходника и собственныйSHA-256, tar/FLAC/transcript integrity, CTC feasibility и UID disjointness.
- [ ] Передавать проверенные неподвижные шарды по SSH с SHA/resume. Можно обучать новый отдельный этап на готовых200/300h раньше завершения передачи500h; назвать точный объём/состав и сохранить provenance. Не менять данные активного trainer.
- [ ] Для полного набора выделить отдельный проектный tmpfs ориентировочно48GiB с контейнерным RAM limit, только если MemAvailable после фактического наполнения>=24GiB. Существующий почти полный /dev/shm не расширять и чужие файлы не трогать. Не хранить на сервере одновременно полный compressed archive и extracted copy.
- [ ] Прогнозировать ETA по измеренной передаче; сначала полезные проверенные данные, а не ждать до конца всей загрузки. Ошибка данных не останавливает обучение на предыдущем проверенном наборе.

### Task3: Ускорение существующего pipeline

**Files:** src/asr/model.py, src/asr/runtime.py, src/asr/data.py; tests/test_packed_stack.py and existing checkpoint/padding regressions.

- [ ] RED: mixed-length eval forward, padding invariance, short inputs, gradients, legacy checkpoint load and exact within-mode resume.
- [ ] Прототип training-only packed-stack: pack один раз, между GRU применять LayerNorm/dropout к PackedSequence.data, unpack один раз перед CTC. Сохранить parameter names/model_spec и legacy inference. Dropout RNG trajectory может измениться; это НОВЫЙ этап, а не обещание битовой идентичности старому.
- [ ] GREEN по FP32/eval tolerances и BF16 finite gradients, затем одинаковый warmed-up benchmark. Продвигать при полезном end-to-end выигрыше>=15% либо существенном увеличении допустимого batch; иначе оставить legacy.
- [ ] Если профиль показывает CPU/feature bottleneck, сначала workers/prefetch/длительные buckets и vectorized CTC-required lengths. Dynamic frame-budget batch только при измеренном выигрыше и тестах coverage/resume; не терять длинные записи ради красивого throughput.
- [ ] Fused AdamW — измеренная опция после проверки установленного PyTorch и checkpoint roundtrip. torch.compile/перенос Mel наGPU — только ограниченный дополнительный probe; отказаться через15–30мин без явного выигрыша. Не переписывать frontend ради гипотетического ускорения.
- [ ] Все полезные изменения опциональны; повторить полный relevant test suite. Не задерживать длинное обучение ради неподтверждённой оптимизации.

### Task4: Mild SpecAugment и запуск двух кандидатов

**Files:** src/asr/specaugment.py (new); src/asr/runtime.py opt-in; tests/test_specaugment.py; new immutable run configs.

- [ ] Тесты: transform только valid frames/train mode; lengths/targets/padding неизменны; короткая запись не полностью замаскирована; RNG restore; inference и legacy unchanged.
- [ ] Начальный mild policy: до2frequency masks шириной8 из80полос и до2time masks ширинойmin(40frames,5% valid length), безtime warp; mask value0 после нормализации. Четыре обязательные waveform augmentations сохраняются.
- [ ] Короткое сравнение75/25 без/соSpecAugment из одного e05c, одинаковые seed10/order/optimizer/batch/fast path. НачальныйLR3e-5, warmup100–300 по результатам stability probe; зафиксировать до запуска. Если меняются несколько факторов, честно назвать comparison конфундированным, а не чистой абляцией.
- [ ] Новый optimizer/scheduler при init нового этапа; horizon/total_steps рассчитывать и фиксировать по фактической скорости и плановому бюджету. Не менять scheduler/seed/LR/batch/data на ходу. Resume — только собственный full checkpoint с полной cfg.
- [ ] После короткого полногоdev выбрать augmentation policy. Основные две GPU:75/25 и85/15, общий проверенный pipeline. При поступлении новых данных штатно закрыть свой этап после свежего резерва и начать новый stage с выбранным проверенным parent; результаты старого этапа сохраняются.

### Task5: Overnight контроль и ограниченные решения

**Files:** new campaign controllers/watchers; ops/final_evaluation.py only isolated roots/signatures; report inputs/Obsidian.

- [ ] Мониторить реальные PID/command/steps/finite loss/gradients/throughput/VRAM/space и свежесть checkpoints. Базовый reference e05c и все прежние кандидаты остаются в отборе.
- [ ] Быстрый фиксированный dev512 — каждые20–30мин; полныйdev-clean/dev-other — async frozen snapshots каждые60–90мин. Выбор только по полному dev; доля clean<=20% обязательна, early alarm clean>18% или деградация>1.5п.п.
- [ ] Если85% other ухудшаетclean или проигрываетother, сохранить этап и запустить более безопасный75% replay; не уничтожать достигнутое. Еслиother не улучшается>=0.3п.п. на двух полных проверках за>=2h, на одной GPU отдельный lower-LR stage от best; вторая продолжает полезный run.
- [ ] Checkpoint в собственном scratch<=5мин; model+full согласованная пара. E full-resume backups ориентир<=15–20мин, новые best приоритетно. При передаче500h учитывать общий измеренный SSH bandwidth: внешнийbackup получает приоритет, данные дозагружаются оставшейся полосой. Не обещать невозможные полные двух-GPU backups каждые5мин на маршруте2MB/s.
- [ ] Использовать content-addressed сохранение неизменившихся parents/best; пропускать устаревшую очередь наблюдения, но не удалять прежние резервы. Перед штатным закрытием — свежий verified reserve; послеactualexit отдельно сохранить закрытые states/logs.
- [ ] OOM/NaN: сохранить ошибку, свойlastfull, новый корректный recovery stage/проверенныйresume. Timeout наблюдения: повторно проверить тот жеPID, не перезапускать обучение. ПерегрузкаCodex/W&B не влияет на detached trainers.
- [ ] CPU/RAM/network/GPU reserves и Kafka claims обновлять; Obsidian own note checkpoint. CLI hang — bounded exact-owned helper termination и direct-note fallback, не остановка проекта.

### Task6: Фиксация14октября16:00 и сдача после проверок

**Files:** separate overnight evaluation root, frozen results/resources, final notebook/report, new delivery directory/bundle/archive; scripts/package_delivery.py source-ref option; tests/test_delivery_source_ref.py; existing baseline immutable.

- [ ] До15:00 подготовить cutoff/watchers с observation grace ПОСЛЕ16:00, проверить восстановление и queue; после15:00 не менять pipeline или данные текущих run.
- [ ] В16:00 прекратить новые optimizer steps штатным cutoff, дождаться final save и фактическогоtrain_exit. Не делать GPUreset/kill чужих процессов; последнее закрытое состояние сохранитьE.
- [ ] Best/last всех старых/новых stages, SHA dedup, fulldev-other primary clean<=20%. Cache key включает weights SHA, frontend/vocab/normalization/decoder/LM/dev identity; старую метрику не переиспользовать после изменения inference semantics.
- [ ] Подобрать greedy/ownbeam/LM по обоим full-dev; базовая13grid, при наличии выигрыша ограниченное уточнение LM0.05/0.1/0.15 иtoken_topk16. Лимит45мин, затем decoderfreeze. Test только послеfreeze, никаких настроек поtest.
- [ ] Проверить selected fullresume/parentchain, анонимныеURL/SHA; НОВЫЙ freshColab дляновых весов с/безGT; полныйотчёт с actualmetrics/experiments/limits, PDF allpagesvisual.
- [ ] Новый отдельный bundle/archive+manifest+SHA+exactrestore; сохранить все промежуточные результаты. Ночная ветка отличается от codex/hw1-asr: убрать hardcoded bundle ref, явно задавать новый source-ref и проверять, что ZIP/manifest/bundle восстанавливают один и тот же frozenHEAD. Тестировать упаковку на non-default branch. Если новая модель не улучшилась/не прошла приёмку, сдавать прежний проверенный комплект, новые эксперименты приложить отдельно.
- [ ] Репозиторий/доставка: текущая GitHub Confirm access не препятствует локальному Git и разрешённому bundle fallback. Новые исходники не публиковать до нового срока. Не заявлятьanonymousclone безactualverification. Goal complete только после обязательной приёмки; heartbeat удалить после фактической сдачи, не по часам16:00.

## Источники технических решений

- OpenSLR LibriSpeech: https://www.openslr.org/12/
- PyTorch tuning/data loading: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- SpecAugment original paper: https://arxiv.org/abs/1904.08779

Эти источники обосновывают направления; конкретный выигрыш на нашей DS2 и установленном PyTorch доказывается собственным измерением.


### Approved RAM reserve update, 14 October 2026
Direct user approval lowers host MemAvailable reserve from32GiB to24GiB. Container cap100GiB/internal headroom8GiB, scratch5GiB, /home20GiB and root1GiB remain unchanged. Preflight uses24.5GiB host/9GiB container startup allowance. Receiver SHA c8a99fa56599afde3e80539c9d185d7bc399904bd54da95152dc2268c85549af readback verified; transfer resumed automatically without trainer signals. Historical32GiB observations remain historical evidence.
