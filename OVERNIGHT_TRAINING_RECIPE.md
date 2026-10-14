# Воспроизведение ночного обучения

Это рецепт фактически выполненной кампании; окончательные выбранные веса и последующие этапы добавляются после завершения обучения. Исходные конфигурации, манифесты и full checkpoint обязательны: итоговые model-only веса не восстанавливают AdamW, scheduler и RNG.

## Данные

Использовать официальные LibriSpeech train-clean-100/360 и проверенные части train-other-500. Список конкретных UID, относительных FLAC-путей, текстов, длительностей и SHA брать из сохранённых манифестов соответствующего этапа. Порядок частей train-other — порядок полных глав исходного TAR, а не случайная выборка. В полном train-other из обучения исключена CTC-несовместимая запись 6754-107485-0012; исходное аудио сохранено. Dev/test UID не входят в обучение.

Полный исходный train-other-500: 30593501606 байт, MD5 d1a0fd59409feb2c614ce4d30c387708, SHA-256 ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2. Наличие этого архива локально само по себе не доказывает передачу полного набора на сервер.

## Построение фиксированной смеси

ops/build_mixed_manifest.py принимает JSON-массивы записей clean/other и excluded UID. Доли задаются по длительности экспозиции. Все исходные clean-записи сохранены; other повторяется целое число раз, затем clean replay добавляется до ближайшей достижимой длительности. Сначала перемешивается clean replay, затем весь список одним random.Random(seed). Повторения не увеличивают уникальные часы.

```sh
python -m ops.build_mixed_manifest --clean clean-unique.json --other other-unique.json --excluded-ids excluded-ids.json --fraction 0.75 --seed 34 --output mix75.json
python -m ops.build_mixed_manifest --clean clean-unique.json --other other-unique.json --excluded-ids excluded-ids.json --fraction 0.85 --seed 34 --output mix85.json
```

Команды выше соответствуют seed этапа expanded003. Для другого этапа использовать его сохранённый seed, а не автоматически34. Output должен быть новым файлом; для точного повтора проверять SHA исходного frozen manifest. data_root можно переносить, относительные пути и содержимое манифеста менять нельзя.

## Запуск и продолжение

Использовать исходный Git commit, записанный в конфигурации/операторском process.json, и зависимости проекта. Скопировать config.json выбранного этапа в новый launch_config.json, адаптировать только пути run_dir/data_root/training_manifest_path/init_checkpoint/resume к своей машине. end_utc при повторе вне кампании заменить на null либо новый явный срок; исторический cutoff 2026-10-14T13:00:00+00:00 уже истечёт. Это адаптация нового запуска, а не изменение сохранённого исходного опыта.

```sh
python -c 'import json,sys; from src.asr.runtime import train; train(json.load(open(sys.argv[1])))' launch_config.json
```

Новый этап от parent model-only использует init_checkpoint и resume=null, поэтому создаёт новый optimizer/scheduler. Продолжение прерванного этапа использует собственный full checkpoint через resume и исходные seed/batch/LR/scheduler/data_epoch_offset/manifest SHA. Не выдавать продолжение за независимый эксперимент. Проверить событие resume_restored и совпадение step/batch_index с full payload.

Основной этап expanded003: 400.65833953125 уникальных other-часа +464.1964880729167 clean; batch256, LR3e-5, warmup300, seed34, offset3300, effective_v2 waveform augmentations, packed stack, mild SpecAugment, BF16 и FP32 CTC, grad clip5. Переносы на source6ba17e1/8b260c2 восстанавливали собственное полное состояние. Source8b: training workers8, validation workers0, prefetch1, skip_resumed_audio=true; будущий полный500-этап планирует workers6, его фактический запуск ещё нужно подтвердить.

При inherited best checkpoint конфигурация внутри full payload может относиться к исходному родительскому этапу; название каталога-алиаса не заменяет эту конфигурацию. Цепочку устанавливать через parent SHA и exact resume edges. Компонент overnight-parents-through0607 содержит18конфигураций/6resume edges; он не содержит автоматически будущие этапы.

## Проверка результата

Сохранять full/model matching, optimizer/scheduler/RNG, manifests, configs, локальные JSONL и offline W&B/media. Архивы E неизменяемы. После16:00 единым протоколом переоценить все SHA-различные best/last на полном dev, выбрать минимум dev-other при dev-clean<=20%, подобрать decoder только на dev, зафиксировать выбор, затем полный test. Новые веса требуют нового свежего Colab и нового комплекта приёмки; старый проверенный комплект остаётся резервом.

### Full train-other stage004, 14 October 07:20 MSK
Both new processes launched after prior400h actualtrain_exit and matching reserves: GPU0PID38827 mix75, GPU1PID38828 mix85. Parent2813ad260c7efbeed187f6c718b87408c3fbc2e06120419a16d15fd4f76d271e was chosen by full greedy dev-clean14.02154333%/dev-other31.35157415%; matching fullSHAc884a4515a987929363a41fb3e39c77457d11fbf53db54fe84249457405d6258. Newstage: source8b260c2, batch256/workers6/LR3e-5/warmup300/seed35/offset3400/cutoff14Oct13UTC; new optimizer/scheduler. Dataset496.8568895uniqueotherh+464.196488clean, fractions.750000091/.850000088, otherrepeats3/6. First75step1finite loss.340431/gradient.411292;85firststep confirmation pending at this observation. This replaces the earlier statement that full500is still transferring. Exact new frozen manifests and finalclosed400logs require capture in addition to earlier components.


### Full500 stage004, продолжения workers6→8, подтверждено 14 октября

Оба первоначальных процесса38827/38828 штатно закрыты после проверенных резервов; фактические продолжения40932/41699 используют expanded004_mix75_intern и expanded004_mix85_intern. Это те же эксперименты с собственным full resume, а не повторная инициализация. Batch256, LR3e-5, warmup300, seed35, offset3400, scheduler, данные и cutoff13:00UTC сохранены; изменились только каталог, resume и число workers8. Source8b260c2, validation_workers0, prefetch1, skip_resumed_audio=true. Предшественник75 закрыт на шаге6459, resume SHA00f27f96c6c16e2db9cc72660148664f489ec49bec2a2287a58b42fdaa9b3b3b;85 —9186, resume SHAe4f1f1144029cac9c722f54b5671057e568ef70a15a1e45adb5fd2f17668d167. Окончательные веса и полные логи этих закрытых запусков сохранены в компонентах905/906; эти номера являются метками архивов, индекс данных остаётся004.

Точная последовательность исходных конфигураций, событий initialize_from_model/resume_restored и SHA родителей проверена в overnight-parent-chain-audit-20261014-100508.json:22 запуска,8 full resume edges, missing=[]. Оригинальные массивы записей нужны из frozen-training-inputs-stage004-20261014-072227.tar.gz; их нельзя заменять повторной случайной выборкой. Исходный набор other содержит496.8568895ч после исключения одной CTC-невыполнимой записи; повторения3/6 дают фактические доли75,0000091%/85,0000088% по длительности. Все464.196488ч clean доступны для replay. Для финальной выбранной модели потребуется обновить аудит после train_exit и приложить её фактическую цепочку, включая родительский e05c-комплект.


Текущий scheduler подтверждён чтением живых config.json 14 октября10:47: epochs33 для75/25 и22 для85/15, соответственно75801 и89584 optimizer steps при accumulation1. Не подставлять историческое предположение40эпох; использовать сохранённую конфигурацию каждого этапа. Наблюдаемые LR2.16418e-5/2.40716e-5 на шагах28660/28020.
