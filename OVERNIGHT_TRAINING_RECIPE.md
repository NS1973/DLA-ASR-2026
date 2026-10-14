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
