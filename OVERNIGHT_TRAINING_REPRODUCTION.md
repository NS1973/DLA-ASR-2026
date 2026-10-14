# Воспроизведение ночного обучения

Это рецепт завершённой кампании и выбранной модели expanded004_mix85_intern__last: model SHA-256 5d2e8e35b01f32cd7b4b7e553de419ab796a545ef606fad994f783f35905eef9, full resume a4ca08dd25c3d4bdffb37697c41c743afb9335354bd753eaa64c5c64fe6b50cd. Обучение завершено 14 октября в 16:00 МСК: выбранный шаг 74028, epoch18, batch_index732. REPRODUCTION.md относится к историческому e05c; его компоненты и родители сохранены отдельно.

## Исходники и окружение

Фактические тренеры используют исходники commit 8b260c2f8cb25fbd73315eb428d83766c7b412e5. 14 октября проверено совпадение с текущей веткой семи файлов после нормализации CRLF/LF: model/runtime/data/codec/metrics/specaugment и ops/build_mixed_manifest.py. Проверка: logs/current-training-source-parity-20261014-1450.json. Замороженный источник Colab 08a25ef содержит тот же runtime/inference/model/decoder; последующие исправления описания цепочки не меняют веса или алгоритм.

Обучение выполнялось под Linux, PyTorch 2.11.0+cu129, две H200 NVL, по одному запуску на GPU. Установить requirements.txt; фактический GPU environment и runtime provenance приложены отдельными компонентами. CPU lock и проверенные 181 тест подтверждают переносимость кода, но не битовую идентичность CUDA на другом оборудовании. Dockerfile в корне предназначен для инференса и не объявляется исходным H200 training image.

## Данные и замороженный порядок

```bash
python ops/download_data.py --parts train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other
mkdir -p data/datasets
for archive in data/downloads/*.tar.gz; do tar -xzf "$archive" -C data/datasets; done
mkdir -p reproduction-inputs
tar -xzf full500_original_frozen_manifests.tar.gz -C reproduction-inputs
```

Внешний downloader сам вычисляет SHA полученного файла; для воспроизведения дополнительно сверить его с опубликованным manifest комплекта. Проверенный train-other-500.tar.gz: 30593501606 байт, SHA-256 ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2. Архив замороженных manifest: SHA-256 223e0acc41024adade626c392a369c01db5045a06c94f8e2df7ff06975c78f54.

Manifest пути относительны data_root и начинаются с LibriSpeech/. Замороженные файлы сохраняют порядок повторов и целевые тексты; повторно генерировать их для точного replay не требуется. Из 148688 исходных other-записей одна CTC-несовместимая 6754-107485-0012 не включена в обучение; исходное аудио сохранено. Dev/test записи и их дикторы в обучении отсутствуют по проверенному аудиту.

| Смесь | Manifest SHA-256 | Уникальные other/clean, часы | Повторы other | Экспозиции other/clean, часы | Записей с повторами |
|---|---|---|---:|---|---:|
| 75/25 | 1472bd4f6d8a72d5842c7604e1553dd8142d2f2dbb3aa45359745880cff39355 | 496.8568895 / 464.1964881 | 3 | 1490.5706685 / 496.8566480 | 587930 |
| 85/15 | 5220e84bf63582af80ea3f8a781e0a404fdd9958f03496936d05bef90f2b2784 | 496.8568895 / 464.1964881 | 6 | 2981.1413370 / 526.0834039 | 1042341 |

Доли рассчитаны по длительности экспозиций: 75.0000091% и 85.0000088%. Повторение не увеличивает уникальный объём данных. ops/build_mixed_manifest.py позволяет создать новый опыт из уникальных clean/other JSON-списков и excluded IDs; новый manifest не следует выдавать за сохранённый порядок этой кампании.

## Полный resume и новая инициализация

Точные исходные конфигурации: docs/repro/expanded004_mix75_intern.config.json и expanded004_mix85_intern.config.json. Рабочие каталоги сохраняют соответствующие full checkpoint с optimizer/scheduler/RNG/gradients и model-only. *_intern — полный resume этапа expanded004_mix75/85, а не новая инициализация оптимизатора. Для 75/25 epochs=33, для 85/15 epochs=22; это фиксированные горизонты cosine scheduler. Не заменять их историческим значением 40.

Для продолжения в новой Linux-папке подготовить соответствующий сохранённый full checkpoint и manifest. Ниже приведён полный resume выбранной модели 85/15 из resume.pt в корне комплекта. Все пути /delivery/... указывают на распакованные компоненты; каталог /delivery/new-run85 должен быть новым. Это продолжение итогового состояния, а не повтор всей истории с нуля.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python - <<'PY'
import hashlib, json
from pathlib import Path
from src.asr.runtime import train

cfg = json.loads(Path('docs/repro/expanded004_mix85_intern.config.json').read_text())
cfg['data_root'] = '/delivery/data/datasets'
cfg['training_manifest_path'] = '/delivery/reproduction-inputs/data-transitions/expanded004/mix85.json'
cfg['resume'] = '/delivery/resume.pt'
cfg['run_dir'] = '/delivery/new-run85'
cfg['init_checkpoint'] = None
cfg['end_utc'] = None
assert not Path(cfg['run_dir']).exists()
assert hashlib.file_digest(Path(cfg['training_manifest_path']).open('rb'), 'sha256').hexdigest() == cfg['training_manifest_sha256']
train(cfg)
PY
```

Здесь отключён только истёкший календарный cutoff и перенесены пути. Batch256, LR3e-5, warmup300, seed35, offset3400, аугментации, sampler, scheduler horizon и manifest сохранены. При полном resume состояние optimizer/scheduler/счётчиков/RNG загружается из собственного checkpoint. Для replay до сохранённого шага задать max_steps равным требуемому абсолютному step текущего этапа; число epochs не менять.

Начало expanded004_mix75/85 от parent_model.pt — другой режим: resume=null, init_checkpoint указывает на сохранённые родительские model-only веса, optimizer/scheduler создаются заново. Для строгого восстановления всей истории использовать исходные config.json, initialize_from_model/resume_restored события и компоненты родителей; не подменять full resume загрузкой только model-only. Закрытые worker6 логи/веса сохранены отдельно, финальные worker8 сохранены в final_worker8_all_closed_states.tar.gz; полные закрытые журналы — в final-worker8-closed-logs.zip.

## Завершённая цепочка и соответствие компонентам

`docs/repro/final-stage-chain.json` содержит все 22 завершённых запуска: точный SHA исходного config, родительские SHA/model state или full-resume state, конечные step/epoch/time и пути журналов внутри компонентов. Их исходные конфигурации без изменений доступны в `docs/repro/closed-stages/`. Исходные абсолютные пути в config — provenance сервера; при переносе изменяются только пути и истёкший календарный cutoff. Данные, scheduler horizon, seed и порядок сохраняются.

Выбранная ветвь проходит через следующие сохранённые состояния; переход model-only означает новый optimizer, full resume сохраняет optimizer/scheduler/RNG:

| Этап | Вход и режим | Состояние, использованное далее |
|---|---|---|
| e05c | Предыдущая собственная модель; historical_e05c_* рецепты/инициализации | model e05cdea1, step46179 |
| main85_packed | e05c, model-only | model8297e2d6, промежуточный last step21400 |
| expanded001_mix75 | model8297e2d6, новый optimizer | model44c02577, промежуточный last step7651 |
| expanded002_mix75 | model44c02577, новый optimizer | best da6ccc5d, step4628; full resume в _intern сохраняет этот best |
| expanded003_mix85 | modelda6ccc5d, новый optimizer | last step7479 → полный resume _intern |
| expanded003_mix85_intern | full resume 1f140def, step7479 | last step10800 → полный resume _intern_intern |
| expanded003_mix85_intern_intern | full resume6169f531, step10800 | best2813ad26, step26803 |
| expanded004_mix85 | model2813ad26, новый optimizer | lastbc1b302b, step9186; fulle4f1f114 |
| expanded004_mix85_intern | fulle4f1f114, сохранённый optimizer/scheduler/RNG | selected last5d2e8e35/full a4ca08dd, step74028 |

Родительские модели и full состояния до expanded003 находятся в parent_chain_through0607.zip; исходные configs там под configs/, модели — content-addressed по SHA. Родитель2813ad26 и его full сохранены в stage004_parent_delta.zip. Закрытые промежуточные full состояния/журналы — closed400_full_states_and_logs.tar.gz и closed_worker6_predecessor_905/906.tar.gz. Финальные четыре model/full пары — final_worker8_all_closed_states.tar.gz. Все названия относятся к файлам внутри полного reproduction архива; их SHA перечислены в manifest.json. Промежуточный parent может быть раньше terminal step соответствующего запуска: не заменять его последним checkpoint.

Для replay отдельного исходного этапа взять его config из docs/repro/closed-stages, соответствующий frozen manifest и точный parent из chain JSON. Для model-only оставить resume=null и указать init_checkpoint на этот parent; для full-resume указать сохранённый полный checkpoint, а не model-only. Абсолютный max_steps задаёт остановку на сохранённом шаге; epochs остаётся исходным горизонтом scheduler. Побитовая идентичность обучения на другом оборудовании или другой версии CUDA не обещается. Для восстановления уже достигнутого результата использовать приложенные model.pt/resume.pt; их совпадение тензоров и полного состояния проверено.

## Выполненная итоговая оценка

Обучение обоих финальных запусков завершилось штатно: 75/25 step75013 в16:00:07 МСК, 85/15 step74028 в16:00:08. Рассмотрены 112 уникальных SHA кандидатов на полных dev-clean/dev-other при BF16 batch16. Выбор — минимум dev-other при dev-clean≤20%; затем 13 настроек decoder на dev. Зафиксирован beam50/token_topk8/LM0.1/word_bonus1, после freeze выполнены полные test-clean/test-other. Test не выбирал модель или decoder.

Полные WER: dev-clean12.2091099592%, dev-other28.1993405040%, test-clean12.1272063299%, test-other29.3811971037%. Счётчики записей/слов/ошибок — FINAL_RESULTS.json; все предсказания и сравнения — evaluation20261014.tar.gz. BF16 dev512 monitor не заменяет эти полные оценки. Статус проверки сдаваемого Colab и комплекта находится в ACCEPTANCE.md; прежние proofs сохраняются отдельно.
