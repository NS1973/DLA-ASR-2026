# Воспроизведение ночного обучения

Это рецепт двух текущих этапов, а не заявление о выбранной финальной модели. Итоговая цепочка родителей и число шагов добавляются после завершения обучения и dev-отбора. REPRODUCTION.md пока описывает исторический комплект e05c.

## Исходники и окружение

Фактические тренеры используют исходники commit 8b260c2f8cb25fbd73315eb428d83766c7b412e5. 14 октября проверено совпадение с текущей веткой семи файлов после нормализации CRLF/LF: model/runtime/data/codec/metrics/specaugment и ops/build_mixed_manifest.py. Проверка: logs/current-training-source-parity-20261014-1450.json. Новые исходники Colab 959ca7c содержат тот же код.

Обучение выполнялось под Linux, PyTorch 2.11.0+cu129, две H200 NVL, по одному запуску на GPU. Установить requirements.txt; фактический GPU environment и runtime provenance приложены отдельными компонентами. CPU lock и проверенные 172 теста подтверждают переносимость кода, но не битовую идентичность CUDA на другом оборудовании. Dockerfile в корне предназначен для инференса и не объявляется исходным H200 training image.

## Данные и замороженный порядок

```bash
python ops/download_data.py --parts train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other
mkdir -p data/datasets
for archive in data/downloads/*.tar.gz; do tar -xzf "$archive" -C data/datasets; done
mkdir -p reproduction-inputs
tar -xzf frozen-training-inputs-stage004-20261014-072227.tar.gz -C reproduction-inputs
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

Для продолжения в новой Linux-папке подготовить соответствующий сохранённый full checkpoint и manifest. Пример для 75/25; для 85/15 заменить config и mix75.json на mix85.json. Все пути /delivery/... ниже указывают на распакованные собственные компоненты; каталог /delivery/new-run75 должен быть новым.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python - <<'PY'
import hashlib, json
from pathlib import Path
from src.asr.runtime import train

cfg = json.loads(Path('docs/repro/expanded004_mix75_intern.config.json').read_text())
cfg['data_root'] = '/delivery/data/datasets'
cfg['training_manifest_path'] = '/delivery/reproduction-inputs/data-transitions/expanded004/mix75.json'
cfg['resume'] = '/delivery/checkpoints/expanded004_mix75/last.pt'
cfg['run_dir'] = '/delivery/new-run75'
cfg['init_checkpoint'] = None
cfg['end_utc'] = None
assert not Path(cfg['run_dir']).exists()
assert hashlib.file_digest(Path(cfg['training_manifest_path']).open('rb'), 'sha256').hexdigest() == cfg['training_manifest_sha256']
train(cfg)
PY
```

Здесь отключён только истёкший календарный cutoff и перенесены пути. Batch256, LR3e-5, warmup300, seed35, offset3400, аугментации, sampler, scheduler horizon и manifest сохранены. При полном resume состояние optimizer/scheduler/счётчиков/RNG загружается из собственного checkpoint. Для replay до сохранённого шага задать max_steps равным требуемому абсолютному step текущего этапа; число epochs не менять.

Начало expanded004_mix75/85 от parent_model.pt — другой режим: resume=null, init_checkpoint указывает на сохранённые родительские model-only веса, optimizer/scheduler создаются заново. Для строгого восстановления всей истории использовать исходные config.json, initialize_from_model/resume_restored события и компоненты родителей; не подменять full resume загрузкой только model-only. Закрытые worker6 логи/веса сохранены отдельно, финальные worker8 сохраняются после actual train_exit.

## Итоговая оценка

После cutoff сравниваются сохранённые best/last старых и новых запусков и промежуточные SHA-дедуплицированные кандидаты. Выбор по полному dev-other при полном dev-clean WER ≤20%; одинаковый final BF16 batch16. Затем сетка собственных beam/LM, фиксация decoder и только после неё полные test-clean/test-other. Промежуточный монитор BF16 batch128 и dev512 не заменяют эту оценку. Новая свежая проверка Colab и окончательный архив обязательны для выбранных весов.
