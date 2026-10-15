# Воспроизведение обучения из этой папки

Все команды выполняются из корня ASR-RESULTS. Уже достигнутые веса находятся в weights/model.pt и weights/resume.pt. Запускать обучение для использования модели не требуется.

## Данные

LibriSpeech — публичный английский корпус OpenSLR12. Полные аудиоданные скачиваются отдельно; в комплекте сохранены фиксированные списки записей, тексты, повторы и порядок.

```bash
python ops/download_data.py --parts train-clean-100 train-clean-360 train-other-500 dev-clean dev-other test-clean test-other
```

Распакуйте скачанные части в data/datasets: ожидается data/datasets/LibriSpeech/train-clean-100 и аналогичные директории остальных частей. Downloader по умолчанию сохраняет загрузки в data/downloads. Для train-other-500 проверены 30593501606 байт и SHA-256 ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2.

Manifests уже распакованы в reproduction/manifests. Итоговый 85/15: reproduction/manifests/data-transitions/expanded004/mix85.json, SHA-256 5220e84bf63582af80ea3f8a781e0a404fdd9958f03496936d05bef90f2b2784. Manifest75/25 лежит рядом. Более ранние stages — expanded001/002/003. Нельзя заменять зафиксированный порядок повторно сгенерированной смесью и называть это точным replay.

496,8568895 уникальных часов other, 464,1964881 clean; экспозиции итоговой смеси85/15 — 2981,1413370/526,0834039 часа. Повторы other не увеличивают уникальный объём. Одна CTC-несовместимая other-запись исключена; проверка разделения train/dev/test сохранена в доказательствах.

## Продолжить итоговый checkpoint

Требуются GPU и обучающие аудиоданные. Команда ниже приведена для Linux/PyTorch2.11.0/CUDA12.9; при подготовке папки она не запускалась.

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python - <<'PY'
import hashlib
import json
from pathlib import Path
from src.asr.runtime import train

root = Path.cwd()
cfg = json.loads((root / 'docs/repro/closed-stages/expanded004_mix85_intern.json').read_text())
cfg['data_root'] = str(root / 'data/datasets')
cfg['training_manifest_path'] = str(root / 'reproduction/manifests/data-transitions/expanded004/mix85.json')
cfg['resume'] = str(root / 'weights/resume.pt')
cfg['run_dir'] = str(root / 'runs/resumed-final85')
cfg['init_checkpoint'] = None
cfg['end_utc'] = None
assert not Path(cfg['run_dir']).exists(), 'Choose an unused run directory'
with Path(cfg['training_manifest_path']).open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == cfg['training_manifest_sha256']
train(cfg)
PY
```

Полный resume восстанавливает optimizer, scheduler, RNG, счётчики и сохранённые градиенты. Состояние итоговой модели: step74028, epoch18, batch_index732, micro_count0. Сохраняются batch256, LR3e-5, warmup300, seed35, offset3400, horizon22, packed_stack и аугментации. Переносятся только файловые пути, отключается истёкший календарный cutoff. Для нового опыта с другими параметрами нужен отдельный run, а не изменение продолжающегося этапа.

## Точная цепочка родителей и конфигураций

Исходные configs всех22ночных этапов: docs/repro/closed-stages. Их исходные серверные пути сохранены как сведения о фактическом запуске. Цепочка, режимы переходов и ожидаемые SHA: docs/repro/final-stage-chain.json. Сохранённые родительские модели и full checkpoint находятся в reproduction/parents/<SHA256>.pt; index.json перечисляет размеры и происхождение. Одинаковые веса, использованные несколькими этапами, хранятся один раз.

Режим own_model_initialization_new_optimizer означает resume=null и init_checkpoint на файл parent_model_sha256. Режим exact_full_resume означает resume на файл parent_resume_sha256; optimizer/scheduler/RNG продолжаются. При переносе training_manifest_path выбрать соответствующий распакованный manifest по исходному config и сверить его SHA. data_root указывает на data/datasets. run_dir должен быть новым, end_utc отключить; batch/seed/horizon/порядок данных сохранить. Для остановки replay на конкретном сохранённом состоянии max_steps задаётся абсолютным шагом этапа, epochs сохраняет исходный горизонт.

Выбранная ветвь: собственная e05c → main85_packed → expanded001_mix75 → expanded002_mix75 и его full resume → expanded003_mix85 → два full resume → expanded004_mix85 → полный resume expanded004_mix85_intern. Точные промежуточные родители могут предшествовать последнему шагу запуска: выбирать их по SHA, а не по названию last.

Ранние рецепты до e05c и их parent-evidence доступны в reproduction/history и reproduction/parent-evidence. Логи всех этапов и сравнений распакованы в logs; указатель verification/log-components.json сопоставляет исходные компоненты с фактическими папками. В исторических JSON archive/member — идентификаторы происхождения; для открытия файлов используйте этот указатель, распаковывать ничего не нужно.

Исторические исходники некоторых этапов старше окончательной версии. Их commits доступны в сохранённой Git-истории; фактический training commit8b260c2f и доказательства совпадения производственного кода сохранены. Новая папка использует исходный итоговый commit1157d2645ea82828380dd5bbe711216fd488f38a с исправленной документацией размещения. Побитовая идентичность всей траектории на другом оборудовании/CUDA не обещается.

## Логи и итоговая оценка

logs/training — все необходимые этапы с начала обучения, метрики, тексты, W&B offline и медиа. logs/comparisons — существующие абляции и исторические сводки. Старые повторные дампы TXT для каждого decoder/candidate не размножены; исходные резервы сохранены отдельно. Текущие полные dev/test предсказания и эталоны доступны в results/corpora.

Выбор модели:112уникальных SHA на полных dev-clean/dev-other, минимум dev-other при dev-clean WER≤20%. Затем13настроек декодера на dev, freeze, затем полный test. Итог: beam50/token_topk8/LM0.1/bonus1. Полный WER test-clean12,12720633%,test-other29,38119710%. Источник точных счётчиков — FINAL_RESULTS.json.


## Ранние50/100часовые части other

Точный состав29767записей раннего100-часового этапа дополнительно сохранён в reproduction/early-other-prefix/manifest-stream.json: исходные FLAC-пути и SHA, длительности и SHA текстов. Полные тексты — early-labels.json в той же папке.50часов входят в100часов; фактические операторы отбора50/100сохранены в operator-sources как исторические исходники с исходными серверными путями. Их следует адаптировать, а не запускать без изменений. Для восстановления100-часовой части из официального train-other-500 выбрать ровно files[source_member], проверить FLAC SHA и transcription_sha256; используемое имя train_parts — train-other100-prefix. Фактический переход с префикса к полному other описан в отчёте. Исторический README этой provenance-папки описывает состояние проверки в момент создания префикса, до последующей загрузки полного500-часового корпуса.
