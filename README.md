# HW1 ASR — итоговая модель, 14 октября 2026

Собственная DeepSpeech2-CTC для английской речи LibriSpeech. Полные итоговые WER: **test-clean 12,12721%, test-other 29,38120%**. CER: **3,79640% и 11,61103%**. Полные dev-clean/dev-other WER: 12,20911%/28,19934%. Точные числа ошибок, объёмы наборов и отбор 112 моделей — FINAL_RESULTS.json и FINAL_MODEL_SELECTION.md. Итоговый отчёт: REPORT.md / REPORT.pdf.

Модель 5d2e8e35 (expanded004_mix85_intern, last) выбрана по полному dev-other при dev-clean WER ≤20%. Декодер: собственный prefix beam50, token_topk8, внешняя trigram LM0.1, word bonus1. Настройки зафиксированы до итогового test. Архитектура, веса и результаты при исправлении документации не менялись.

## Быстрый запуск в Google Colab

Откройте DLA_ASR_Demo.ipynb в Google Colab, по возможности включите T4 GPU (при недоступном GPU инференс автоматически работает на CPU), вставьте публичную ссылку на ZIP/TAR набора в поле DATASET_URL и выполните все ячейки. Другие параметры менять не требуется. Веса, исходники и LM загружаются автоматически с проверкой SHA-256. Ноутбук показывает встроенный пример и запускает отдельные inference.py и calc_metrics.py на пользовательском наборе.

Для точного запуска исходники устанавливаются из публичного зафиксированного source ZIP с SHA-256. Код клонирования сдаваемой ветки также присутствует; текущий статус публичного репозитория и фактические проверки указаны в ACCEPTANCE.md. Секреты и GitHub-токены не нужны для зафиксированного ZIP.

Пример с эталоном: https://drive.google.com/file/d/1lNBQAlVLU3OnMq3GDdfVI8HAdtRiGflr/view
Пример без эталонов, три аудиоформата: https://drive.google.com/file/d/1HKYGtBXU5oB5Y0ixWT_Wze6v4ADF8gDf/view

Без эталонов сохраняются предсказания; WER/CER не рассчитываются. Это проверка интерфейса, не оценка качества. Итоговые числа выше получены на полных официальных test-наборах.

## Установка в чистое окружение

Python3.12. Распакуйте source.zip в новую папку либо восстановите Git:

```bash
git clone --branch codex/hw1-asr-overnight-20261014 source.bundle DLA-ASR-2026
cd DLA-ASR-2026
python -m venv .venv
```

Активируйте окружение: Linux/macOS — `source .venv/bin/activate`, Windows PowerShell — `.venv\Scripts\Activate.ps1`.

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m pytest tests -q
```

Для точной CPU среды предоставлен requirements-lock-cpu.txt и доказательства чистой установки. requirements-report.txt содержит дополнительные зависимости генерации PDF; они не нужны для распознавания. Сохранены проверки Colab PyTorch2.11.0/TeslaT4 и новая полная проверка окончательного notebook на CPU; лимит GPU аккаунта отражён в ACCEPTANCE.md. Обучение выполнено на H200 с BF16; при CPU инференсе используется FP32, поэтому побитовое равенство разных устройств не заявляется.

## Скачать все внешние ресурсы

Веса, пример и LM:

```bash
python scripts/download_resources.py --manifest resources.json --output-dir artifacts
```

Полные закрытые журналы обучения с самого первого этапа, W&B offline и медиа:

```bash
python scripts/download_resources.py --manifest training-resources.json --output-dir training-artifacts
```

Оба вызова проверяют SHA-256; повторный запуск использует готовые проверенные файлы и возобновляет незавершённую передачу. Тяжёлые веса, аудио и журналы не помещаются в Git. В полном reproduction.zip все журналы, родители, model.pt, resume.pt и LM уже приложены; скачивание их повторно не требуется. Начните с ARCHIVE_README.md. Файл training-resources.json публикуется только с реально проверенными ссылками.

## Инференс на своём наборе

Формат: `audio/ID.wav` (также FLAC/MP3, mono/stereo, произвольная частота) и необязательные `transcriptions/ID.txt`. Идентификаторы сохраняются. Папка предсказаний задаётся явно:

```bash
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions decoder=beam beam_width=50 token_topk=8 beam_workers=2 lm_path=artifacts/3-gram.pruned.1e-7.arpa.gz lm_weight=0.1 word_bonus=1.0
python calc_metrics.py --reference-dir /path/to/dataset/transcriptions --prediction-dir predictions
```

Для весов из полного архива замените checkpoint и lm_path на пути к распакованным файлам. Словарь: английские буквы, пробел, апостроф, CTC blank. CustomDirDataset находится в src/datasets/custom_dir_dataset.py; Hydra config — src/configs/asr_inference.yaml. WER/CER корпусные: сумма ошибок, делённая на суммарное число слов/символов.

## Обучение и воспроизведение

Для запуска собственного нового опыта:

```bash
python train.py --help
python train.py data_root=/path/to/LibriSpeech run_dir=runs/new-experiment
```

Точный рецепт итоговой модели, сохранённые configs, frozen manifests и перенос путей: OVERNIGHT_TRAINING_REPRODUCTION.md, FINAL_TRAINING_RECIPE.json и RESUME_RECIPE.md. Полный resume загружает optimizer/scheduler/RNG; model-only init создаёт новый этап. Не подменяйте одно другим. Конфигурации фиксируют batch256, workers8, LR3e-5, warmup300, seed35, offset3400, cosine horizons, packed_stack и аугментации.

Данные: 496,8569 уникальных часов other и 464,1965 clean. Финальная смесь85/15 по длительности экспозиций сохраняет clean replay; повторение other не увеличивает уникальный объём. Исходный LibriSpeech скачивается отдельно. Manifests фиксируют точные записи/порядок; все родительские состояния сохранены. Полная историческая траектория на другом железе не объявляется побитово идентичной.

## Документы и прежние результаты

ACCEPTANCE.md — текущие подтверждённые проверки; исторические ledgers имеют явную пометку. README_E05C_HISTORICAL.md, REPORT_BASELINE и REPORT_E640 относятся к прежним моделям. Их метрики не являются текущими. Все прежние архивы и веса сохранены отдельно.

Использовалась помощь LLM. Правила курса запрещают LLM; техническая воспроизводимость не отменяет это несоответствие.
