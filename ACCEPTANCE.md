# Приёмка итоговой ASR-модели, 14 октября 2026

Эта таблица относится к модели 5d2e8e35, а не к прежним baseline/e640/e05c. Незавершённый пункт не считается выполненным.

| Требование | Подтверждённый факт |
|---|---|
| Собственная CTC-модель | 24 471 453 параметра, собственное обучение; все родители и полные configs сохранены |
| Шаблон и Git-история | Template ancestor e9ca57d; source.bundle сохраняет исходные commit, ZIP и каноническое Git-tree восстанавливаются точно |
| Интерфейсы | train.py, inference.py, CustomDirDataset в src/datasets, Hydra configs, отдельный calc_metrics.py; 181 passed / 2 Windows symlink skipped на текущем исправленном исходнике (15,56сек) |
| Чистая установка | Зафиксированные 55 CPU зависимостей и проверки source7f9; в текущем исправлении runtime/inference/model/decoder не менялись; дополнительная проверка точного export и bundle выполняется перед упаковкой |
| Аугментации и logging | Четыре waveform аугментации, мягкий SpecAugment; полные JSONL/W&B offline/target-prediction/audio/спектрограммы включены с первого этапа |
| Полные dev/test | 112 уникальных кандидатов, dev-other при clean≤20%, 13decoder settings, freeze до test; все четыре полных набора проверены по counts/errors |
| Собственный beam и LM | На одной модели/полном dev-other: greedy29,78331%, beam50 29,13166%, beam50+LM28,19934%; реальное улучшение |
| Resume | Model/full all-tensor match, optimizer/scheduler/RNG и диагностический backward; сдаваемые веса не изменялись |
| Colab, с эталоном | Окончательные12cells выполнены в свежем CPU Colab, counts3–8, Python errors0; source полностью совпадает с master кроме DATASET_URL; actual ZIP/metrics проверены. Proof Colab_v2_withGT.executed.proof.json |
| Colab, без эталонов | Те же12cells, counts10–15, Python errors0; отдельный новый workspace, IDs original-16k/stereo-44k/stereo-8k, metrics=null, metrics.json отсутствует. Proof Colab_v2_noGT.executed.proof.json |
| Установка Colab | Fresh CPU project absent подтверждён до установки; source08a25ef ZIP SHA6ad282c1, manifestB2 SHAc003c6ab. Новая GPU сессия отклонена лимитом аккаунта; прежние успешные TeslaT4/PyTorch2.11.0+cu130 proofs сохранены отдельно |
| Отчёт | Четыре страницы PDF, все визуально проверены; полные таблицы/beam-LM сравнение/графики закрытых финальных запусков/анализ и ограничения |
| Ресурсы для инференса | Model, LM, example и frozen source публично доступны с SHA-256; decoder идентичен полной оценке |
| Полные журналы скриптом | Все три компонента скачаны анонимно реальным scripts/download_resources.py; bytes/SHA совпали. Proof all-training-resources-anonymous-verification.json |
| Публичный Git | Репозиторий публичный, anonymous GitHub API200. Финальный commit/anonymousclone/точный Git-tree фиксируются внешним corrected-v2-verification.json после сборки |
| Новый полный архив | v1 сохранён неизменным; v2 создаётся отдельно. Размер/SHA внешнего ZIP, проверка всех членов/CRC и exact bundle restore фиксируются снаружи в corrected-v2-verification.json, receipt Drive — после завершённой загрузки |

Клиентский Colab files.download зависит от доступности output-frame. ZIP также можно скачать через панель «Файлы». Фактические байты ZIP независимо получены и проверены, а не объявлены скачанными по попытке нажатия.

При UI проверке автоматическое заполнение формы не сохранило DATASET_URL в коде: failed-прогон сохранён отдельно. Причина локализована чтением исходной ячейки; ввод и readback параметра исправлены, затем оба полных прогона повторены успешно. Это не замена исходного master другим notebook.

Полный test-clean WER12,12721%, test-other29,38120%. Качество соответствует строке CTC test-other≤30% до штрафов/бонусов, фактическую оценку выставляет курс. Прямой запрет LLM остаётся несоблюдённым и раскрыт в отчёте; техническая приёмка этого не отменяет.
