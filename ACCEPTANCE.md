# Приёмка итоговой ASR-модели, 14 октября 2026

Эта таблица относится к модели 5d2e8e35, а не к прежним baseline/e640/e05c. Незавершённый пункт не считается выполненным.

| Требование | Подтверждённый факт |
|---|---|
| Собственная CTC-модель | 24 471 453 параметра, собственное обучение; все родители и полные configs сохранены |
| Шаблон и Git-история | Template ancestor e9ca57d; публичный git clone сохраняет историю шаблона и реализации; в ASR-RESULTS история находится в .git |
| Интерфейсы | train.py, inference.py, CustomDirDataset в src/datasets, Hydra configs, отдельный calc_metrics.py; 181 passed / 2 Windows symlink skipped на текущем исправленном исходнике (15,56сек) |
| Чистая установка | Зафиксированные 55 CPU зависимостей и проверки source7f9 и текущий39a5c7f:181passed/2skip,pipcheck; runtime/inference/model/decoder после этого не менялись; точный export/bundle окончательного выпуска проверяется перед упаковкой |
| Аугментации и logging | Четыре waveform аугментации, мягкий SpecAugment; полные JSONL/W&B offline/target-prediction/audio/спектрограммы включены с первого этапа |
| Полные dev/test | 112 уникальных кандидатов, dev-other при clean≤20%, 13decoder settings, freeze до test; все четыре полных набора проверены по counts/errors |
| Собственный beam и LM | На одной модели/полном dev-other: greedy29,78331%, beam50 29,13166%, beam50+LM28,19934%; реальное улучшение |
| Resume | Model/full all-tensor match, optimizer/scheduler/RNG и диагностический backward; сдаваемые веса не изменялись |
| Colab, с эталоном | Окончательные12cells выполнены в свежем CPU Colab, counts3–8, Python errors0; source полностью совпадает с master кроме DATASET_URL; actual ZIP/metrics проверены. Proof Colab_v2_withGT.executed.proof.json |
| Colab, без эталонов | Те же12cells, counts10–15, Python errors0; отдельный новый workspace, IDs original-16k/stereo-44k/stereo-8k, metrics=null, metrics.json отсутствует. Proof Colab_v2_noGT.executed.proof.json |
| Установка Colab | Fresh CPU project absent подтверждён до установки; source08a25ef ZIP SHA6ad282c1, manifestB2 SHAc003c6ab. Новая GPU сессия отклонена лимитом аккаунта; прежние успешные TeslaT4/PyTorch2.11.0+cu130 proofs сохранены отдельно |
| Отчёт | Четыре страницы PDF, все визуально проверены; полные таблицы/beam-LM сравнение/графики закрытых финальных запусков/анализ и ограничения |
| Ресурсы для инференса | Model, LM, example и frozen source публично доступны с SHA-256; decoder идентичен полной оценке |
| Полные журналы скриптом | Все9компонентов: ранние5ZIP, ночные3ZIP и промежуточный worker6ZIP; реальный scripts/download_resources.py проверяет bytes/SHA. Итоговое доказательство all-nine-training-resources-verification.json в corrected-v2-acceptance.zip |
| Публичный Git | Репозиторий публичный, anonymous GitHub API200. Исходный код модели зафиксирован commit1157d264; Git-история и template ancestor проверены; последующие правки затрагивают документацию |
| Готовая папка результатов | ASR-RESULTS проверена: исходники с Git-историей, модель/full checkpoint, распакованная LM, полные логи, manifests/родители и результаты. Приёмка: verification/folder-verification.json внутри готовой папки. Все прежние комплекты сохранены |

Клиентский Colab files.download зависит от доступности output-frame. ZIP также можно скачать через панель «Файлы». Фактические байты ZIP независимо получены и проверены, а не объявлены скачанными по попытке нажатия.

При UI проверке автоматическое заполнение формы не сохранило DATASET_URL в коде: failed-прогон сохранён отдельно. Причина локализована чтением исходной ячейки; ввод и readback параметра исправлены, затем оба полных прогона повторены успешно. Это не замена исходного master другим notebook.

Полный test-clean WER12,12721%, test-other29,38120%. Качество соответствует строке CTC test-other≤30% до штрафов/бонусов, фактическую оценку выставляет курс. Прямой запрет LLM остаётся несоблюдённым и раскрыт в отчёте; техническая приёмка этого не отменяет.
