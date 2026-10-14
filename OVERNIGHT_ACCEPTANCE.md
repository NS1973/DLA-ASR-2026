# Приёмка ночной кампании ASR — незавершённая

Дата проверки:14 октября2026,10:20МСК. Исторический ACCEPTANCE.md относится к e05c и не доказывает приёмку будущей модели.

Инструкция курса: https://github.com/markovka17/dla/blob/2026/hw1ASR/README.md (повторно прочитана14октября). Для CTC test-clean WER20 соответствует строке качества7, test-other WER30 —строке8 до штрафов/бонусов. Фактическая итоговая оценка зависит от проверки преподавателя. Запрет LLM раскрыт в TEMPLATE_PROVENANCE.md; техническая приёмка его не отменяет.

| Проверка | Фактическое состояние / доказательство |
|---|---|
| История шаблона, текущая ветка | source-checkpoint-20261014-115600/verification.json: commitdd37d9e,219файлов ZIP/clone побайтово совпали, template ancestor e9ca57d |
| Чистая CPU установка | clean-install-acceptance-20261014-0744.zip:162passed2skipped,pipcheckRC0; не доказательство нового GPU Colab |
| Полный train-other и replayclean | frozen-training-inputs-stage004-20261014-072227.tar.gz;496.8568895uniqueotherh и464.196488clean,не500ровно |
| Сохранность родителей | overnight-parent-chain-audit-20261014-115647.json:22runs/8resumeedges,missing=[]; после16обновить |
| Промежуточный выбор | 37SHA в проверенном full-dev-monitor; greedy b2d1e664 clean13.27892357/other30.23082358%; отдельный более ранний6b97 +beam50LM.1 clean12.60431602/other29.18465887% подтверждён независимым подсчётом; не окончательная оценка |
| Обязательный CustomDirDataset/Hydra | src/datasets/custom_dir_dataset.py экспортирует рабочий класс; inference.py использует src/configs/asr_inference.yaml; calc_metrics.py отдельный |
| Ускорение | worker8-e2e-audio-throughput-20261014-083138.json:+16.43%аудиочасов/час,наблюдательное сравнение; не переносить loader-only+33% на всё обучение |
| Полные закрытые логи | Компоненты закрытых этапов сохраняются; новые живые этапы требуют финального capture после фактического train_exit |
| Новый finaldev/decoder/test | Ожидают16:00 и actualexit; одинаковый BF16batch16,всеbest/last,SHAдедуп,минимумother приclean<=20; freeze доtest |
| Собственный beam и LM | После выбора нужны реальные сравнения greedy/beam/LM на одной модели и одном полном наборе; старое сравнение не заменяет новое |
| Новый ресурсы/Colab/PDF | Ожидают выбранных весов; анонимный SHA-download,новый свежий Colab12cells with/noGT,все страницы PDF визуально проверить |
| Итоговый bundle/архив | Ожидают финального отчёта/ресурсов; scripts/package_delivery.py --source-ref codex/hw1-asr-overnight-20261014,логические closed-snapshot-proof,не raw content-addressed TAR |
| Публикация | Новая ветка приватная до фактической сдачи; Gitbundle fallback разрешён,но не равен публичному gitрепозиторию; anonymousclone только после реальной проверки |

Все тяжёлые доказательства находятся в E:\DLA-ASR-2026-backups\01a13245-8e70-7373-8dae-fbd7d26072df\20261013-1115. Архивы Apple и другие каталоги E не затрагиваются. Цель остаётся активной до обязательной приёмки.

14 октября12:05: текущий tests/169passed2skipped; это локальная CPU регрессия, не новый чистый GPU/Colab запуск. Actual runtime source ZIP18files включает фактический receiver с резервом24GiB. Очередь guarded cutoff45336/evaluator45397/server37656; ресурсная проверка послеactualexit, без изменения обучения.
