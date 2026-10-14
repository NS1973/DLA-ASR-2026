# Приёмка ночной кампании ASR — незавершённая

## Актуальный этап после завершения обучения

Обновлено 14 октября 2026 года, 16:46 МСК. Таблица ниже заменяет устаревшие текущие статусы в исторических записях далее. Итоговая сдача ещё не принята.

| Требование | Проверенный факт | Что осталось |
|---|---|---|
| Завершение обучения в 16:00 | Оба тренера штатно завершились в 13:00:07/08 UTC, steps 75013/74028; все 22 этапа terminal | Новые optimizer steps не запускать |
| Закрытые checkpoint | expanded-predecessors-final/999-20261014-160102/verification.json: все четыре best/last model/full пары | Проверка окончательно выбранного resume и привязки к модели |
| Полные закрытые логи | final-worker8-closed-logs.zip: 6988 логических файлов двух последних запусков, все SHA/CRC; прежние логические архивы сохранены | Включить все компоненты в итоговый manifest/архив |
| Родители | overnight-parent-chain-audit-20261014-162859.json: 22 этапа, 8 точных full-resume рёбер, missing=[] | Привязать выбранную кандидатуру к проверенной цепочке |
| Финальный dev-отбор | evaluator PID46126 подтверждён живым; одинаковый BF16/batch16, выбор other при clean<=20 | Дождаться всех кандидатур и selection, затем decoder grid |
| Test | Новых ночных итоговых test-метрик пока нет | Только после фиксации модели/декодера — полные test-clean/test-other |
| Исходники Colab A | commit959ca7c, ZIP1256401 bytes; анонимное скачивание SHA947880caa84b9739388ddde5dbd0d98fd49518c3a4cbb0b9ef8aefb08ea19f62; colab-source-A-anonymous-verification-20261014.json | Новый manifest с окончательными весами/декодером и свежий Colab with/noGT |
| Чистая установка | Source A восстановлен из bundle/ZIP, 172 passed/2 Windows symlink skipped; 55 pinned versions | GPU/Colab проверка выбранной модели остаётся отдельным требованием |
| Упаковка | Новый staging содержит сохранённые конфигурации, родителей, логи и проверки; новый source-ref обязателен | Финальные отчёт/PDF, resources, bundle/архив и точное восстановление |

При подготовке оценки старый helper остановился с mount permission denied, а не таймаутом. Причина — унаследованный seccomp. Временный ограниченный helper изменил только размер собственного tmpfs после проверки RAM; основной контейнер и другие сервисы не перезапускались. 49 моделей скопированы внутри сервера с проверкой исходного и целевого SHA; каталог опубликован. Доказательства: final-own-scratch-remount-recovery.json и final-server-local-model-copy-verification.json.

Коннектор Drive при новой попытке загрузить 329543094 bytes ответил HTTP413 с лимитом 104857600 bytes до вызова действия. Файл через API не создан; браузерная загрузка была запущена, затем отменена по новому указанию пользователя: сдача только локальным архивом. Локальная проверенная копия сохранена. Старое указание лимита 512 MiB не считать актуальным.

## Исторические проверки

Дата проверки:14 октября2026,10:20МСК. Исторический ACCEPTANCE.md относится к e05c и не доказывает приёмку будущей модели.

Инструкция курса: https://github.com/markovka17/dla/blob/2026/hw1ASR/README.md (повторно прочитана14октября). Для CTC test-clean WER20 соответствует строке качества7, test-other WER30 —строке8 до штрафов/бонусов. Фактическая итоговая оценка зависит от проверки преподавателя. Запрет LLM раскрыт в TEMPLATE_PROVENANCE.md; техническая приёмка его не отменяет.

| Проверка | Фактическое состояние / доказательство |
|---|---|
| История шаблона, текущая ветка | source-checkpoint-20261014-115600/verification.json: commitdd37d9e,219файлов ZIP/clone побайтово совпали, template ancestor e9ca57d |
| Чистая CPU установка | clean-current-code-acceptance-20261014-1245.zip:172passed2skipped,pipcheckRC0,55точныхверсий,source007d49b; не доказательство нового GPU Colab |
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

14 октября12:12: фактический новый checkpoint b2d1e664 загружен обычным inference.py на CPU/FP32. Проверены beam50+LM на одной TRAIN-аудиозаписи с эталоном, greedy на3TRAIN-derived WAV/FLAC/MP3 без эталонов(8/16/48кГц,mono/stereo), точные UID, metrics=null безGT и совпадение отдельного calc_metrics.py с inference summary. Архив cpu-real-model-compatibility-20261014-1210.zip SHA649b949c8588662c9741e219eddbdb59322a411e857fc4bc88eba11174b20bca,342231байт,allmemberSHA/CRC. Это функциональная проверка текущего реального кандидата, не качество по train, не финальный Colab и не подтверждение BF16/FP32 идентичности.

14 октября12:50: новый acceptanceZIP5827945байт SHA f2c468f388f34f6dfe803ca0ca00438b1933ecd54fa74540eeb94476f794309d включает чистую установку, полный sourcebundle007d49b,191исходныйфайл ZIP/восстановленного checkout побайтово совпали при исходном Gitcore.autocrlf=true,templateancestor проверен. Прежний тест с принудительным autocrlf=false выявил только EOL и сохранён как диагностика. Обе новые небольшие приватные Driveкопии скачаны обратно и совпали поSHA; доступ owner-only. Старый e05c облачный2.2GB архив имеет подтверждённую загрузку/размер/приватность, облачныйSHA пока не проверен.


14 октября14:50: проверены 45 SHA-записей full-dev-monitor. Последний полный greedy dev-other29.85593154%/dev-clean13.14106099%(85LAST65c85274);75BEST29.87555939/13.06385795. Все четыре новых modelSHA, соответствующие snapshot pairs и полные counts проверены в full-dev-round-20261014-1435-verification.json. Это monitor BF16batch128, не finalbatch16/test. Полные ночные test ещё не выполнялись.

Подготовлен source959ca7c Colab ZIP1256401SHA947880caa84b9739388ddde5dbd0d98fd49518c3a4cbb0b9ef8aefb08ea19f62 и Gitbundle3239423SHA6329dc2530237cb423b64f5fdba8eeddc9e1d8b702d86fce5bc65b7b870effcd: точный HEAD/221ZIPфайл/CRC/восстановление;172теста прошли,2пропуска WindowsSYMLINK. PrivateDrive1VMYIOB54X3j9xr2gFsqXM0xOXTNBtFqR скачан обратно с тем жеSHA; anonymouscheck ещё нет. Ссылка на прежние e05c ресурсы в исходном архиве должна замениться внешним finalmanifest после freeze. OVERNIGHT_COLAB_RELEASE.md задаёт проверку привязки выбранной модели/decoder/LM.

OVERNIGHT_TRAINING_REPRODUCTION.md и docs/repro/*config.json содержат проверенный перенос путей и точные конфигурации обоих ночных запусков. Семь исходных файлов model/runtime/data/codec/metrics/specaugment/manifestbuilder побайтово равны фактическим runtime после нормализации EOL; Python-блок рецепта синтаксически проверен. Повтор всего исторического обучения с нуля на новой машине не заявляется. Новая finalselected модель, GPUColab, finalPDF и архив по-прежнему ожидают окончания обучения и приёмки.

14 октября16:48: пользователь уточнил сдачу — локальный архив и только полные метрики. Публикация репозитория/облачного архива для доставки больше не требуется. Финальный полный dev-отбор и test, содержимое архива и проверки воспроизводимости остаются обязательными.

14 октября16:55: исправлен сборщик final report evidence: scripts/build_final_results.py проверяет актуальный22-stage terminal proof через ops/delivery_evidence.py вместо старого четырёхэтапного формата. 181тест прошёл,2Windows-symlinkпропуска; actual terminal proof принят. Inference/training/evaluator код не менялся. При финальном capture использовать этот tracked script, не logs/build_final_results.py.
