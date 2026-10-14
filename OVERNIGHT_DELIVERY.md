# Сборка после обучения 14 октября

Это инструкция подготовки, а не подтверждение финальной приёмки. Обучение
заканчивается 14 октября в 16:00 МСК; окончательные метрики и комплект появляются
после проверки. Сохранённые комплекты 13 октября остаются отдельными резервами.

## Источники результата

- `/overnight/cutoff20261014/complete.json`: фактическое завершение всех этапов.
- `/overnight/evaluation20261014-root/runs/final-evaluation`: полный dev, выбор
  модели, сетка декодеров, freeze и только затем полный test.
- Локальный `final20261014-selected/verification.json`: SHA модели/full checkpoint
  и совпадение тензоров и состояния.
- `evaluation20261014.tar.gz` и его проверка: исходные предсказания, числа ошибок,
  resume acceptance, runtime provenance и полный состав корпусов.
- `expanded-predecessors-final/*/verification.json`: закрытые этапы, включая
  продолжения с сохранённым optimizer. Файл без завершившейся проверки не включать.
- `closed-log-index-initial/verification.json` и
  `closed-log-index-specprobe/verification.json`: ранние ночные этапы.

Все локальные пути выше относятся к выделенному каталогу резервов задачи на E.
Содержимое архивов Apple и остальные каталоги E не относятся к сборке.

## Проверки перед упаковкой

1. Подтвердить terminal PID и `train_exit`, модель и декодер зафиксированы до test,
   правило выбора — минимум полного dev-other при полном dev-clean WER ≤20%.
2. Сверить SHA выбранных весов и full checkpoint, восстановление optimizer,
   scheduler, RNG и градиентов. Сохранить всю цепочку инициализаций и продолжений.
3. Обновить `FINAL_RESULTS.json`, `FINAL_TRAINING_RECIPE.json`, отчёт, инструкции
   воспроизведения, notebook и `resources.json` по фактическому новому результату.
   Исторические результаты в прежних комплектах не менять.
4. Проверить анонимное скачивание ресурсов и их SHA. Выполнить новый notebook
   целиком в свежем Colab с эталонами и без них; сохранить выполненные notebook и
   доказательства. Старый проверенный Colab не подтверждает новые веса.
5. Проверить установку из чистого экспорта, актуальные тесты, все страницы нового
   PDF и точное соответствие исходников frontend/decoder зависимостям в Colab.
6. Сохранить новые доказательства приёмки, полные закрытые логи, исходные метрики
   и изображения, training-data provenance, операторские скрипты и родительские
   веса. Не выдавать временный staging за завершённый комплект.

## Упаковка логов

`scripts/package_delivery.py` принимает повторяемый аргумент
`--closed-snapshot-proof`. Он использует проверенный
`ops.snapshot_log_packaging.write_snapshot_logs`: восстанавливает логические
имена из content-addressed хранилища, проверяет SHA и `train_exit`, исключает
веса и derived NPZ, затем проверяет каждый ZIP member.

Не передавать content-addressed `closed-stages.tar.gz` через `--logs-archive`:
внутренние `blobs/<sha>` не являются именами логов. Аргумент `--logs-archive`
предназначен для обычного TAR с исходными логическими путями.

Перед запуском положить в НОВЫЙ каталог комплекта все дополнительные материалы:
старые проверенные log ZIP, отдельные parent ZIP, evaluation TAR, data provenance,
доказательства Colab/ресурсов/PDF и операторский ZIP. Их SHA попадут в manifest.
Ничего не перезаписывать в прежних каталогах сдачи.

Пример команды, после подстановки фактических проверенных путей:

```powershell
.venv\Scripts\python.exe scripts/package_delivery.py `
  --source-ref codex/hw1-asr-overnight-20261014 `
  --model <selected_model.pt> --resume-checkpoint <selected_resume.pt> `
  --lm <selected_language_model> `
  --closed-snapshot-proof <closed_stage_verification.json> `
  --closed-snapshot-proof <another_closed_stage_verification.json> `
  --output <new_release_directory>
```

Все исходники и документы предварительно закоммитить. `source-ref` должен
указывать на frozen HEAD. Не использовать старый `package_final2040.py`:
он содержит имена прежней ветки, отчёта и источников приёмки.

## Проверка готового комплекта

Проверить SHA/размер каждого файла manifest и каждого member `reproduction.zip`,
ZIP CRC, все parent-model SHA, model/full matching. Восстановить `source.bundle`
через `git clone --branch codex/hw1-asr-overnight-20261014`, сверить HEAD,
чистоту checkout и наличие исходного шаблона в истории. Сравнить source ZIP с
тем же commit. Записать внешний SHA всего reproduction ZIP и фактический итог
проверок. Публичность и anonymous clone подтверждать отдельным реальным запуском;
при недоступной авторизации использовать разрешённый Git bundle fallback.

Цель закрывается после обязательной приёмки нового комплекта. Напоминалка
останавливается после выдачи результата, а не автоматически в 16:00.
