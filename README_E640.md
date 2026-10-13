# Промежуточный комплект e640

Это отдельный проверенный кандидат, сохранённый до окончания нового обучения в 20:40 МСК. Окончательная модель может быть выбрана позднее только по полным dev-наборам. Исторический baseline и все его архивы сохранены отдельно.

Полные WER: dev-clean 13.72744%, dev-other 37.57949%, test-clean 13.97976%, test-other 39.88499%. Декодер beam10, LM weight 0.2, word bonus 1, token_topk 8. Подробности и ограничения — REPORT_E640.md / REPORT_E640.pdf; точные счётчики — INTERMEDIATE_RESULTS.json.

Установка и инференс из исходников:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate; Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_resources.py --manifest INTERMEDIATE_RESOURCES.json
python inference.py checkpoint=artifacts/model.pt custom_dir=/path/to/dataset output_dir=predictions decoder=beam beam_width=10 token_topk=8 beam_workers=2 lm_path=artifacts/3-gram.pruned.1e-7.arpa.gz lm_weight=0.2 word_bonus=1.0
```

Пользовательская папка содержит audio/ с WAV/FLAC/MP3 и необязательную transcriptions/ с TXT того же ID. Без эталонов сохраняются только предсказания. DLA_ASR_Demo.ipynb использует resources.json; для этого промежуточного кандидата скопируйте INTERMEDIATE_RESOURCES.json поверх resources.json в распакованных исходниках перед запуском notebook. В корне отдельного промежуточного комплекта resources.json уже соответствует e640; source.zip сохраняет исходный Git-снимок с обоими манифестами.

model.pt — модель, resume.pt — соответствующий full checkpoint. initializations.zip и extension-initializations.zip содержат собственную цепочку родителей. logs.zip и extension-logs.zip — полные закрытые журналы двух кампаний; evaluation-e640.tar.gz — полные dev/test сравнения и предсказания. acceptance-e640.zip содержит исполненные notebook и проверки с эталонами и без них. Исторические проверки baseline явно отделены.

SHA256SUMS и manifest.json задают размеры и хеши файлов. source.bundle содержит полную Git-историю; восстановление: git clone --branch codex/hw1-asr source.bundle restored-source. Точный commit записан в manifest.json. Отсутствие публичного anonymous clone не скрывается: источник пока приватен, Git bundle служит согласованным резервным способом передачи.

Обучающий рецепт исходного родителя — REPRODUCTION.md; параметры дополнительного этапа — REPORT_E640.md и закрытый config.json в extension-logs.zip. При новом запуске каталог должен быть новым, календарный cutoff отключается end_utc=null. Для возобновления собственного этапа используйте resume=/path/to/resume.pt; это отличается от init_checkpoint с созданием нового optimizer/scheduler.

Правила курса запрещают LLM. AI-помощь раскрыта; техническая проверка не подтверждает соблюдение этого правила.
