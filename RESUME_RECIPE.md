# Продолжение итоговой модели

Итоговые model.pt и resume.pt — разные артефакты. Их SHA-256 приведены в FINAL_TRAINING_RECIPE.json. resume.pt соответствует шагу74028, epoch18, batch_index732, micro_count0 и содержит optimizer/scheduler/RNG. Загружайте только доверенные собственные checkpoints.

Точные команды переноса путей и загрузки полного состояния: раздел «Полный resume и новая инициализация» в OVERNIGHT_TRAINING_REPRODUCTION.md. Используйте config expanded004_mix85_intern и manifest mix85; замените cfg.resume на распакованный итоговый resume.pt. Данные должны соответствовать frozen manifest, SHA5220e84b…; точный полный хеш в рецепте. Отключите только истёкший end_utc, задайте новый run_dir. Не меняйте batch, seed, scheduler horizon или данные внутри восстанавливаемого этапа.

Подтверждение совпадения всех тензоров, optimizer/scheduler/RNG и диагностического backward приложено в selected-model-full-verification.json и evaluation20261014.tar.gz. Диагностический пробный шаг не изменял сдаваемую модель. Для нового обучения с другим рецептом используйте отдельный run и model-only initialization; это новый опыт, а не точный resume.
