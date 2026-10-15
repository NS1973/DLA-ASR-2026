# Выбор акустической модели по полным dev-наборам

Зафиксирована модель `lr_candidate_overnight_expanded004_mix85_intern__last`, SHA-256 `5d2e8e35b01f32cd7b4b7e553de419ab796a545ef606fad994f783f35905eef9`. Рассмотрены 112 уникальных моделей; одинаковые SHA оценивались один раз. Все оценки ниже — greedy, BF16, batch16, полный dev-clean2703записи/54402слова и dev-other2864записи/50948слов.

Критерий: минимальный корпусный WER dev-other среди вариантов с dev-clean WER<=20%; равенство разрешается по dev-clean и имени. Выбор воспроизведён из сохранённого `results/evaluation/runs/final-evaluation/selection.json` в готовой папке ASR-RESULTS. Test не использовался для выбора. Декодер выбран отдельным полным dev-сравнением; эта таблица фиксирует акустический отбор до decoder/test, итоговые метрики находятся в FINAL_RESULTS.json.

| Кандидат | clean errors/words | clean WER,% | other errors/words | other WER,% |
|---|---:|---:|---:|---:|
| lr_candidate_overnight_expanded004_mix85_intern__last | 7053/54402 | 12.964597 | 15174/50948 | 29.783308 |
| lr_candidate_overnight_expanded004_mix75_intern__last | 7015/54402 | 12.894747 | 15177/50948 | 29.789197 |
| lr_candidate_overnight_expanded004_mix85_intern__best | 7097/54402 | 13.045476 | 15186/50948 | 29.806862 |
| lr_candidate_cached_65c852741b885416ce4102d58a5592b9eb01ceedc4db651475974a20d9bc9d12 | 7148/54402 | 13.139223 | 15195/50948 | 29.824527 |
| lr_candidate_cached_74b3b3068062bf0da9d091be54b653ab71ef7f4b33bbe0ccc7ce431ae7124054 | 7069/54402 | 12.994008 | 15204/50948 | 29.842192 |
| lr_candidate_overnight_expanded004_mix75_intern__best | 7074/54402 | 13.003198 | 15208/50948 | 29.850043 |
| lr_candidate_cached_ddbf8a8b3afe4fc14ec0f18fa3d9f1d957df890b4abfe2e27332f0c03ab5c8dd | 7105/54402 | 13.060182 | 15219/50948 | 29.871634 |
| lr_candidate_cached_e7c340960327bf942f079810d9415d7757cfadd0d862ef516ce67b2e124158ff | 7055/54402 | 12.968273 | 15224/50948 | 29.881448 |
| lr_candidate_cached_795c8aadb72208348d44053ccea719648fe1227cceeea4c56a9b331f1385c2de | 7076/54402 | 13.006875 | 15228/50948 | 29.889299 |
| lr_candidate_cached_d2c386a09327b728a316a0bd09f7a2cd013fe05e59e9bb9e46aef3bf3f8905f0 | 7158/54402 | 13.157604 | 15289/50948 | 30.009029 |

Полная таблица всех112кандидатов: results/full-dev-model-selection.csv. Итоговые предсказания и эталоны — results/corpora; сводки оценки и события фиксации — results/evaluation/runs/final-evaluation.

Преимущество выбранного85/15last над75/25last составляет3словесныеошибки на50948словах (0.005888процентного пункта). Это не доказательство устойчивого превосходства смеси85/15; модель выбрана по заранее установленному критерию.
