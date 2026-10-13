# Возобновление собственного полного checkpoint

Для продолжения этапа недостаточно задать только resume=... с остальными значениями по умолчанию: загрузчик данных и функция scheduler создаются из текущей конфигурации. Сохраните batch/accumulation, train_parts с повторениями, число эпох scheduler, warmup, LR, seed, augmentation_profile, data_epoch_offset и checkpoint_monitor исходного этапа.

Следующая команда использует конфигурацию из доверенного собственного full checkpoint. Она меняет только расположение checkpoint, данных, нового каталога и календарный cutoff. Данные предварительно восстанавливаются в ту же структуру LibriSpeech; порядок ID, тексты и длительности должны совпадать с сохранённым train_manifest.json. Для other100 используются точные ID/хеши из other100-data-provenance.zip и сохранённый канонический архив поднабора; первые 50 часов входят в эти 100 часов.

```python
from pathlib import Path
import torch
from src.asr.runtime import train

checkpoint = Path("resume.pt").resolve()
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
required = {"config", "model", "optimizer", "scheduler", "state",
            "rng_python", "rng_numpy", "rng_torch", "rng_cuda", "gradients"}
if not required <= payload.keys():
    raise ValueError("Нужен full checkpoint, а не model-only")
config = dict(payload["config"])
config.update(resume=str(checkpoint), run_dir="runs/reproduced-resume",
              data_root="data/datasets", end_utc=None)
if Path(config["run_dir"]).exists():
    raise FileExistsError(config["run_dir"])
train(config)
```

Скрипт запускается из корня исходников с установленными requirements.txt. device сохраняется из исходного этапа; для обучения нужен доступный соответствующий GPU. Изменение устройства или batch — отдельное изменение рецепта, а не обещание идентичного продолжения. Предусмотренное восстановление после OOM с меньшим batch сбрасывает batch_index и частично накопленные градиенты на границу эпохи; этот факт записывается в лог.

init_checkpoint отличается от resume: он берёт собственные модельные веса и создаёт новый optimizer/scheduler. Все исходные родительские веса и initialize_from_model события сохранены в архивах и не заменяются после выбора модели. Для новых mixed50/mixed100 этапов состояние родителя проверяется по фактическому SHA и source_state из закрытых логов.

Проверка восстановления выбранного финального checkpoint выполняется отдельно: точные параметры, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG и градиенты, затем диагностический FP32 CPU-шаг без записи изменённых весов. Эта проверка не объявляется полным повторным обучением; воспроизводимость с нуля требует повторения всех сохранённых этапов, данных и конфигураций.
