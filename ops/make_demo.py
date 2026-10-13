"""Build a Colab notebook with one required dataset-link input."""
import json
from pathlib import Path


def markdown(text):
    return {'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}


def code(text, form=False):
    return {'cell_type':'code','metadata':{'cellView':'form'} if form else {},'source':text.splitlines(keepends=True),'outputs':[],'execution_count':None}


cells = [
 markdown('''# ДЗ-1: распознавание английской речи (CTC)
Включите GPU: **Среда выполнения → Сменить среду выполнения → T4 GPU**.
Вставьте ссылку на ZIP/TAR-архив с папкой `audio/` и, при наличии, `transcriptions/`.
Затем выполните **Среда выполнения → Выполнить все**. Остальные ячейки менять не нужно.
Код и веса загружаются автоматически. Если эталонов нет, сохраняются только предсказания.
'''),
 code('''DATASET_URL = "" #@param {type:"string"}
REPOSITORY_URL = "https://github.com/NS1973/DLA-ASR-2026.git"
''',True),
 markdown('## 1. Установка проекта\nКлонируем сдаваемую ветку и устанавливаем зафиксированные зависимости.'),
 code('''import os, subprocess, sys
from pathlib import Path
PROJECT = Path('/content/DLA-ASR-2026')
if not PROJECT.exists():
    subprocess.run(['git','clone','--branch','codex/hw1-asr',REPOSITORY_URL,str(PROJECT)],check=True)
os.chdir(PROJECT)
subprocess.run([sys.executable,'-m','pip','install','-q','-r','requirements.txt'],check=True)
import torch
print('PyTorch:',torch.__version__,'GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
'''),
 markdown('## 2. Загрузка модели\nСкрипт проверяет SHA-256 весов по манифесту сдачи.'),
 code('''import json
subprocess.run([sys.executable,'scripts/download_resources.py'],check=True)
assert Path('artifacts/model.pt').is_file()
manifest = json.loads(Path('resources.json').read_text())
INFERENCE_OPTIONS = manifest.get('inference_overrides', ['decoder=greedy'])
print('Настройки сдаваемой модели:', INFERENCE_OPTIONS)
'''),
 markdown('## 3. Пример использования\nНебольшой настоящий пример показывает инференс и отдельный расчёт метрик.'),
 code('''subprocess.run([sys.executable,'inference.py','checkpoint=artifacts/model.pt','custom_dir=artifacts/example','output_dir=predictions/example','workers=0',*INFERENCE_OPTIONS],check=True)
subprocess.run([sys.executable,'calc_metrics.py','--reference-dir','artifacts/example/transcriptions','--prediction-dir','predictions/example'],check=True)
from IPython.display import Audio, display
audio = next(Path('artifacts/example/audio').iterdir())
display(Audio(filename=str(audio)))
uid = audio.stem
print('Эталон:', Path(f'artifacts/example/transcriptions/{uid}.txt').read_text().strip())
print('Распознано:', Path(f'predictions/example/{uid}.txt').read_text().strip())
'''),
 markdown('## 4. Ваш набор данных\nПоддерживаются прямая ссылка на архив и публичная ссылка Google Drive. Архив может содержать одну внешнюю папку.'),
 code('''if not DATASET_URL.strip():
    raise ValueError('Вставьте ссылку на ваш набор в поле DATASET_URL первой ячейки.')
subprocess.run([sys.executable,'scripts/prepare_custom_dataset.py','--url',DATASET_URL,'--output-dir','data/user_dataset'],check=True)
dataset = Path('data/user_dataset/dataset_path.txt').read_text().strip()
subprocess.run([sys.executable,'inference.py','checkpoint=artifacts/model.pt',f'custom_dir={dataset}','output_dir=predictions/user','workers=0',*INFERENCE_OPTIONS],check=True)
if (Path(dataset)/'transcriptions').is_dir():
    subprocess.run([sys.executable,'calc_metrics.py','--reference-dir',str(Path(dataset)/'transcriptions'),'--prediction-dir','predictions/user','--output','predictions/user/metrics.json'],check=True)
else:
    print('Эталонов нет: метрики не рассчитываются. Предсказания сохранены в predictions/user/.')
'''),
 markdown('## 5. Скачать результат'),
 code('''import shutil
shutil.make_archive('/content/asr_predictions','zip','predictions/user')
from google.colab import files
files.download('/content/asr_predictions.zip')
'''),
]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'DLA_ASR_Demo.ipynb'},'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells}
for index,cell in enumerate(cells):cell['id']=f'cell-{index:02d}'
Path('DLA_ASR_Demo.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding='utf-8')
