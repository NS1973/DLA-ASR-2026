"""Regression for processing two user links in one notebook runtime."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
from scripts.prepare_custom_dataset import extract
from calc_metrics import directory_metrics


def test_second_notebook_dataset_does_not_include_first(tmp_path, monkeypatch, capsys):
    notebook=json.loads((Path(__file__).parents[1]/'DLA_ASR_Demo.ipynb').read_text(encoding='utf-8'))
    cell=''.join(next(c['source'] for c in notebook['cells'] if c['cell_type']=='code' and 'prepare_custom_dataset.py' in ''.join(c['source'])))
    archives={}
    for name in ['first','second']:
        archive=tmp_path/f'{name}.zip'
        with zipfile.ZipFile(archive,'w') as output:
            output.writestr(f'audio/{name}.wav',b'audio fixture')
            output.writestr(f'transcriptions/{name}.txt',name)
        archives[name]=archive
    outputs=[]
    def run(args,check):
        if args[1].endswith('prepare_custom_dataset.py'):
            root=Path(args[args.index('--output-dir')+1]);root.mkdir(parents=True,exist_ok=True)
            dataset=extract(archives[args[args.index('--url')+1]],root/'extracted')
            (root/'dataset_path.txt').write_text(str(dataset.resolve()),encoding='utf-8')
        elif args[1]=='inference.py':
            options=dict(a.split('=',1) for a in args[2:])
            dataset=Path(options['custom_dir']);dest=Path(options['output_dir']);dest.mkdir(parents=True,exist_ok=True)
            for audio in (dataset/'audio').iterdir():
                (dest/f'{audio.stem}.txt').write_text(audio.stem,encoding='utf-8')
            outputs.append({p.stem for p in dest.glob('*.txt')})
        elif args[1]=='calc_metrics.py':
            result=directory_metrics(args[args.index('--reference-dir')+1],args[args.index('--prediction-dir')+1])
            Path(args[args.index('--output')+1]).write_text(json.dumps(result),encoding='utf-8')
    # Colab uses UTF-8 by default; reproduce that default on Windows too.
    original_read=Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: original_read(self, *a, **{"encoding":"utf-8", **kw}))
    monkeypatch.chdir(tmp_path)
    namespace={'Path':Path,'sys':SimpleNamespace(executable='python'),'subprocess':SimpleNamespace(run=run),'INFERENCE_OPTIONS':[]}
    for link in ['first','second']:
        namespace['DATASET_URL']=link
        exec(compile(cell,'notebook-user-cell','exec'),namespace)
    assert outputs==[{'first'},{'second'}]

    assert capsys.readouterr().out.count('\"wer\": 0.0') == 2
