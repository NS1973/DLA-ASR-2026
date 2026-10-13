"""Package verified, locally collected artifacts without including credentials/data.

The caller supplies a frozen model and run artifacts; this does not select a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--runs', required=True)
    p.add_argument('--resume-checkpoint', help='Full matching checkpoint with optimizer, scheduler and RNG states')
    p.add_argument('--lm', help='Frozen external LM, when selected by dev evaluation')
    p.add_argument('--output', default='delivery/final')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Commit the final source changes before packaging')
    subprocess.run(['git','archive','--format=zip',f'--output={output / "source.zip"}','HEAD'],cwd=root,check=True)
    subprocess.run(['git','bundle','create',str(output/'source.bundle'),'codex/hw1-asr'],cwd=root,check=True)
    shutil.copy2(args.model, output/'model.pt')
    if args.resume_checkpoint:
        shutil.copy2(args.resume_checkpoint, output/'resume.pt')
    if args.lm:
        shutil.copy2(args.lm, output/Path(args.lm).name)
    with zipfile.ZipFile(output/'logs.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        runs = Path(args.runs).resolve()
        for path in sorted(runs.rglob('*')):
            if path.is_file() and path.suffix not in {'.pt','.pth','.tmp'}:
                archive.write(path,path.relative_to(runs))
    for name in ['REPORT.md','REPORT.pdf','DLA_ASR_Demo.ipynb','requirements.txt','requirements-lock-cpu.txt','requirements-report.txt','resources.json','lm_sources.json']:
        if (root/name).exists():
            shutil.copy2(root/name,output/name)
    files = {path.name:{'bytes':path.stat().st_size,'sha256':digest(path)}
             for path in sorted(output.iterdir()) if path.is_file() and path.name not in {'SHA256SUMS','manifest.json','reproduction.zip'}}
    (output/'manifest.json').write_text(json.dumps({'commit':commit,'files':files},indent=2),encoding='utf-8')
    (output/'SHA256SUMS').write_text(''.join(f'{meta["sha256"]}  {name}\n' for name,meta in files.items()),encoding='ascii')
    with zipfile.ZipFile(output/'reproduction.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'reproduction.zip':
                archive.write(path,path.name)
    print(json.dumps({'output':str(output),'commit':commit,'archive_sha256':digest(output/'reproduction.zip')}))


if __name__ == '__main__':
    main()
