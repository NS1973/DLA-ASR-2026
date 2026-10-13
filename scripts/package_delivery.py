"""Package verified, locally collected artifacts without including credentials/data.

The caller supplies a frozen model and run artifacts; this does not select a model.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_source(root, output, source_ref=None):
    root,output=Path(root).resolve(),Path(output).resolve()
    if source_ref is None:
        try:
            source_ref=subprocess.check_output(['git','symbolic-ref','--short','HEAD'],cwd=root,text=True).strip()
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Detached HEAD requires an explicit named source-ref') from error
    reference=source_ref if source_ref.startswith('refs/heads/') else 'refs/heads/'+source_ref
    subprocess.run(['git','check-ref-format',reference],cwd=root,check=True,capture_output=True)
    commit=subprocess.check_output(['git','rev-parse','--verify',reference+'^{commit}'],cwd=root,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if commit!=head:
        raise RuntimeError('source-ref must identify frozen HEAD so copied report/resources match source')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'source.zip').exists() or (output/'source.bundle').exists():
        raise FileExistsError('Existing source export preserved; use a new delivery directory')
    subprocess.run(['git','archive','--format=zip',f'--output={output / "source.zip"}',reference],cwd=root,check=True)
    subprocess.run(['git','bundle','create',str(output/'source.bundle'),reference],cwd=root,check=True)
    return {'commit':commit,'source_ref':reference,'source_branch':reference.removeprefix('refs/heads/')}


def write_logs_zip(destination, directory=None, logs_archive=None):
    """Stream a captured TAR directly, or support the original directory input."""
    if (directory is None) == (logs_archive is None):
        raise ValueError('Supply exactly one log directory or archive')
    destination = Path(destination)
    temporary = Path(str(destination) + '.tmp')
    excluded = {'.pt', '.pth', '.tmp', '.npz', '.partial'}
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, strict_timestamps=False) as archive:
            if logs_archive is not None:
                with tarfile.open(logs_archive, 'r|*') as source:
                    for member in source:
                        if member.isdir():
                            continue
                        name = PurePosixPath(member.name)
                        if (not member.isfile() or not name.parts or name.is_absolute()
                                or '..' in name.parts or '\\' in member.name or ':' in name.parts[0]):
                            raise ValueError(f'Unsafe log archive member: {member.name}')
                        if name.suffix in excluded:
                            continue
                        with source.extractfile(member) as incoming, archive.open(
                                member.name, 'w', force_zip64=member.size >= zipfile.ZIP64_LIMIT) as outgoing:
                            shutil.copyfileobj(incoming, outgoing, length=2**20)
            else:
                root = Path(directory).resolve()
                if not root.is_dir():
                    raise ValueError(f'Logs directory missing: {root}')
                for path in sorted(root.rglob('*')):
                    if path.is_file() and not path.is_symlink() and path.suffix not in excluded:
                        archive.write(path, path.relative_to(root))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    logs = p.add_mutually_exclusive_group(required=True)
    logs.add_argument('--runs', help='Directory of collected run artifacts')
    logs.add_argument('--logs-archive', help='Captured TAR/TAR.GZ; streams without unpacking many files')
    p.add_argument('--resume-checkpoint', help='Full matching checkpoint with optimizer, scheduler and RNG states')
    p.add_argument('--lm', help='Frozen external LM, when selected by dev evaluation')
    p.add_argument('--output', default='delivery/final')
    p.add_argument('--source-ref', help='Named branch at frozen HEAD; defaults to current branch')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(['git','status','--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('Commit the final source changes before packaging')
    source=export_source(root,output,args.source_ref)
    commit=source['commit']
    shutil.copy2(args.model, output/'model.pt')
    if args.resume_checkpoint:
        shutil.copy2(args.resume_checkpoint, output/'resume.pt')
    if args.lm:
        shutil.copy2(args.lm, output/Path(args.lm).name)
    write_logs_zip(output/'logs.zip', directory=args.runs, logs_archive=args.logs_archive)
    for name in ['REPORT.md','REPORT.pdf','DLA_ASR_Demo.ipynb','requirements.txt','requirements-lock-cpu.txt','requirements-report.txt','resources.json','lm_sources.json']:
        if (root/name).exists():
            shutil.copy2(root/name,output/name)
    files = {path.name:{'bytes':path.stat().st_size,'sha256':digest(path)}
             for path in sorted(output.iterdir()) if path.is_file() and path.name not in {'SHA256SUMS','manifest.json','reproduction.zip'}}
    (output/'manifest.json').write_text(json.dumps({**source,'files':files},indent=2),encoding='utf-8')
    (output/'SHA256SUMS').write_text(''.join(f'{meta["sha256"]}  {name}\n' for name,meta in files.items()),encoding='ascii')
    with zipfile.ZipFile(output/'reproduction.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != 'reproduction.zip':
                archive.write(path,path.name)
    print(json.dumps({'output':str(output),'commit':commit,'archive_sha256':digest(output/'reproduction.zip')}))


if __name__ == '__main__':
    main()
