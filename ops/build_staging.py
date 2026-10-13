"""Build a fixed, reproducible training fragment while the full corpus transfers."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('data/downloads')
out = root/'staging'
(out/'audio').mkdir(parents=True, exist_ok=True)
(out/'transcriptions').mkdir(exist_ok=True)
texts, audio = {}, set()
with tarfile.open(root/'train-clean-100.tar.gz', 'r|gz') as archive:
    for member in archive:
        if not member.isfile():
            continue
        name = Path(member.name).name
        if name.endswith('.trans.txt'):
            for line in archive.extractfile(member).read().decode().splitlines():
                uid, text = line.split(' ',1)
                texts[uid] = text
        elif name.endswith('.flac') and len(audio) < 1024:
            (out/'audio'/name).write_bytes(archive.extractfile(member).read())
            audio.add(Path(name).stem)
        if len(audio) >= 1024 and audio <= texts.keys():
            break
for uid in sorted(audio):
    (out/'transcriptions'/f'{uid}.txt').write_text(texts[uid]+'\n',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps({'source':'train-clean-100','selection':'first 1024 archive records','uids':sorted(audio)},indent=2))
dest = root/'staging.tar.gz'
with tarfile.open(dest, 'w:gz') as archive:
    archive.add(out, arcname='staging')
with dest.open('rb') as stream:
    digest = hashlib.file_digest(stream,'sha256').hexdigest()
print(json.dumps({'records':len(audio),'bytes':dest.stat().st_size,'sha256':digest}),flush=True)
