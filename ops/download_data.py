"""Resumable official LibriSpeech downloads and a real training bootstrap."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
from pathlib import Path
import tarfile
import time
import requests

BASE = "https://www.openslr.org/resources/12/"
PARTS = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]


def finalize_archive(temp, dest):
    # Windows readers may momentarily deny rename. Do not return to HTTP Range
    # downloading after the complete archive has already been verified.
    for attempt in range(150):
        try:
            temp.replace(dest)
            return
        except PermissionError:
            time.sleep(.2)
    raise PermissionError(f'Complete archive could not be renamed: {temp}')


def download(part, root):
    dest = root / f"{part}.tar.gz"
    marker = dest.with_suffix(dest.suffix + ".sha256")
    if dest.exists() and marker.exists():
        return
    temp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(8):
        try:
            offset = temp.stat().st_size if temp.exists() else 0
            while True:
                # Bounded ranges avoid gateway buffering of a whole multi-GB response.
                end = offset + 16 * 2 ** 20 - 1
                with requests.get(BASE + dest.name, headers={"Range": f"bytes={offset}-{end}"}, stream=True, timeout=(15, 60)) as r:
                    r.raise_for_status()
                    content_range = r.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if r.status_code == 206 and (not match or int(match[1]) != offset):
                        raise ValueError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab" if offset and r.status_code == 206 else "wb"
                    if mode == "wb":
                        offset = 0
                    with temp.open(mode) as f:
                        for chunk in r.iter_content(2 ** 20):
                            if chunk:
                                f.write(chunk)
                                offset += len(chunk)
                    if r.status_code == 200 or (match and offset == int(match[3])):
                        break
                    if match and offset != int(match[2]) + 1:
                        raise ValueError("Incomplete range response")
            with tarfile.open(temp, "r:gz") as archive:
                for member in archive:
                    if member.name.endswith(".trans.txt"):
                        archive.extractfile(member).read()
            finalize_archive(temp, dest)
            h = hashlib.file_digest(dest.open("rb"), "sha256").hexdigest()
            marker.write_text(h + "\n", encoding="ascii")
            print(json.dumps({"downloaded": part, "bytes": dest.stat().st_size, "sha256": h}), flush=True)
            return
        except Exception as exc:
            print(f"RETRY {part} {attempt + 1} {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Failed to download {part}")


def bootstrap(root, limit=96):
    out = root / "bootstrap"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "transcriptions").mkdir(exist_ok=True)
    texts = {}
    audio = set()
    with requests.get(BASE + "train-clean-100.tar.gz", stream=True, timeout=(15, 60)) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        with tarfile.open(fileobj=r.raw, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if name.endswith(".trans.txt"):
                    for line in archive.extractfile(member).read().decode().splitlines():
                        uid, text = line.split(" ", 1)
                        texts[uid] = text
                elif name.endswith(".flac") and len(audio) < limit:
                    (out / "audio" / name).write_bytes(archive.extractfile(member).read())
                    audio.add(Path(name).stem)
                if len(audio) >= limit and audio <= texts.keys():
                    break
    for uid in sorted(audio):
        (out / "transcriptions" / (uid + ".txt")).write_text(texts[uid] + "\n", encoding="utf-8")
    with tarfile.open(root / "bootstrap.tar.gz", "w:gz") as archive:
        archive.add(out, arcname="bootstrap")
    print(f"BOOTSTRAP_READY {len(audio)} actual train-clean-100 utterances", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/downloads")
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--parts", nargs="*", default=PARTS)
    a = p.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if a.bootstrap:
        bootstrap(root)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda part: download(part, root), a.parts))
