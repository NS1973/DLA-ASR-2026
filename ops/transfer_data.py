"""Background transfer queue; never extracts unverified archives remotely."""
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/protei_ii/dla-asr-2026"
parser = argparse.ArgumentParser()
parser.add_argument('--parts', nargs='+', default=["bootstrap", "train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"])
PARTS = parser.parse_args().parts


def gmc(*args):
    subprocess.run([sys.executable, str(ROOT / "ops/gmc.py"), *args], check=True)


for part in PARTS:
    archive = ROOT / "data/downloads" / f"{part}.tar.gz"
    while not archive.exists():
        time.sleep(10)
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    dest = REMOTE + "/incoming/" + archive.name
    for attempt in range(10):
        try:
            gmc("push", str(archive), dest)
            target = REMOTE + ("/data" if part == "bootstrap" else "/data/datasets")
            command = f"mkdir -p {shlex.quote(target)} && echo '{digest}  {dest}' | sha256sum -c - && tar -xzf {shlex.quote(dest)} -C {shlex.quote(target)} && touch {shlex.quote(REMOTE + '/data/' + part + '.ready')}"
            gmc("exec", command, "--timeout", "600")
            print(f"REMOTE_DATA_READY {part}", flush=True)
            break
        except Exception as exc:
            print(f"TRANSFER_RETRY {part} {attempt + 1}: {exc}", flush=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"Transfer exhausted: {part}")
