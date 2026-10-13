"""Mirror a growing official archive, then verify before extraction.

Wait for primary data to avoid slowing its initial transfer. The remote .ready
marker is created only after the complete source hash and extraction succeed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from gmc import connect_h200
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--part',default='train-clean-360')
p.add_argument('--wait-ready',default='train-clean-100')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
remote = '/home/protei_ii/dla-asr-2026'
source = root/'data/downloads'/f'{args.part}.tar.gz'
partial = Path(str(source)+'.partial')
destination = f'{remote}/incoming/{source.name}'
cutoff = datetime.fromisoformat('2026-10-13T03:00:00+00:00').timestamp()
for attempt in range(100):
    if time.time() >= cutoff:
        print('Optional-data transfer cutoff reached; primary training continues',flush=True)
        break
    transport = None
    try:
        transport = connect_h200()
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get_channel().settimeout(60)
        try:
            sftp.stat(f'{remote}/data/{args.wait_ready}.ready')
        except FileNotFoundError:
            sftp.close(); transport.close(); time.sleep(15); continue
        try:
            offset = sftp.stat(destination+'.partial').st_size
        except FileNotFoundError:
            offset = 0
        with sftp.open(destination+'.partial','ab',bufsize=2**20) as output:
            output.set_pipelined(True)
            checked_at = -256*2**20
            while time.time() < cutoff:
                path = source if source.exists() else partial
                if not path.exists():
                    time.sleep(5); continue
                with path.open('rb') as stream:
                    stream.seek(offset)
                    chunk = stream.read(8*2**20)
                if chunk:
                    if offset-checked_at >= 256*2**20:
                        channel = transport.open_session(timeout=10)
                        channel.settimeout(30)
                        channel.exec_command("python3 -c \"import os;s=os.statvfs('/home/protei_ii/dla-asr-2026');print(s.f_bavail*s.f_frsize)\"")
                        free = int(channel.makefile('r').read().strip())
                        if channel.recv_exit_status() or free < 20*2**30 + 256*2**20:
                            raise RuntimeError('Remote free-space reserve reached')
                        channel.close()
                        checked_at = offset
                        print(f'PROGRESS {args.part} {offset} bytes; free={free}',flush=True)
                    output.write(chunk)
                    offset += len(chunk)
                elif source.exists():
                    break
                else:
                    time.sleep(5)
        if not source.exists() or offset != source.stat().st_size:
            continue
        sftp.posix_rename(destination+'.partial',destination)
        sftp.close(); transport.close(); transport = None
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream,'sha256').hexdigest()
        command = f"echo '{digest}  {destination}' | sha256sum -c - && tar -xzf {shlex.quote(destination)} -C {remote}/data/datasets && touch {remote}/data/{args.part}.ready"
        subprocess.run([sys.executable,str(root/'ops/gmc.py'),'exec',command,'--timeout','600'],check=True)
        print(f'REMOTE_DATA_READY {args.part} sha256={digest}',flush=True)
        break
    except Exception as exc:
        print(f'RETRY {attempt}: {type(exc).__name__}: {exc}',flush=True)
        time.sleep(10)
    finally:
        if transport is not None:
            transport.close()
