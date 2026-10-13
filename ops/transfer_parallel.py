"""Parallel SFTP channels writing disjoint ranges; resume uses an acknowledged prefix."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def mirror_round(pool, transfer, committed, available, workers=2, block_size=64*2**20):
    spans=[(start,min(block_size,available-start))
           for start in range(committed,min(available,committed+workers*block_size),block_size)]
    futures=[pool.submit(transfer,start,count) for start,count in spans]
    for future in futures:
        future.result()  # Includes remote close/ack before advancing the prefix.
    return committed+sum(count for _,count in spans)



def prepare_remote(control,destination,marker):
    final=destination.removesuffix('.partial')
    try:
        return control.stat(final).st_size,final
    except FileNotFoundError:
        pass
    try:
        with control.open(marker) as stream:offset=json.load(stream)['committed']
        with control.open(destination,'r+b') as stream:stream.truncate(offset)
    except FileNotFoundError:
        try:
            offset=control.stat(destination).st_size
        except FileNotFoundError:
            with control.open(destination,'wb'):
                pass
            offset=0
    return offset,None


def finalize_command(destination,remote,digest):
    final=destination.removesuffix('.partial')
    # Idempotent if the previous connection died after rename/hash/extraction.
    return (f"if [ ! -f {final} ]; then mv {destination} {final}; fi && "
            f"echo '{digest}  {final}' | sha256sum -c - && "
            f"tar -xzf {final} -C {remote}/data/datasets && "
            f"(test -e {remote}/data/train-clean-360.ready || touch {remote}/data/train-clean-360.ready)")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,choices=[2,4],default=4)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from gmc import connect_h200
    import paramiko
    root=Path(__file__).resolve().parents[1]
    remote='/home/protei_ii/dla-asr-2026'
    source=root/'data/downloads/train-clean-360.tar.gz'
    temporary=Path(str(source)+'.partial')
    destination=remote+'/incoming/train-clean-360.tar.gz.partial'
    marker=destination+'.acknowledged.json'
    cutoff=datetime.fromisoformat('2026-10-13T03:00:00+00:00').timestamp()
    for attempt in range(100):
        if time.time()>=cutoff:
            print('Optional transfer cutoff; primary training continues',flush=True)
            return
        transport=None
        try:
            transport=connect_h200()
            control=paramiko.SFTPClient.from_transport(transport)
            control.get_channel().settimeout(60)
            offset,already_complete=prepare_remote(control,destination,marker)
            if already_complete:
                control.close();transport.close();transport=None
                if not source.exists():
                    time.sleep(3);continue
                with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                subprocess.run([sys.executable,str(root/'ops/gmc.py'),'exec',
                                finalize_command(destination,remote,digest),'--timeout','600'],check=True)
                print(f'REMOTE_DATA_READY train-clean-360 sha256={digest}',flush=True)
                return
            def commit(value):
                with control.open(marker+'.new','w') as stream:
                    stream.write(json.dumps({'committed':value,'utc':datetime.now(timezone.utc).isoformat()}))
                control.posix_rename(marker+'.new',marker)
            commit(offset)
            channels=[paramiko.SFTPClient.from_transport(transport) for _ in range(args.workers)]
            for channel in channels:channel.get_channel().settimeout(60)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                while time.time()<cutoff:
                    path=source if source.exists() else temporary
                    available=path.stat().st_size
                    if available==offset:
                        if source.exists():break
                        time.sleep(3);continue
                    c=transport.open_session(timeout=10)
                    c.settimeout(30)
                    c.exec_command("python3 -c \"import os;s=os.statvfs('/home/protei_ii/dla-asr-2026');print(s.f_bavail*s.f_frsize)\"")
                    free=int(c.makefile('r').read().strip())
                    if c.recv_exit_status() or free<20*2**30+args.workers*64*2**20:
                        raise RuntimeError('Remote free-space reserve reached')
                    c.close()
                    base=offset
                    def transfer(start,count):
                        index=(start-base)//(64*2**20)
                        with path.open('rb') as stream:
                            stream.seek(start);content=stream.read(count)
                        if len(content)!=count:raise ValueError('Source shrank during transfer')
                        with channels[index].open(destination,'r+b',bufsize=2**20) as output:
                            output.set_pipelined(True)
                            output.seek(start)
                            output.write(content)
                    started=time.monotonic()
                    offset=mirror_round(pool,transfer,offset,available,workers=args.workers)
                    commit(offset)
                    print(json.dumps({'committed_bytes':offset,'available_bytes':available,
                                      'batch_seconds':round(time.monotonic()-started,2),'free_bytes':free}),flush=True)
            for channel in channels:channel.close()
            control.close();transport.close();transport=None
            if not source.exists() or offset!=source.stat().st_size:continue
            with source.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            command=finalize_command(destination,remote,digest)
            subprocess.run([sys.executable,str(root/'ops/gmc.py'),'exec',command,'--timeout','600'],check=True)
            print(f'REMOTE_DATA_READY train-clean-360 sha256={digest}',flush=True)
            return
        except Exception as exc:
            print(f'RETRY {attempt}: {type(exc).__name__}: {exc}',flush=True)
            time.sleep(10)
        finally:
            if transport is not None:transport.close()
    raise RuntimeError('Transfer exhausted retries; confirmed prefix is preserved')


if __name__=='__main__':main()
