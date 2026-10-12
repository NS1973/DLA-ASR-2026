"""Local GMC SSH/SFTP utility. Credentials stay in the existing protected loader."""
import argparse
import hashlib
import os
from pathlib import Path
import posixpath
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.environ.get("GMC_LOADER_DIR", r"C:\CodexHome\credentials\112"))
from gmc_tunnel import connect_h200
import paramiko


def mkdirs(sftp, directory):
    parts = directory.split("/")
    path = "/" if directory.startswith("/") else ""
    for part in parts:
        if not part:
            continue
        path = posixpath.join(path, part)
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def upload(sftp, local, remote):
    local = Path(local)
    mkdirs(sftp, posixpath.dirname(remote))
    partial = remote + ".partial"
    try:
        offset = sftp.stat(partial).st_size
    except FileNotFoundError:
        offset = 0
    if offset > local.stat().st_size:
        raise ValueError("Remote partial exceeds source size")
    with local.open("rb") as src, sftp.open(partial, "ab", bufsize=1024 * 1024) as dst:
        src.seek(offset)
        dst.set_pipelined(True)
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    try:
        sftp.posix_rename(partial, remote)
    except IOError:
        try:
            sftp.remove(remote)
        except FileNotFoundError:
            pass
        sftp.rename(partial, remote)
    print(f"UPLOADED {local.name} {local.stat().st_size}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("exec")
    e.add_argument("command")
    e.add_argument("--timeout", type=int, default=60)
    u = sub.add_parser("push")
    u.add_argument("local")
    u.add_argument("remote")
    d = sub.add_parser("pull")
    d.add_argument("remote")
    d.add_argument("local")
    args = p.parse_args()
    t = connect_h200()
    try:
        if args.action == "exec":
            c = t.open_session(timeout=8)
            c.exec_command(args.command)
            end = time.monotonic() + args.timeout
            while True:
                if c.recv_ready():
                    print(c.recv(32768).decode("utf-8", errors="replace"), end="", flush=True)
                if c.recv_stderr_ready():
                    print(c.recv_stderr(32768).decode("utf-8", errors="replace"), end="", file=sys.stderr, flush=True)
                if c.exit_status_ready() and not c.recv_ready() and not c.recv_stderr_ready():
                    sys.exit(c.recv_exit_status())
                if time.monotonic() > end:
                    c.close()
                    raise TimeoutError("Remote command deadline exceeded; detached jobs are unaffected")
                time.sleep(.05)
        else:
            sftp = paramiko.SFTPClient.from_transport(t)
            if args.action == "push":
                upload(sftp, args.local, args.remote)
            else:
                Path(args.local).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(args.remote, args.local)
            sftp.close()
    finally:
        t.close()


if __name__ == "__main__":
    main()
