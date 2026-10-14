"""Pure guards for owned training exits and final evaluation resources."""
from pathlib import Path
import time

GIB = 2**30


def process_matches_launch(pid, launch_config, proc_root=Path('/proc'), *,
                           expected_start_ticks=None, expected_namespace=None):
    process = Path(proc_root) / str(pid)
    try:
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return False
        if str(launch_config).encode() not in (process / 'cmdline').read_bytes().split(b'\0'):
            return False
        if expected_start_ticks is not None and int(fields[19]) != expected_start_ticks:
            return False
        if expected_namespace is not None and (process / 'ns/pid').stat().st_ino != expected_namespace:
            return False
        return True
    except (FileNotFoundError, ProcessLookupError):
        return False


def projected_scratch_quota(total, free, required_free, *, quantum=8*GIB, max_quota=72*GIB):
    if min(total, free, required_free) < 0 or free > total or quantum <= 0:
        raise ValueError('Invalid scratch capacity')
    target = total if free >= required_free else ((total + required_free - free + quantum - 1) // quantum) * quantum
    if target > max_quota:
        raise ValueError('Required scratch quota exceeds the project bound')
    return target


def validate_post_exit_capacity(all_terminal, available, cgroup_headroom, projected_extra,
                                *, host_reserve=24*GIB, internal_reserve=8*GIB):
    if not all_terminal:
        raise ValueError('Training must actually exit before capacity changes')
    if projected_extra < 0:
        raise ValueError('Projected allocation cannot be negative')
    if available - projected_extra < host_reserve:
        raise ValueError('Projected allocation violates host memory reserve')
    if cgroup_headroom - projected_extra < internal_reserve:
        raise ValueError('Projected allocation violates container memory reserve')


def stable_registry_paths(root, writers_alive, *, sleep=time.sleep, interval=5):
    def inventory():
        return {p: (p.stat().st_mtime_ns, p.stat().st_size)
                for p in sorted(Path(root).glob('*/verified.json'))}

    while True:
        if writers_alive():
            sleep(interval)
            continue
        before = inventory()
        sleep(interval)
        if not writers_alive() and before == inventory():
            return list(before)
