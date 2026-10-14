"""Freeze duration-weighted training exposure; repetitions are never unique hours."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random


def build_manifest(clean, other, other_fraction, seed=10, excluded_ids=()):
    if not 0 < other_fraction < 1 or not clean or not other:
        raise ValueError('Nonempty clean/other and fraction between zero and one required')
    seen = set()
    excluded = set(excluded_ids)
    for row in clean + other:
        if row['uid'] in seen:
            raise ValueError('Input utterance IDs must be unique')
        seen.add(row['uid'])
        if row['uid'] in excluded:
            raise ValueError('Training contains an excluded dev/test ID')
        path = PurePosixPath(row['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in row['path'] or ':' in row['path']:
            raise ValueError('Manifest paths must be safe relative POSIX paths')
        if not math.isfinite(row['duration']) or row['duration'] <= 0:
            raise ValueError('Duration must be finite and positive')
    clean_time = sum(x['duration'] for x in clean)
    other_time = sum(x['duration'] for x in other)
    repeats = max(1, math.ceil(clean_time * other_fraction / ((1-other_fraction)*other_time)))
    target_clean = other_time * repeats * (1-other_fraction) / other_fraction
    replay = list(clean)
    rng = random.Random(seed)
    order = list(clean)
    rng.shuffle(order)
    actual_clean = clean_time
    index = 0
    while actual_clean < target_clean:
        row = order[index % len(order)]
        # Nearest attainable duration using whole utterances; all originals retained.
        if abs(actual_clean + row['duration'] - target_clean) > abs(actual_clean - target_clean):
            break
        replay.append(row)
        actual_clean += row['duration']
        index += 1
    rows = replay + list(other) * repeats
    rng.shuffle(rows)
    return {'schema_version': 1, 'seed': seed, 'entries': rows,
            'statistics': {'requested_other_fraction': other_fraction,
                'actual_other_fraction': other_time * repeats / (actual_clean + other_time * repeats),
                'unique_clean_hours': clean_time / 3600, 'unique_other_hours': other_time / 3600,
                'other_repeats': repeats, 'clean_exposure_hours': actual_clean / 3600,
                'other_exposure_hours': other_time * repeats / 3600,
                'unique_utterances': len(seen), 'exposure_utterances': len(rows)}}


def load_manifest(path, data_root, expected_sha256, intern_entries=False):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('Training manifest SHA mismatch')
    canonical = {}
    def intern(row):
        if not isinstance(row.get('uid'), str) or 'path' not in row:
            return row
        versions = canonical.setdefault(row['uid'], [])
        for existing in versions:
            if existing == row and list(existing) == list(row):
                return existing
        versions.append(row)
        return row
    document = json.loads(content, object_hook=intern if intern_entries else None)
    if document['schema_version'] != 1:
        raise ValueError('Unsupported training manifest version')
    rows = document['entries']
    root = Path(data_root).resolve()
    result = []
    resolved_rows = {}
    for row in rows:
        if intern_entries and id(row) in resolved_rows:
            result.append(resolved_rows[id(row)])
            continue
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or ':' in row['path'] or '\\' in row['path']:
            raise ValueError('Manifest paths must be safe relative POSIX paths')
        # Dataset views intentionally symlink immutable clean/dev directories.
        # Validate lexical traversal above, but preserve the operator-owned view.
        resolved = root.joinpath(*relative.parts)
        if not resolved.is_relative_to(root):
            raise ValueError('Manifest path escapes data_root')
        entry = {**row, 'path': str(resolved)}
        result.append(entry)
        if intern_entries:
            resolved_rows[id(row)] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clean', required=True)
    parser.add_argument('--other', required=True)
    parser.add_argument('--excluded-ids', required=True)
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    document = build_manifest(json.loads(Path(args.clean).read_text()),
        json.loads(Path(args.other).read_text()), args.fraction, args.seed,
        json.loads(Path(args.excluded_ids).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Frozen manifest already exists')
    output.write_text(json.dumps(document, ensure_ascii=True, separators=(',', ':')), encoding='utf-8')
    print(json.dumps({'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), **document['statistics']}))


if __name__ == '__main__':
    main()
