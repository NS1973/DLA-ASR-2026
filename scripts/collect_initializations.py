"""Collect immutable own parent checkpoints with their original log evidence."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch


def campaign_run(name):
    return name in {'main', 'no_aug'} or name.startswith(
        ('refine100', 'finetune100', 'finetune460', 'lr_candidate_'))


def collect(runs, output, legacy_evidence=None):
    runs, output = Path(runs).resolve(), Path(output).resolve()
    legacy = json.loads(Path(legacy_evidence).read_text()) if legacy_evidence else []
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output)+'.tmp')
    events, models = [], {}
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for log in sorted(runs.glob('*/metrics.jsonl')):
                run = log.parent.name
                if not campaign_run(run):
                    continue
                raw = log.read_bytes()
                # Only complete records from still-active append-only logs.
                raw = raw[:raw.rfind(b'\n')+1]
                for line in raw.splitlines():
                    event = json.loads(line)
                    if event.get('event') != 'initialize_from_model':
                        continue
                    evidence, kind = event, 'recorded_at_initialization'
                    if not event.get('snapshot') or not event.get('source_sha256'):
                        matches = [proof for proof in legacy if proof['run'] == run
                                   and proof['source'] == event['path']
                                   and proof['source_state'] == event['source_state']
                                   and proof['initialize_utc'] == event['utc']]
                        if len(matches) != 1:
                            raise ValueError(f'Missing initialization evidence for {run}')
                        evidence, kind = matches[0], 'retrospective'
                    snapshot = Path(evidence['snapshot']).resolve()
                    if not snapshot.is_relative_to(runs):
                        raise ValueError(f'Parent snapshot is outside project runs: {snapshot}')
                    content = snapshot.read_bytes()
                    digest = hashlib.sha256(content).hexdigest()
                    if digest != evidence['source_sha256']:
                        raise ValueError(f'Parent snapshot SHA-256 mismatch for {run}')
                    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
                    if 'model' not in payload or payload.get('state') != event['source_state']:
                        raise ValueError(f'Parent snapshot state mismatch for {run}')
                    if digest not in models:
                        archive.writestr(f'{digest}.pt', content)
                        models[digest] = {'bytes': len(content), 'state': payload['state']}
                    events.append({'run': run, 'sha256': digest, 'event': event,
                                   'evidence_kind': kind, 'evidence': evidence})
            metadata = {'events': events, 'models': models,
                        'initialization_events': len(events), 'unique_models': len(models)}
            archive.writestr('initializations.json', json.dumps(metadata, indent=2))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy-evidence', type=Path)
    args = parser.parse_args()
    metadata = collect(args.runs, args.output, args.legacy_evidence)
    print(json.dumps({key: metadata[key] for key in ['initialization_events', 'unique_models']}))


if __name__ == '__main__':
    main()
