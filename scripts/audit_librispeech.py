"""Inspect official audio headers and CTC alignment feasibility without Mel caches."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from src.asr.codec import Codec


def audit(root, part):
    directory = Path(root)/'LibriSpeech'/part
    codec = Codec()
    seen, rates, channels = set(), Counter(), Counter()
    problems = []
    seconds, checked, min_margin, max_seconds, max_characters = 0., 0, None, 0., 0
    transcripts = sorted(directory.rglob('*.trans.txt'))
    if not transcripts:
        raise ValueError(f'No transcripts found: {directory}')
    for transcript in transcripts:
        for line in transcript.read_text(encoding='utf-8').splitlines():
            uid, text = line.split(' ', 1)
            if uid in seen:
                problems.append({'uid': uid, 'problem': 'duplicate_id'})
            seen.add(uid)
            path = transcript.parent/(uid+'.flac')
            try:
                info = sf.info(path)
            except Exception as error:
                problems.append({'uid': uid, 'problem': 'audio_header_error', 'type': type(error).__name__})
                continue
            checked += 1
            seconds += info.duration
            max_seconds = max(max_seconds, info.duration)
            rates[info.samplerate] += 1
            channels[info.channels] += 1
            encoded = codec.encode(text)
            max_characters = max(max_characters, len(encoded))
            required = len(encoded)+sum(a == b for a,b in zip(encoded, encoded[1:]))
            # Official LibriSpeech is mono 16kHz. Avoid a duration approximation
            # for unexpected input; record it explicitly instead.
            if info.samplerate != 16000 or info.channels != 1:
                problems.append({'uid': uid, 'problem': 'unexpected_audio_format'})
                continue
            samples = max(400, info.frames)
            frames = samples//160+1
            output_frames = (frames+3)//4
            margin = output_frames-required
            min_margin = margin if min_margin is None else min(min_margin, margin)
            if margin < 0:
                problems.append({'uid': uid, 'problem': 'ctc_infeasible',
                                 'output_frames': output_frames, 'required_frames': required})
    audio_count = sum(1 for _ in directory.rglob('*.flac'))
    if audio_count != len(seen):
        problems.append({'problem': 'audio_transcript_count_mismatch',
                         'audio_files': audio_count, 'unique_transcript_ids': len(seen)})
    return {'part': part, 'utterances': len(seen), 'audio_headers_checked': checked,
            'audio_files': audio_count, 'hours': seconds/3600, 'sample_rates': dict(rates),
            'channels': dict(channels), 'max_duration_seconds': max_seconds,
            'max_target_characters': max_characters, 'minimum_ctc_margin': min_margin,
            'problems': problems, 'passed': not problems,
            'completed_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/work/data/datasets')
    parser.add_argument('--part', default='train-clean-360')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ready-file', type=Path)
    parser.add_argument('--ready-deadline', default='2026-10-13T03:00:00+00:00')
    args = parser.parse_args()
    if args.ready_file:
        deadline = datetime.fromisoformat(args.ready_deadline).timestamp()
        print(json.dumps({'event': 'waiting_for_data', 'marker': str(args.ready_file)}), flush=True)
        while not args.ready_file.exists():
            if time.time() >= deadline:
                print('Optional data was not ready before the cutoff; audit not run.', flush=True)
                return
            time.sleep(5)
    result = audit(args.data_root, args.part)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(destination)+'.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps(result), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
