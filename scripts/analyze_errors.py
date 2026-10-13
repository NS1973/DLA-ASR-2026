"""Describe measured word errors in inference JSONL; does not select a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.asr.codec import Codec
from src.asr.metrics import corpus_metrics


def align_words(reference, prediction):
    """Return one minimum-edit alignment, preferring diagonal on ties.

    S/D/I decomposition can differ for tied alignments; the total is invariant.
    """
    distances = [list(range(len(prediction) + 1))]
    for i, ref in enumerate(reference, 1):
        row = [i]
        for j, pred in enumerate(prediction, 1):
            row.append(min(distances[-1][j-1] + (ref != pred),
                           distances[-1][j] + 1, row[-1] + 1))
        distances.append(row)
    i, j = len(reference), len(prediction)
    errors = []
    while i or j:
        if i and j and distances[i][j] == distances[i-1][j-1] + (reference[i-1] != prediction[j-1]):
            if reference[i-1] != prediction[j-1]:
                errors.append(('substitution', reference[i-1], prediction[j-1]))
            i -= 1
            j -= 1
        elif i and distances[i][j] == distances[i-1][j] + 1:
            errors.append(('deletion', reference[i-1], ''))
            i -= 1
        else:
            errors.append(('insertion', '', prediction[j-1]))
            j -= 1
    return errors[::-1]


def analyze(records):
    identifiers, counts, confusions, examples = set(), Counter(), Counter(), []
    references, predictions = [], []
    for record in records:
        uid = record['uid']
        if uid in identifiers:
            raise ValueError(f'Duplicate ID: {uid}')
        identifiers.add(uid)
        if record.get('target') is None:
            raise ValueError(f'Missing reference: {uid}')
        target, prediction = Codec.normalize(record['target']), Codec.normalize(record['prediction'])
        errors = align_words(target.split(), prediction.split())
        counts.update(kind for kind, _, _ in errors)
        confusions.update(errors)
        references.append(target)
        predictions.append(prediction)
        examples.append({'uid': uid, 'target': target, 'prediction': prediction,
                         'word_errors': len(errors), 'words': len(target.split())})
    metrics = corpus_metrics(references, predictions)
    if sum(counts.values()) != metrics['word_errors']:
        raise AssertionError('Alignment counts disagree with corpus edit distance')
    return {'metrics': metrics,
            'alignment_rule': 'minimum edit distance; diagonal, deletion, insertion tie order',
            'error_counts': {kind: counts[kind] for kind in ['substitution', 'deletion', 'insertion']},
            'frequent_errors': [{'type': kind, 'reference': ref, 'prediction': pred, 'count': count}
                                for (kind, ref, pred), count in confusions.most_common(20)],
            'example_selection': 'five largest absolute word-error counts; UID breaks ties',
            'examples': sorted(examples, key=lambda row: (-row['word_errors'], row['uid']))[:5]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='inference.py predictions.jsonl')
    parser.add_argument('--summary', help='Optional inference summary.json to cross-check all corpus counts')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = analyze([json.loads(line) for line in Path(args.predictions).read_text(encoding='utf-8').splitlines()])
    if args.summary:
        expected = json.loads(Path(args.summary).read_text(encoding='utf-8'))['metrics']
        if result['metrics'] != expected:
            raise ValueError('Analysis does not match the supplied inference summary')
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': args.output, 'metrics': result['metrics'], 'error_counts': result['error_counts']}))


if __name__ == '__main__':
    main()
