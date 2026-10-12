"""Corpus WER/CER with strict ID matching (missing predictions are errors)."""
import argparse
import json
from pathlib import Path
from src.asr.metrics import corpus_metrics


def directory_metrics(reference_dir, prediction_dir):
    references = {p.stem: p for p in Path(reference_dir).glob("*.txt")}
    predictions = {p.stem: p for p in Path(prediction_dir).glob("*.txt")}
    if not references:
        raise ValueError("No ground-truth TXT files")
    missing, unexpected = references.keys() - predictions.keys(), predictions.keys() - references.keys()
    if missing or unexpected:
        raise ValueError(f"ID mismatch: missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}")
    ids = sorted(references)
    return corpus_metrics([references[i].read_text(encoding="utf-8").strip() for i in ids],
                          [predictions[i].read_text(encoding="utf-8").strip() for i in ids])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--prediction-dir", required=True)
    p.add_argument("--output")
    args = p.parse_args()
    result = directory_metrics(args.reference_dir, args.prediction_dir)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
