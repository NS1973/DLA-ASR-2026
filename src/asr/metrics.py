import math
from src.asr.codec import Codec


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, pred in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != pred)))
        previous = current
    return previous[-1]


def corpus_metrics(references, predictions):
    if len(references) != len(predictions):
        raise ValueError("Reference and prediction counts differ")
    word_errors = char_errors = words = chars = 0
    for reference, prediction in zip(references, predictions):
        reference, prediction = Codec.normalize(reference), Codec.normalize(prediction)
        word_errors += edit_distance(reference.split(), prediction.split())
        char_errors += edit_distance(reference, prediction)
        words += len(reference.split())
        chars += len(reference)

    def ratio(errors, total):
        return errors / total if total else (0.0 if errors == 0 else math.inf)

    return {"wer": ratio(word_errors, words), "cer": ratio(char_errors, chars), "word_errors": word_errors,
            "char_errors": char_errors, "words": words, "chars": chars, "utterances": len(references)}
