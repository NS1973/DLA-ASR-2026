import itertools
import math
import numpy as np


def test_greedy_preserves_repeats_separated_by_blank():
    from src.asr.codec import Codec
    c = Codec()
    a = c.encode("a")[0]
    assert c.greedy([0, a, a, 0, a, 0]) == "aa"
    assert c.greedy([0, 0]) == ""
    assert c.normalize("  DON'T,  Stop! ") == "don't stop"


def test_prefix_beam_matches_exhaustive_path_sum():
    from src.asr.codec import Codec
    c = Codec(alphabet="ab")
    probabilities = np.array([[.30, .35, .35], [.45, .40, .15], [.35, .25, .40]])
    totals = {}
    for path in itertools.product(range(3), repeat=3):
        text = c.greedy(path)
        totals[text] = totals.get(text, 0) + math.prod(probabilities[t, token] for t, token in enumerate(path))
    best = max(totals, key=totals.get)
    assert c.beam_search(np.log(probabilities), width=100, token_topk=3) == best


def test_beam_handles_adjacent_repeated_characters():
    from src.asr.codec import Codec
    c = Codec(alphabet="a")
    assert c.beam_search(np.log([[.01, .99], [.99, .01], [.01, .99]]), width=10) == "aa"


def test_corpus_metrics_weight_by_total_words():
    from src.asr.metrics import corpus_metrics
    result = corpus_metrics(["a b c d", "e"], ["a b c d", "wrong"])
    assert result["wer"] == .2
    assert result["word_errors"] == 1
    assert result["words"] == 5


def test_empty_reference_metrics_are_defined():
    from src.asr.metrics import corpus_metrics
    assert corpus_metrics([""], [""])["wer"] == 0
    assert math.isinf(corpus_metrics([""], ["word"])["wer"])
