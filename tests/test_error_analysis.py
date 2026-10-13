import itertools

import pytest

from src.asr.metrics import edit_distance


def test_word_alignment_reports_known_error_types():
    from scripts.analyze_errors import align_words
    assert align_words('we like blue cats'.split(), 'we love blue cats today'.split()) == [
        ('substitution', 'like', 'love'), ('insertion', '', 'today')]
    assert align_words('one two three'.split(), 'one three'.split()) == [('deletion', 'two', '')]
    assert align_words([], ['hello']) == [('insertion', '', 'hello')]
    assert align_words(['hello'], []) == [('deletion', 'hello', '')]


def test_alignment_edit_counts_match_independent_distance_on_short_sequences():
    from scripts.analyze_errors import align_words
    sequences = [list(x) for size in range(4) for x in itertools.product('ab', repeat=size)]
    for reference in sequences:
        for prediction in sequences:
            assert len(align_words(reference, prediction)) == edit_distance(reference, prediction)


def test_analysis_rejects_missing_references_and_duplicate_ids():
    from scripts.analyze_errors import analyze
    with pytest.raises(ValueError, match='reference'):
        analyze([{'uid': 'a', 'target': None, 'prediction': 'hello'}])
    with pytest.raises(ValueError, match='Duplicate'):
        analyze([{'uid': 'a', 'target': 'hello', 'prediction': 'hello'}] * 2)
