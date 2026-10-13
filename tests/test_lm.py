import math
import numpy as np
import pytest


def write_lm(tmp_path):
    path = tmp_path/'tiny.arpa'
    path.write_text(r'''
\data\
ngram 1=5
ngram 2=5
\1-grams:
-1 <s> 0
-1 </s> 0
-5 <unk> 0
-1 a -0.2
-1 b 0
\2-grams:
-0.01 <s> a
-0.01 a b
-3 a a
-0.01 b </s>
-1 a </s>
\end\
''')
    return path


def test_arpa_backoff_and_complete_word_prefix(tmp_path):
    from src.asr.lm import ArpaLM
    lm = ArpaLM(write_lm(tmp_path))
    assert lm.word_logprob(('a',),'b') == pytest.approx(-.01*math.log(10))
    assert lm.word_logprob(('a',),'missing') == pytest.approx(-5.2*math.log(10))
    assert lm.score('a incomple',False) == lm.score('a ',False)
    assert lm.score('a b',True)[0] > lm.score('a a',True)[0]


def test_own_beam_optional_lm_changes_ambiguous_word(tmp_path):
    from src.asr.codec import Codec
    from src.asr.lm import ArpaLM
    codec = Codec(alphabet='ab ')
    probabilities = np.array([[.003,.99,.003,.004],[.003,.003,.004,.99],[.01,.55,.43,.01]])
    assert codec.beam_search(np.log(probabilities),width=20,token_topk=4) == 'a a'
    assert codec.beam_search(np.log(probabilities),width=20,token_topk=4,lm=ArpaLM(write_lm(tmp_path)),lm_weight=.5,word_bonus=1) == 'a b'
