"""Optional ARPA n-gram shallow fusion; no external ASR decoder is used."""
from functools import lru_cache
import gzip
import math
from pathlib import Path
import re
import sys


class ArpaLM:
    def __init__(self, path):
        self.ngrams = {}
        self.order = 0
        current = 0
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                line = line.strip()
                match = re.fullmatch(r'\\(\d+)-grams:',line)
                if match:
                    current = int(match[1]); self.order = max(self.order,current)
                    continue
                if not line or line.startswith('\\') or not current:
                    continue
                fields = line.split()
                if len(fields) < current+1:
                    continue
                words = tuple(sys.intern(word.lower()) for word in fields[1:current+1])
                probability = float(fields[0])*math.log(10)
                backoff = float(fields[current+1])*math.log(10) if len(fields)>current+1 else 0.0
                self.ngrams[words] = (probability,backoff)
        if not self.order or not self.ngrams:
            raise ValueError(f'No n-grams found in {path}')

    def word_logprob(self, history, word):
        word = word if (word,) in self.ngrams else '<unk>'
        context = tuple(history[-(self.order-1):]) if self.order>1 else ()
        total = 0.0
        while True:
            entry = self.ngrams.get(context+(word,))
            if entry is not None:
                return total+entry[0]
            if not context:
                return total-12*math.log(10)
            total += self.ngrams.get(context,(0.0,0.0))[1]
            context = context[1:]

    @lru_cache(maxsize=50000)
    def _score_words(self, words, final):
        history = ('<s>',)
        score = 0.0
        for word in words:
            score += self.word_logprob(history,word)
            mapped = word if (word,) in self.ngrams else '<unk>'
            history = (*history,mapped)[-(self.order-1):] if self.order>1 else ()
        if final:
            score += self.word_logprob(history,'</s>')
        return score,len(words)

    def score(self, text, final=False):
        # An unfinished final word receives no word score until a space/EOS.
        complete = text if final else text.rsplit(' ',1)[0] if ' ' in text else ''
        return self._score_words(tuple(complete.split()),final)
