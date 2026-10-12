import math
import re
import string
import numpy as np


def logadd(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


class Codec:
    def __init__(self, alphabet=None):
        self.alphabet = alphabet if alphabet is not None else string.ascii_lowercase + " '"
        self.vocab = [""] + list(self.alphabet)
        self.char_to_id = {c: i for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    @staticmethod
    def normalize(text):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z ']+", "", text.lower())).strip()

    def encode(self, text):
        return [self.char_to_id[c] for c in self.normalize(text)]

    def greedy(self, ids):
        result, previous = [], None
        for index in ids:
            index = int(index)
            if index and index != previous:
                result.append(self.vocab[index])
            previous = index
        return "".join(result).strip()

    def beam_search(self, log_probs, width=20, token_topk=8):
        """CTC prefix search, summing blank/nonblank paths in log space.

        No external decoder implementation is used. width limits prefixes,
        token_topk limits per-frame character expansion; blank is always kept.
        """
        if width < 1:
            raise ValueError("Beam width must be positive")
        beams = {(): (0.0, -math.inf)}
        for frame in np.asarray(log_probs):
            k = min(token_topk, len(frame))
            tokens = np.argpartition(frame, -k)[-k:] if k < len(frame) else np.arange(len(frame))
            next_beams = {}

            def add(prefix, blank=-math.inf, nonblank=-math.inf):
                old_b, old_n = next_beams.get(prefix, (-math.inf, -math.inf))
                next_beams[prefix] = (logadd(old_b, blank), logadd(old_n, nonblank))

            for prefix, (pb, pn) in beams.items():
                total = logadd(pb, pn)
                add(prefix, blank=total + float(frame[0]))
                for token in tokens:
                    token = int(token)
                    if token == 0:
                        continue
                    lp = float(frame[token])
                    if prefix and token == prefix[-1]:
                        add(prefix, nonblank=pn + lp)
                        add(prefix + (token,), nonblank=pb + lp)
                    else:
                        add(prefix + (token,), nonblank=total + lp)
            beams = dict(sorted(next_beams.items(), key=lambda item: logadd(*item[1]), reverse=True)[:width])
        best = max(beams, key=lambda prefix: logadd(*beams[prefix]))
        return "".join(self.vocab[i] for i in best).strip()
