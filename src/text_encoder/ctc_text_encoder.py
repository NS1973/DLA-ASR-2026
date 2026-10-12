import torch
from src.asr.codec import Codec

class CTCTextEncoder(Codec):
    EMPTY_TOK = ""
    def __init__(self, alphabet=None, **kwargs):
        super().__init__(alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = self.char_to_id
    def __getitem__(self,index):
        return self.vocab[index]
    def encode(self,text):
        return torch.tensor(super().encode(text),dtype=torch.long).unsqueeze(0)
    def decode(self,inds):
        return "".join(self.vocab[int(i)] for i in inds).strip()
    def ctc_decode(self,inds):
        return self.greedy(inds)
    normalize_text = staticmethod(Codec.normalize)
