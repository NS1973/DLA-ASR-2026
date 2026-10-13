import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DeepSpeech2CTC(nn.Module):
    """Two strided convolutions, stacked bidirectional GRUs, CTC projection."""
    def __init__(self, n_feats=80, n_tokens=29, hidden=512, layers=5, conv_channels=32, dropout=.1):
        super().__init__()
        self.spec = dict(n_feats=n_feats, n_tokens=n_tokens, hidden=hidden, layers=layers,
                         conv_channels=conv_channels, dropout=dropout)
        self.conv1 = nn.Conv2d(1, conv_channels, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels * 2, 3, stride=2, padding=1)
        # Frame-wise normalization cannot leak statistics from padded future frames.
        reduced_features = (n_feats + 3) // 4
        dimension = reduced_features * conv_channels * 2
        self.input_norm = nn.LayerNorm(dimension)
        self.rnns = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(layers):
            self.rnns.append(nn.GRU(dimension if i == 0 else hidden * 2, hidden, batch_first=True, bidirectional=True))
            self.norms.append(nn.LayerNorm(hidden * 2))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_tokens)

    @staticmethod
    def transform_input_lengths(lengths):
        return (lengths + 3) // 4

    @staticmethod
    def mask_time(x, lengths):
        mask = torch.arange(x.shape[-1], device=x.device)[None, :] < lengths.to(x.device)[:, None]
        return x * mask[:, None, None, :]

    def forward(self, spectrogram, spectrogram_length, **kwargs):
        lengths = spectrogram_length.to(dtype=torch.long, device="cpu")
        x = self.mask_time(spectrogram[:, None], lengths)
        x = torch.relu(self.conv1(x))
        x = self.mask_time(x, (lengths + 1) // 2)
        x = torch.relu(self.conv2(x))
        output_lengths = self.transform_input_lengths(lengths)
        x = x.permute(0, 3, 1, 2).flatten(2)
        x = self.input_norm(x)
        if self.training and getattr(self, 'packed_stack', False):
            total_length = x.shape[1]
            packed = pack_padded_sequence(x, output_lengths, batch_first=True, enforce_sorted=False)
            for rnn, norm in zip(self.rnns, self.norms):
                packed, _ = rnn(packed)
                packed = packed._replace(data=self.dropout(norm(packed.data)))
            x, _ = pad_packed_sequence(packed, batch_first=True, total_length=total_length)
        else:
            for rnn, norm in zip(self.rnns, self.norms):
                packed = pack_padded_sequence(x, output_lengths, batch_first=True, enforce_sorted=False)
                packed, _ = rnn(packed)
                x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
                x = self.dropout(norm(x))
        log_probs = self.head(x).float().log_softmax(-1)
        return {"log_probs": log_probs, "log_probs_length": output_lengths}
