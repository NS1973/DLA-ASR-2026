"""Mild length-preserving masking of normalized log-Mel training features."""
import torch
from torch import nn


class MildSpecAugment(nn.Module):
    def forward(self, spectrogram, lengths):
        if not self.training:
            return spectrogram
        batch, bands, frames = spectrogram.shape
        device = spectrogram.device
        lengths = lengths.to(device)
        freq_mask = torch.zeros(batch, bands, dtype=torch.bool, device=device)
        time_mask = torch.zeros(batch, frames, dtype=torch.bool, device=device)
        band_index = torch.arange(bands, device=device)[None]
        time_index = torch.arange(frames, device=device)[None]
        for _ in range(2):
            width = torch.randint(0, min(8, bands-1)+1, (batch, 1), device=device)
            start = (torch.rand(batch, 1, device=device) * (bands-width+1)).long()
            freq_mask |= (band_index >= start) & (band_index < start+width)
            cap = (lengths // 20).clamp(max=40)[:, None]
            width = (torch.rand(batch, 1, device=device) * (cap+1)).long()
            start = (torch.rand(batch, 1, device=device) * (lengths[:, None]-width+1)).long()
            time_mask |= (time_index >= start) & (time_index < start+width)
        valid = time_index < lengths[:, None]
        mask = (freq_mask[:, :, None] | time_mask[:, None, :]) & valid[:, None, :]
        return spectrogram.masked_fill(mask, 0)
