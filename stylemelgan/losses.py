# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# adapted from https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/losses/stft_loss.py
from random import Random
from typing import Tuple
import torch.nn.functional as F
import torch
from torch.nn import Module
from distutils.version import LooseVersion

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x: torch.Tensor,
         n_fft: int,
         hop_length: int,
         win_length: int) -> torch.Tensor:
    window = torch.hann_window(win_length, device=x.device)
    if is_pytorch_17plus:
        x_stft = torch.stft(
            input=x, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window, return_complex=False)
    else:
        x_stft = torch.stft(
            input=x, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window)

    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class MultiResStftLoss(Module):

    def __init__(self) -> None:
        super().__init__()
        self.n_ffts = (256, 2048)
        self.hop_sizes = (50, 512)
        self.win_lengths = [200, 1200]
        self.random = Random(42)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_loss = 0.
        spec_loss = 0.
        for i in range(5):
            n_fft = self.random.randint(*self.n_ffts)
            hop_length = self.random.randint(*self.hop_sizes)
            win_length = self.random.randint(*self.n_ffts)
            win_length = min(win_length, n_fft)
            x_stft = stft(x=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            y_stft = stft(x=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            norm_loss += F.l1_loss(torch.log(x_stft), torch.log(y_stft))
            spec_loss += torch.norm(y_stft - x_stft, p="fro") / torch.norm(y_stft, p="fro")
        return norm_loss / len(self.n_ffts), spec_loss / len(self.n_ffts)