# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# adapted from https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/losses/stft_loss.py
from typing import Tuple
import torch.nn.functional as F
import torch
from torch.nn import Module
from distutils.version import LooseVersion
from scipy.signal import get_window
import numpy as np

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
        self.n_ffts = [1024, 2048, 512]
        self.hop_sizes = [120, 240, 50]
        self.win_lengths = [600, 1200, 240]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_loss = 0.
        spec_loss = 0.
        for n_fft, hop_length, win_length in zip(self.n_ffts, self.hop_sizes, self.win_lengths):
            x_stft = stft(x=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            y_stft = stft(x=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            norm_loss += F.l1_loss(torch.log(x_stft), torch.log(y_stft))
            spec_loss += torch.norm(y_stft - x_stft, p="fro") / torch.norm(y_stft, p="fro")
        return norm_loss / len(self.n_ffts), spec_loss / len(self.n_ffts)


class TorchSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))

    def transform(self, input_data):
        forward_transform = torch.stft(
            input_data,
            self.filter_length, self.hop_length, self.win_length, window=self.window,
            return_complex=True)

        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude, phase):
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length, self.hop_length, self.win_length, window=self.window)

        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return