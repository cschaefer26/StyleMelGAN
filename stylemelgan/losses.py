# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
# adapted from https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/losses/stft_loss.py

import torch

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

