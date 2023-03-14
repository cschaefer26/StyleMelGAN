from typing import Tuple, Dict, Any
import math
import torch
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Tanh

from stylemelgan.common import WNConv1d, WNConvTranspose1d
from stylemelgan.utils import read_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAX_WAV_VALUE = 32768.0


import torch
from torch import nn


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=200000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale_1 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.scale_2 = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.scale_3 = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * 0.0001)
        pe[:, 1::2] = torch.cos(position * 0.0001)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, shift=0) -> torch.Tensor:
        x = x.transpose(1, 2)# shape: [T, N]
        x = x + self.scale_3 * self.pe[:x.size(0), :]
        return self.dropout(x).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x


class FNet(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pe = PositionalEncoding(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        x = self.pe(x)
        x = x.transpose(1, 2)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.transpose(1, 2)
        return x


class ResStack(nn.Module):
    def __init__(self, channel, num_layers=4):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(num_layers)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))
            for i in range(num_layers)
        ])

    def forward(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        self.generator = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mel_channel, 256, kernel_size=7, stride=1)),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4)),

            FNet(128, 6, mlp_dim=128),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4)),

            FNet(64, 6, mlp_dim=64),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            FNet(32, 6, mlp_dim=32),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)),

            FNet(16, 6, mlp_dim=16),

            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(16, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self,
                  mel: torch.Tensor,
                  pad_steps: int = 10) -> torch.Tensor:
        with torch.no_grad():
            pad = torch.full((1, 80, pad_steps), -11.5129).to(mel.device)
            mel = torch.cat((mel, pad), dim=2)
            audio = self.forward(mel).squeeze()
            audio = audio[:-(256 * pad_steps)]
        return audio

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Generator':
        return Generator(mel_channels=config['audio']['n_mels'],
                               **config['model'])

    @classmethod
    def from_checkpoint(cls, file: str) -> 'Generator':
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        config = checkpoint['config']
        model = Generator.from_config(config)
        model.load_state_dict(config['g_model'])
        return model


if __name__ == '__main__':
    import time
    config = read_config('../configs/melgan_config.yaml')
    model = Generator(80)
    x = torch.randn(3, 80, 1000)
    start = time.time()
    for i in range(1):
        y = model(x)
    dur = time.time() - start

    print('dur ', dur)

    #y = model(x)
    #print(y.shape)
    #assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
