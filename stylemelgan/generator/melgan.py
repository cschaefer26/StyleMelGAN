from typing import Tuple, Dict, Any

import torch
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Tanh

from stylemelgan.common import WNConv1d, WNConvTranspose1d
from stylemelgan.utils import read_config
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAX_WAV_VALUE = 32768.0



class ResStack(nn.Module):
    def __init__(self, channel, num_layers=4):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(2**i),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=2**i)),
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

            ResStack(128, num_layers=7),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4)),

            ResStack(64, num_layers=10),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)),

            ResStack(32, num_layers=11),

            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)),

            ResStack(16, num_layers=12),

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

    config = read_config('../configs/melgan_config.yaml')
    model = Generator(80)
    x = torch.randn(3, 80, 1000)
    print(x.shape)

    y = model(x)
    print(y.shape)
    #assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
