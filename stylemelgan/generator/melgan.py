import time

import torch
from torch.nn import Module, Conv1d, InstanceNorm1d, LeakyReLU, Linear
from torch.nn.utils import spectral_norm, weight_norm
from torch.nn.functional import interpolate, relu
import torch.nn.functional as F
import torch.nn as nn
MAX_WAV_VALUE = 32768.0


class ResStack(nn.Module):
    def __init__(self, in_channel, channel, num_layers=4):
        super(ResStack, self).__init__()

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3**i),
                nn.utils.weight_norm(nn.Conv1d(in_channel if i == 0 else channel, channel, kernel_size=3, dilation=3**i)),
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
            )
            for i in range(num_layers)
        ])

        self.shortcuts = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(in_channel if i == 0 else channel, channel, kernel_size=1))
            for i in range(num_layers)
        ])

    def forward(self, x, mel):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            x = shortcut(x) + block(x)
        return x, mel

    def remove_weight_norm(self):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            nn.utils.remove_weight_norm(block[2])
            nn.utils.remove_weight_norm(block[4])
            nn.utils.remove_weight_norm(shortcut)


class TadeUp(Module):

    def __init__(self, mel_channels, channels: int, kernel_size=3):
        super().__init__()
        self.norm = InstanceNorm1d(channels)
        self.relu = LeakyReLU(0.2)
        self.conv_feat = weight_norm(Conv1d(mel_channels, channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='reflect'))
        self.conv_beta = weight_norm(Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='reflect'))
        self.conv_gamma = weight_norm(Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2, padding_mode='reflect'))

    def forward(self, x, mel):
        mel = F.interpolate(mel, scale_factor=2, mode='nearest')
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        feat = self.conv_feat(mel)
        feat = self.relu(feat)
        gamma = self.conv_gamma(feat)
        beta = self.conv_beta(feat)
        c = self.norm(x)
        x = c * gamma + beta
        return x, mel


class Generator(nn.Module):
    def __init__(self, mel_channel):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel
        self.first_conv = nn.Conv1d(mel_channel, 256,
                                    kernel_size=7, padding=3, padding_mode='reflect')

        self.tades = nn.ModuleList([
            TadeUp(mel_channel, 256),
            TadeUp(mel_channel, 256),
            TadeUp(mel_channel, 256),
            ResStack(256, 128, num_layers=5),
            TadeUp(mel_channel, 128),
            TadeUp(mel_channel, 128),
            TadeUp(mel_channel, 128),
            ResStack(128, 64, num_layers=7),
            TadeUp(mel_channel, 64),
            ResStack(64, 32, num_layers=8),
            TadeUp(mel_channel, 32),
            ResStack(32, 32, num_layers=9)
        ])
        self.postnet = nn.Sequential(
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv1d(32, 1, kernel_size=7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def _forward(self, mel):
        mel = mel.detach()
        x = self.first_conv(mel)
        for tade in self.tades:
            x, mel = tade(x, mel)
        x = self.postnet(x)
        return x

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0 # roughly normalize spectrogram
        return self._forward(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.tades):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()
        for idx, layer in enumerate(self.postnet):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        hop_length = 256
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)
        audio = self.forward(mel)
        audio = audio.squeeze() # collapse all dimension except time axis
        audio = audio[:-(hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()

        return audio



if __name__ == '__main__':

    #tade = TadeUp(mel_channels=80, channels=128)
    generator = Generator(80)
    start = time.time()
    mel = torch.randn(3, 80, 1000)
    x = torch.randn(3, 128, 1000)
    y = generator(mel)
    print(y.size())

    dur = time.time() - start
    print(x.shape)
    print(y.shape)
    print(f'dur: {dur}')
    #assert y.shape == torch.Size([3, 1, 2560])

