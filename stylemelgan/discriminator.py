import torch
import torch.nn as nn
from torch.nn import Sequential, LeakyReLU

from stylemelgan.common import WNConv1d
from stylemelgan.losses import stft


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Discriminator(nn.Module):

    def __init__(self, relu_slope: float = 0.2):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            Sequential(
                WNConv1d(1, 16, kernel_size=15, stride=1, padding=7, padding_mode='reflect'),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
                LeakyReLU(relu_slope, inplace=True)
            ),
            WNConv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        features = []
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [Discriminator() for _ in range(3)]
        )

        self.pooling = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

    def forward(self, x):
        ret = list()

        for pool, disc in zip(self.pooling, self.discriminators):
            x = pool(x)
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]


class SpecDiscriminator(nn.Module):

    def __init__(self, n_fft: int, relu_slope: float = 0.2):
        super().__init__()
        self.discriminator = nn.ModuleList([
            Sequential(
                WNConv1d(n_fft, 1024, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(1024, 1024, kernel_size=7, stride=2, padding=3, groups=16),
                LeakyReLU(relu_slope, inplace=True)
            ),
            Sequential(
                WNConv1d(1024, 1024, kernel_size=7, stride=2, padding=3, groups=16),
                LeakyReLU(relu_slope, inplace=True)
            ),
            WNConv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        ])

    def forward(self, x):
        features = []
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


class MultiScaleSpecDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.n_ffts = [1024, 2048, 512]
        self.hop_sizes = [120, 240, 50]
        self.win_lengths = [600, 1200, 240]

        self.discriminators = nn.ModuleList(
            [SpecDiscriminator(n_fft // 2 + 1) for n_fft in self.n_ffts]
        )

    def forward(self, x):
        ret = list()
        for n_fft, hop_length, win_length, disc in zip(self.n_ffts, self.hop_sizes, self.win_lengths, self.discriminators):
            x_stft = stft(x=x.squeeze(1), n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            x_stft = torch.log(x_stft)
            ret.append(disc(x_stft.transpose(1, 2)))

        return ret  # [(feat, score), (feat, score), (feat, score)]




if __name__ == '__main__':
    model = MultiScaleSpecDiscriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    features, score = model(x)
    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)