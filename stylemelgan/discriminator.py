import torch
import torch.nn as nn
from torch.nn import Sequential, LeakyReLU, GRU
from torch.nn import Module
from distutils.version import LooseVersion
from stylemelgan.common import WNConv1d


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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



class PitchPredictor(nn.Module):

    def __init__(self, relu_slope: float = 0.2):
        super(PitchPredictor, self).__init__()

        self.convs = nn.Sequential(
            WNConv1d(513, 512, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            LeakyReLU(relu_slope, inplace=True),
            WNConv1d(512, 512, kernel_size=3, stride=1, padding=1),
            LeakyReLU(relu_slope, inplace=True),
            WNConv1d(512, 512, kernel_size=3, stride=1, padding=1)
        )
        self.rnn = GRU(512, 256, bidirectional=True, batch_first=True)
        self.out_conv = WNConv1d(512, 1, 1)

    def forward(self, x):
        x_stft = stft(x=x.squeeze(1), n_fft=1024, hop_length=256, win_length=1024)
        x = self.convs(x_stft.transpose(1, 2))
        x, _ = self.rnn(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = self.out_conv(x)
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



if __name__ == '__main__':
    model = PitchPredictor()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    out = model(x)

    print(out.size())