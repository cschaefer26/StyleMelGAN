import torch
import torch.nn as nn
from torch.nn import Sequential, LeakyReLU
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F
from stylemelgan.common import WNConv1d
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, Linear, GRU

LRELU_SLOPE = 0.1

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class PitchPredictor(nn.Module):

    def __init__(self, relu_slope: float = 0.2):
        super(PitchPredictor, self).__init__()

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
        for module in self.discriminator:
            x = module(x)
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


class DiscriminatorR(nn.Module):
    def __init__(self, resolution):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = LRELU_SLOPE

        norm_f = weight_norm
        self.d_mult = 1

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1) #[B, F, TT]

        return mag


class MultiSpecDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorR([1024, 120, 600]),
            DiscriminatorR([2048, 240, 1200]),
            DiscriminatorR([512, 50, 240]),
        ])

    def forward(self, y):
        ret = list()
        for i, d in enumerate(self.discriminators):
            ret.append(d(y))

        return ret


if __name__ == '__main__':
    model = MultiSpecDiscriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    res = model(x)
    print(res)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)