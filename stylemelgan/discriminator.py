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


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y):
        ret = list()
        for i, d in enumerate(self.discriminators):
            ret.append(d(y))

        return ret


if __name__ == '__main__':
    model = MultiPeriodDiscriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    res = model(x)
    print(res)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)