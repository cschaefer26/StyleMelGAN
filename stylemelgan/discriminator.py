import torch
import torch.nn as nn
from torch.nn import Sequential, LeakyReLU
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F
from stylemelgan.common import WNConv1d
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d, Linear, GRU
import numpy as np

from scipy import signal as sig
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
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

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


class CoMBD(torch.nn.Module):

    def __init__(self, filters, kernels, groups, strides, use_spectral_norm=False):
        super(CoMBD, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList()
        init_channel = 1
        for i, (f, k, g, s) in enumerate(zip(filters, kernels, groups, strides)):
            self.convs.append(norm_f(Conv1d(init_channel, f, k, s, padding=get_padding(k, 1), groups=g)))
            init_channel = f
        self.conv_post = norm_f(Conv1d(filters[-1], 1, 3, 1, padding=get_padding(3, 1)))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        #fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MDC(torch.nn.Module):

    def __init__(self, in_channel, channel, kernel, stride, dilations, use_spectral_norm=False):
        super(MDC, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch.nn.ModuleList()
        self.num_dilations = len(dilations)
        for d in dilations:
            self.convs.append(norm_f(Conv1d(in_channel, channel, kernel, stride=1, padding=get_padding(kernel, d),
                                            dilation=d)))

        self.conv_out = norm_f(Conv1d(channel, channel, 3, stride=stride, padding=get_padding(3, 1)))

    def forward(self, x):
        xs = None
        for l in self.convs:
            if xs is None:
                xs = l(x)
            else:
                xs += l(x)

        x = xs / self.num_dilations

        x = self.conv_out(x)
        x = F.leaky_relu(x, 0.1)
        return x


class SubBandDiscriminator(torch.nn.Module):

    def __init__(self, init_channel, channels, kernel, strides, dilations, use_spectral_norm=False):
        super(SubBandDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.mdcs = torch.nn.ModuleList()

        for c, s, d in zip(channels, strides, dilations):
            self.mdcs.append(MDC(init_channel, c, kernel, s, d))
            init_channel = c
        self.conv_post = norm_f(Conv1d(init_channel, 1, 3, padding=get_padding(3, 1)))

    def forward(self, x):
        fmap = []

        for l in self.mdcs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        #fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/parallel_wavegan
class PQMF(torch.nn.Module):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))  # TODO: (taps - 1) -> taps
            phase = (-1)**k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)

            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x,
                               self.updown_filter * self.N,
                               stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x




tkernels = [7, 5, 3]
fkernel = 5
tchannels = [64, 128, 256, 256, 256]
fchannels = [32, 64, 128, 128, 128]
tstrides = [[1, 1, 3, 3, 1], [1, 1, 3, 3, 1], [1, 1, 3, 3, 1]]
fstride = [1, 1, 3, 3, 1]
tdilations = [[[5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11], [5, 7, 11]], [[3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7], [3, 5, 7]], [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]]
fdilations = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [2, 3, 5], [2, 3, 5]]
pqmf_n = 16
pqmf_m = 64
freq_init_ch = 128
tsubband = [6, 11, 16]


class MultiSubBandDiscriminator(torch.nn.Module):

    def __init__(self):

        super(MultiSubBandDiscriminator, self).__init__()

        self.fsbd = SubBandDiscriminator(init_channel=freq_init_ch, channels=fchannels, kernel=fkernel,
                                         strides=fstride, dilations=fdilations)

        self.tsubband1 = tsubband[0]
        self.tsbd1 = SubBandDiscriminator(init_channel=self.tsubband1, channels=tchannels, kernel=tkernels[0],
                                          strides=tstrides[0], dilations=tdilations[0])

        self.tsubband2 = tsubband[1]
        self.tsbd2 = SubBandDiscriminator(init_channel=self.tsubband2, channels=tchannels, kernel=tkernels[1],
                                          strides=tstrides[1], dilations=tdilations[1])

        self.tsubband3 = tsubband[2]
        self.tsbd3 = SubBandDiscriminator(init_channel=self.tsubband3, channels=tchannels, kernel=tkernels[2],
                                          strides=tstrides[2], dilations=tdilations[2])

        self.pqmf_n = PQMF(N=pqmf_n, taps=256, cutoff=0.03, beta=10.0)
        #self.pqmf_m = PQMF(N=pqmf_m, taps=256, cutoff=0.1, beta=9.0)

    def forward(self, x):
        features = []

        # Time analysis
        xn = self.pqmf_n(x)

        q3, feat_q3 = self.tsbd3(xn[:, :self.tsubband3, :])
        features.append((feat_q3, q3))

        q2, feat_q2 = self.tsbd2(xn[:, :self.tsubband2, :])
        features.append((feat_q2, q2))

        q1, feat_q1 = self.tsbd1(xn[:, :self.tsubband1, :])

        features.append((feat_q1, q1))

        # Frequency analysis
        #xm = self.pqmf_m.analysis(x)

        #xm = xm.transpose(-2, -1)

        #q4, feat_q4 = self.fsbd(xm)
        #features.append((feat_q4, q4))

        return features




if __name__ == '__main__':
    model = MultiSubBandDiscriminator()

    x = torch.randn(3, 1, 22050)
    print(x.shape)

    res = model(x)
    print(res)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)