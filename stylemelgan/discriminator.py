import torch
import torch.nn as nn
from torch.nn import Sequential, LeakyReLU

from stylemelgan.common import WNConv1d


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad1d(7),
                nn.utils.weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])

    def forward(self, x):
        '''
            returns: (list of 6 features, discriminator score)
            we directly predict score without last sigmoid function
            since we're using Least Squares GAN (https://arxiv.org/abs/1611.04076)
        '''
        features = list()
        for module in self.discriminator:
            x = module(x)
            features.append(x)
        return features[:-1], features[-1]


class RNNDiscriminator(nn.Module):

    def __init__(self, dim_in=1024, dim=256):
        super(RNNDiscriminator, self).__init__()
        self.conv_in = nn.utils.weight_norm(nn.Conv1d(dim_in, dim, kernel_size=1, stride=1))
        self.rnn = nn.GRU(dim, dim, bidirectional=True)
        self.conv_out = nn.utils.weight_norm(nn.Conv1d(2 * dim, 1, kernel_size=1, stride=1))

    def forward(self, x):
        x = self.conv_in(x)
        x, _ = self.rnn(x.transpose(1, 2))
        x = self.conv_out(x.transpose(1, 2))
        return x


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


class MultiScaleRNNDiscriminator(nn.Module):

    def __init__(self):
        super(MultiScaleRNNDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList(
            [RNNDiscriminator() for _ in range(3)]
        )

    def forward(self, feats):
        ret = list()

        for feat, disc in zip(feats, self.discriminators):
            ret.append((None, disc(feat)))

        return ret  # [(feat, score), (feat, score), (feat, score)]



if __name__ == '__main__':

    x = torch.randn(3, 1, 22050)

    model = MultiScaleDiscriminator()
    out = model(x)

    feats = [o[0][-1] for o in out]

    rnn_disc = MultiScaleRNNDiscriminator()
    feat_out = rnn_disc(feats)

    print(feat_out)

    exit()
    print(x.shape)

    for feat in features:
        print(feat.shape)
    print(score.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)