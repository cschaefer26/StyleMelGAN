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
    def __init__(self,
                 params=[
                     (1, 16, 15, 1, 1),
                     (16, 64, 41, 4, 4),
                     (64, 256, 41, 4, 16),
                     (256, 1024, 41, 4, 64),
                     (1024, 1024, 41, 4, 256),
                     (1024, 1024, 5, 1, 1),
                     (1024, 1, 3, 1, 1)
                 ]
                 ):
        super(Discriminator, self).__init__()

        in_dim, out_dim, kernel, stride, groups = params[0]
        modules = [
            nn.Sequential(
                nn.ReflectionPad1d(kernel//2),
                nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ]
        for in_dim, out_dim, kernel, stride, groups in params[1:-1]:
            modules.append(
                nn.Sequential(
                    nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups)),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
        in_dim, out_dim, kernel, stride, groups = params[-1]
        modules.append(
            nn.utils.weight_norm(nn.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, padding=kernel//2, groups=groups)),
        )
        self.discriminator = nn.ModuleList(modules)

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


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()

        params_1 = [(1, 16, 15, 1, 1),
                    (16, 64, 41, 4, 4),
                    (64, 256, 41, 4, 16),
                    (256, 1024, 41, 4, 64),
                    (1024, 1024, 41, 4, 256),
                    (1024, 1024, 5, 1, 1),
                    (1024, 1, 3, 1, 1)]

        params_2 = [(1, 16, 15, 1, 1),
                    (16, 64, 41, 2, 4),
                    (64, 256, 41, 2, 16),
                    (256, 1024, 41, 8, 64),
                    (1024, 1024, 41, 8, 256),
                    (1024, 1024, 5, 1, 1),
                    (1024, 1, 3, 1, 1)]

        params_3 = [(1, 16, 15, 1, 1),
                    (16, 64, 41, 6, 4),
                    (64, 256, 41, 6, 16),
                    (256, 1024, 41, 6, 64),
                    (1024, 1024, 41, 6, 256),
                    (1024, 1024, 5, 1, 1),
                    (1024, 1, 3, 1, 1)]

        self.disc_1 = nn.ModuleList(
            [Discriminator(params_1) for _ in range(3)]
        )

        self.disc_2 = nn.ModuleList(
            [Discriminator(params_2) for _ in range(3)]
        )

        self.disc_3 = nn.ModuleList(
            [Discriminator(params_3) for _ in range(3)]
        )

        self.pooling_1 = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

        self.pooling_2 = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )

        self.pooling_3 = nn.ModuleList(
            [Identity()] +
            [nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False) for _ in range(1, 3)]
        )


    def forward(self, x):
        ret = list()

        x_p = x
        for pool, disc in zip(self.pooling_1, self.disc_1):
            x_p = pool(x_p)
            ret.append(disc(x_p))

        x_p = x
        for pool, disc in zip(self.pooling_2, self.disc_2):
            x_p = pool(x_p)
            ret.append(disc(x_p))

        x_p = x
        for pool, disc in zip(self.pooling_3, self.disc_3):
            x_p = pool(x_p)
            ret.append(disc(x_p))

        return ret  # [(feat, score), (feat, score), (feat, score)]



if __name__ == '__main__':
    model = MultiScaleDiscriminator()

    x = torch.randn(3, 1, 22050)

    out = model(x)

    print(out)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)