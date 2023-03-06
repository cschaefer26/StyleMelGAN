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


class KernelPredictor(torch.nn.Module):
    ''' Kernel predictor for the location-variable convolutions'''
    def __init__(
            self,
            cond_channels,
            conv_in_channels,
            conv_out_channels,
            conv_layers,
            conv_kernel_size=3,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
            kpnet_nonlinear_activation="LeakyReLU",
            kpnet_nonlinear_activation_params={"negative_slope":0.1},
    ):
        '''
        Args:
            cond_channels (int): number of channel for the conditioning sequence,
            conv_in_channels (int): number of channel for the input sequence,
            conv_out_channels (int): number of channel for the output sequence,
            conv_layers (int): number of layers
        '''
        super().__init__()

        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kpnet_kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers   # l_w
        kpnet_bias_channels = conv_out_channels * conv_layers                                           # l_b

        self.input_conv = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(cond_channels, kpnet_hidden_channels, 5, padding=2, bias=True)),
            getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
        )

        self.residual_convs = nn.ModuleList()
        padding = (kpnet_conv_size - 1) // 2
        for _ in range(3):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Dropout(kpnet_dropout),
                    nn.utils.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                    nn.utils.weight_norm(nn.Conv1d(kpnet_hidden_channels, kpnet_hidden_channels, kpnet_conv_size, padding=padding, bias=True)),
                    getattr(nn, kpnet_nonlinear_activation)(**kpnet_nonlinear_activation_params),
                )
            )
        self.kernel_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_kernel_channels, kpnet_conv_size, padding=padding, bias=True))
        self.bias_conv = nn.utils.weight_norm(
            nn.Conv1d(kpnet_hidden_channels, kpnet_bias_channels, kpnet_conv_size, padding=padding, bias=True))

    def forward(self, c):
        '''
        Args:
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)
        '''
        batch, _, cond_length = c.shape
        c = self.input_conv(c)
        for residual_conv in self.residual_convs:
            residual_conv.to(c.device)
            c = c + residual_conv(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = k.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            cond_length,
        )
        bias = b.contiguous().view(
            batch,
            self.conv_layers,
            self.conv_out_channels,
            cond_length,
        )

        return kernels, bias

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv[0])
        nn.utils.remove_weight_norm(self.kernel_conv)
        nn.utils.remove_weight_norm(self.bias_conv)
        for block in self.residual_convs:
            nn.utils.remove_weight_norm(block[1])
            nn.utils.remove_weight_norm(block[3])

class LVCBlock(torch.nn.Module):
    '''the location-variable convolutions'''
    def __init__(
            self,
            in_channels,
            out_channels,
            cond_channels,
            stride,
            dilations,
            lReLU_slope=0.2,
            conv_kernel_size=3,
            cond_hop_length=256,
            kpnet_hidden_channels=64,
            kpnet_conv_size=3,
            kpnet_dropout=0.0,
    ):
        super().__init__()

        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size

        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=out_channels,
            conv_out_channels=2 * out_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            kpnet_dropout=kpnet_dropout,
            kpnet_nonlinear_activation_params={"negative_slope":lReLU_slope}
        )


        self.convt_pre = nn.Sequential(
            nn.LeakyReLU(lReLU_slope),
            nn.utils.weight_norm(nn.ConvTranspose1d(in_channels, out_channels, 2 * stride, stride=stride, padding=stride // 2 + stride % 2, output_padding=stride % 2)),
        )

        self.conv_blocks = nn.ModuleList()
        for dilation in dilations:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(lReLU_slope),
                    nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, conv_kernel_size, padding=dilation * (conv_kernel_size - 1) // 2, dilation=dilation)),
                    nn.LeakyReLU(lReLU_slope),
                )
            )

    def forward(self, x, c):
        ''' forward propagation of the location-variable convolutions.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length)
            c (Tensor): the conditioning sequence (batch, cond_channels, cond_length)

        Returns:
            Tensor: the output sequence (batch, in_channels, in_length)
        '''
        _, in_channels, _ = x.shape         # (B, c_g, L')

        x = self.convt_pre(x)               # (B, c_g, stride * L')

        _, out_channels, _ = x.shape
        kernels, bias = self.kernel_predictor(c)

        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)                # (B, c_g, stride * L')

            k = kernels[:, i, :, :, :, :]   # (B, 2 * c_g, c_g, kernel_size, cond_length)
            b = bias[:, i, :, :]            # (B, 2 * c_g, cond_length)

            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)    # (B, 2 * c_g, stride * L'): LVC
            x = x + torch.sigmoid(output[ :, :out_channels, :]) * torch.tanh(output[:, out_channels:, :]) # (B, c_g, stride * L'): GAU

        return x

    def location_variable_convolution(self, x, kernel, bias, dilation=1, hop_size=256):
        ''' perform location-variable convolution operation on the input sequence (x) using the local convolution kernl.
        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.
        Args:
            x (Tensor): the input sequence (batch, in_channels, in_length).
            kernel (Tensor): the local convolution kernel (batch, in_channel, out_channels, kernel_size, kernel_length)
            bias (Tensor): the bias for the local convolution (batch, out_channels, kernel_length)
            dilation (int): the dilation of convolution.
            hop_size (int): the hop_size of the conditioning sequence.
        Returns:
            (Tensor): the output sequence after performing local convolution. (batch, out_channels, in_length).
        '''
        batch, _, in_length = x.shape
        batch, _, out_channels, kernel_size, kernel_length = kernel.shape
        assert in_length == (kernel_length * hop_size), "length of (x, kernel) is not matched"

        padding = dilation * int((kernel_size - 1) / 2)
        x = F.pad(x, (padding, padding), 'constant', 0)     # (batch, in_channels, in_length + 2*padding)
        x = x.unfold(2, hop_size + 2 * padding, hop_size)   # (batch, in_channels, kernel_length, hop_size + 2*padding)

        if hop_size < dilation:
            x = F.pad(x, (0, dilation), 'constant', 0)
        x = x.unfold(3, dilation, dilation)     # (batch, in_channels, kernel_length, (hop_size + 2*padding)/dilation, dilation)
        x = x[:, :, :, :, :hop_size]
        x = x.transpose(3, 4)                   # (batch, in_channels, kernel_length, dilation, (hop_size + 2*padding)/dilation)
        x = x.unfold(4, kernel_size, 1)         # (batch, in_channels, kernel_length, dilation, _, kernel_size)

        o = torch.einsum('bildsk,biokl->bolsd', x, kernel)
        o = o.to(memory_format=torch.channels_last_3d)
        bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
        o = o + bias
        o = o.contiguous().view(batch, out_channels, -1)

        return o

    def remove_weight_norm(self):
        self.kernel_predictor.remove_weight_norm()
        nn.utils.remove_weight_norm(self.convt_pre[1])
        for block in self.conv_blocks:
            nn.utils.remove_weight_norm(block[1])


class Generator(nn.Module):
    """UnivNet Generator"""
    def __init__(self, hp):
        super(Generator, self).__init__()
        self.mel_channel = 80
        self.noise_dim = 64
        self.hop_length = 256

        self.res_stack = nn.ModuleList()

        c1, c2, c3, c4, c5 = [128, 64, 32, 32, 32]

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(64, c1, 7, padding=3, padding_mode='reflect'))


        self.up_1 = LVCBlock(
            c1, c2, 80, stride=8, dilations=[1, 3, 9, 27, 81], lReLU_slope=0.2,
            cond_hop_length=8, kpnet_conv_size=3
        )

        self.up_2 = LVCBlock(
            c2, c3, 80, stride=8, dilations=[1, 3, 9, 27, 81], lReLU_slope=0.2,
            cond_hop_length=64, kpnet_conv_size=3
        )

        self.up_3 = LVCBlock(
            c3, c4, 80, stride=2, dilations=[1, 3, 9, 27, 81], lReLU_slope=0.2,
            cond_hop_length=128, kpnet_conv_size=3
        )

        self.up_4 = LVCBlock(
            c4, c5, 80, stride=2, dilations=[1, 3, 9, 27, 81], lReLU_slope=0.2,
            cond_hop_length=256, kpnet_conv_size=3
        )

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(c5, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, c):

        z = torch.randn(c.size(0), self.noise_dim, c.size(2)).to(c.device)
        '''
        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)
            z (Tensor): the noise sequence (batch, noise_dim, in_length)

        '''
        z = self.conv_pre(z)                # (B, c_g, L)

        z = self.up_1(z, c)
        z = self.up_2(z, c)
        z = self.up_3(z, c)
        z = self.up_4(z, c)

        z = self.conv_post(z)               # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        with torch.no_grad():
            # pad input mel with zeros to cut artifact
            # see https://github.com/seungwonpark/melgan/issues/8
            zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
            mel = torch.cat((c, zero), dim=2)

            audio = self.forward(mel)
            audio = audio.squeeze() # collapse all dimension except time axis
            audio = audio[:-(self.hop_length*10)]
            #audio = MAX_WAV_VALUE * audio
            #audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
            #audio = audio.short()

        return audio


if __name__ == '__main__':
    import time
    config = read_config('../configs/melgan_config.yaml')
    model = Generator(80)
    x = torch.randn(3, 80, 1000)
    start = time.time()
    for i in range(1):
        y = model.forward(x)
    dur = time.time() - start

    print('dur ', dur)

    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 256000])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
