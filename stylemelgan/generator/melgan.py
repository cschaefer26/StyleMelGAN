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

from functools import partial


class TADELayer(torch.nn.Module):
    """TADE Layer module."""

    def __init__(
            self,
            in_channels=64,
            aux_channels=80,
            kernel_size=9,
            bias=True,
            upsample_factor=2,
            upsample_mode="nearest",
    ):
        """Initilize TADE layer."""
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(in_channels)
        self.aux_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                aux_channels,
                in_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.gated_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                in_channels * 2,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
                ),
            # NOTE(kan-bayashi): Use non-linear activation?
        )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').
        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled aux tensor (B, in_channels, T * aux_upsample_factor).
        """
        x = self.norm(x)
        c = self.upsample(c)
        c = self.aux_conv(c)
        cg = self.gated_conv(c)
        cg1, cg2 = cg.split(cg.size(1) // 2, dim=1)
        # NOTE(kan-bayashi): Use upsample for noise input here?
        y = cg1 * self.upsample(x) + cg2
        # NOTE(kan-bayashi): Return upsampled aux here?
        return y, c


class TADEResBlock(torch.nn.Module):
    """TADEResBlock module."""

    def __init__(
            self,
            in_channels=64,
            aux_channels=80,
            kernel_size=9,
            dilation=2,
            bias=True,
            upsample_factor=2,
            upsample_mode="nearest",
            gated_function="softmax",
    ):
        """Initialize TADEResBlock module."""
        super().__init__()
        self.tade1 = TADELayer(
            in_channels=in_channels,
            aux_channels=aux_channels,
            kernel_size=kernel_size,
            bias=bias,
            # NOTE(kan-bayashi): Use upsample in the first TADE layer?
            upsample_factor=1,
            upsample_mode=upsample_mode,
        )
        self.gated_conv1 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            padding=(kernel_size - 1) // 2,
            )
        self.tade2 = TADELayer(
            in_channels=in_channels,
            aux_channels=in_channels,
            kernel_size=kernel_size,
            bias=bias,
            upsample_factor=upsample_factor,
            upsample_mode=upsample_mode,
        )
        self.gated_conv2 = torch.nn.Conv1d(
            in_channels,
            in_channels * 2,
            kernel_size,
            1,
            bias=bias,
            dilation=dilation,
            padding=(kernel_size - 1) // 2 * dilation,
            )
        self.upsample = torch.nn.Upsample(
            scale_factor=upsample_factor, mode=upsample_mode
        )
        if gated_function == "softmax":
            self.gated_function = partial(torch.softmax, dim=1)
        elif gated_function == "sigmoid":
            self.gated_function = torch.sigmoid
        else:
            raise ValueError(f"{gated_function} is not supported.")

    def forward(self, x, c):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            c (Tensor): Auxiliary input tensor (B, aux_channels, T').
        Returns:
            Tensor: Output tensor (B, in_channels, T * in_upsample_factor).
            Tensor: Upsampled auxirialy tensor (B, in_channels, T * in_upsample_factor).
        """
        residual = x

        x, c = self.tade1(x, c)
        x = self.gated_conv1(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        x, c = self.tade2(x, c)
        x = self.gated_conv2(x)
        xa, xb = x.split(x.size(1) // 2, dim=1)
        x = self.gated_function(xa) * torch.tanh(xb)

        # NOTE(kan-bayashi): Return upsampled aux here?
        return self.upsample(residual) + x, c

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


class Generator(torch.nn.Module):
    """Style MelGAN generator module."""

    def __init__(
            self,
            in_channels=128,
            aux_channels=80,
            channels=64,
            out_channels=1,
            kernel_size=9,
            dilation=2,
            bias=True,
            noise_upsample_scales=[11, 2, 2, 2],
            noise_upsample_activation="LeakyReLU",
            noise_upsample_activation_params={"negative_slope": 0.2},
            upsample_scales=[2, 2, 2, 2, 2, 2, 2, 2, 1],
            upsample_mode="nearest",
            gated_function="softmax",
            use_weight_norm=True,
    ):
        """Initilize Style MelGAN generator.
        Args:
            in_channels (int): Number of input noise channels.
            aux_channels (int): Number of auxiliary input channels.
            channels (int): Number of channels for conv layer.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of conv layers.
            dilation (int): Dilation factor for conv layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            noise_upsample_scales (list): List of noise upsampling scales.
            noise_upsample_activation (str): Activation function module name for noise upsampling.
            noise_upsample_activation_params (dict): Hyperparameters for the above activation function.
            upsample_scales (list): List of upsampling scales.
            upsample_mode (str): Upsampling mode in TADE layer.
            gated_function (str): Gated function in TADEResBlock ("softmax" or "sigmoid").
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        self.in_channels = in_channels

        noise_upsample = []
        in_chs = in_channels
        for noise_upsample_scale in noise_upsample_scales:
            # NOTE(kan-bayashi): How should we design noise upsampling part?
            noise_upsample += [
                torch.nn.ConvTranspose1d(
                    in_chs,
                    channels,
                    noise_upsample_scale * 2,
                    stride=noise_upsample_scale,
                    padding=noise_upsample_scale // 2 + noise_upsample_scale % 2,
                    output_padding=noise_upsample_scale % 2,
                    bias=bias,
                    )
            ]
            noise_upsample += [
                getattr(torch.nn, noise_upsample_activation)(
                    **noise_upsample_activation_params
                )
            ]
            in_chs = channels
        self.noise_upsample = torch.nn.Sequential(*noise_upsample)
        self.noise_upsample_factor = np.prod(noise_upsample_scales)

        self.blocks = torch.nn.ModuleList()
        aux_chs = aux_channels
        for upsample_scale in upsample_scales:
            self.blocks += [
                TADEResBlock(
                    in_channels=channels,
                    aux_channels=aux_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    bias=bias,
                    upsample_factor=upsample_scale,
                    upsample_mode=upsample_mode,
                    gated_function=gated_function,
                ),
            ]
            aux_chs = channels
        self.upsample_factor = np.prod(upsample_scales)

        self.output_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                channels,
                out_channels,
                kernel_size,
                1,
                bias=bias,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Tanh(),
        )

    def forward(self, c, z=None):

        c = (c + 5.0) / 5.0 # roughly normalize spectrogram
        """Calculate forward propagation.
        Args:
            c (Tensor): Auxiliary input tensor (B, channels, T).
            z (Tensor): Input noise tensor (B, in_channels, 1).
        Returns:
            Tensor: Output tensor (B, out_channels, T ** prod(upsample_scales)).
        """
        if z is None:
            noise_size = (
                1,
                self.in_channels,
                (c.size(2) - 1) // self.noise_upsample_factor + 1,
            )
            z = torch.randn(*noise_size, dtype=torch.float).to(
                next(self.parameters()).device
            )
        x = self.noise_upsample(z)[:, :, :c.size(-1)]
        for block in self.blocks:
            x, c = block(x, c)
        x = self.output_conv(x)
        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:

                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def inference(self, c, normalize_before=False):
        return self.forward(c).detach().cpu()



if __name__ == '__main__':
    import time
    config = read_config('../configs/melgan_config.yaml')
    model = Generator(80)
    model.eval()
    x = torch.randn(3, 80, 1000)
    start = time.time()
    for i in range(1):
        y = model.inference(x)
    dur = time.time() - start

    print('dur ', dur)

    y = model(x)
    print(y.shape)
    assert y.shape == torch.Size([3, 1, 256000])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
