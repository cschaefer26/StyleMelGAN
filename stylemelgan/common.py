import torch
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Conv1d, ConvTranspose1d, Tanh
from torch.nn.utils import weight_norm


class WNConv1d(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 padding_mode: str = 'zeros') -> None:
        super().__init__()
        conv = Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride,
                      dilation=dilation, padding=padding,
                      padding_mode=padding_mode, groups=groups)
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WNConvTranspose1d(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0) -> None:
        super().__init__()
        conv = ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)