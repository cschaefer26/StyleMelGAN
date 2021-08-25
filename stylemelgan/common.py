import torch
from torch.nn import Module, Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm


class WNConv1d(Module):

    def __init__(self, *args, **kwargs,) -> None:
        super().__init__()
        conv = Conv1d(*args, **kwargs)
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class WNConvTranspose1d(Module):

    def __init__(self, *args, **kwargs,) -> None:
        super().__init__()
        conv = ConvTranspose1d(*args, **kwargs)
        self.conv = weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)