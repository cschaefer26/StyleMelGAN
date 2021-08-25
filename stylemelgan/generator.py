from typing import Tuple

import torch
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Conv1d, ConvTranspose1d, Tanh
from torch.nn.utils import weight_norm

from stylemelgan.common import WNConv1d, WNConvTranspose1d


class ResBlock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dilation: int,
                 relu_slope: float = 0.2):
        super().__init__()
        self.conv_block = Sequential(
            LeakyReLU(relu_slope),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                     kernel_size=3, dilation=dilation, padding=dilation,
                     padding_mode='reflect'),
            LeakyReLU(relu_slope),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                     kernel_size=1)
        )
        self.residual = WNConv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.conv_block(x)


class ResStack(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int,
                 relu_slope: float = 0.2):
        super().__init__()

        self.res_blocks = ModuleList([
            ResBlock(in_channels=in_channels if i == 0 else out_channels,
                     out_channels=out_channels,
                     dilation=3 ** i,
                     relu_slope=relu_slope)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for res_block in self.res_blocks:
            x = res_block(x)
        return x


class MelganGenerator(Module):

    def __init__(self,
                 mel_channels: int,
                 channels: Tuple = (512, 256, 128, 64, 32),
                 res_layers: Tuple = (5, 7, 8, 9),
                 relu_slope: float = 0.2,
                 padding_val: float = -11.5129) -> None:
        super().__init__()

        self.padding_val = padding_val
        self.mel_channels = mel_channels
        self.hop_length = 256
        c_0, c_1, c_2, c_3, c_4 = channels
        r_0, r_1, r_2, r_3 = res_layers

        self.blocks = Sequential(
            WNConv1d(mel_channels, c_0, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            LeakyReLU(relu_slope),
            WNConvTranspose1d(c_0, c_1, kernel_size=16, stride=8, padding=4),
            ResStack(c_1, c_1, num_layers=r_0),
            LeakyReLU(relu_slope),
            WNConvTranspose1d(c_1, c_2, kernel_size=16, stride=8, padding=4),
            ResStack(c_2, c_2, num_layers=r_1),
            LeakyReLU(relu_slope),
            WNConvTranspose1d(c_2, c_3, kernel_size=4, stride=2, padding=1),
            ResStack(c_3, c_3, num_layers=r_2),
            LeakyReLU(relu_slope),
            WNConvTranspose1d(c_3, c_4, kernel_size=4, stride=2, padding=1),
            ResStack(c_4, c_4, num_layers=r_3),
            LeakyReLU(relu_slope),
            WNConv1d(c_4, 1, kernel_size=7, padding=3, padding_mode='reflect'),
            Tanh()
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = (mel.detach() + 5.0) / 5.0
        return self.blocks(mel)

    def inference(self,
                  mel: torch.Tensor,
                  pad_steps: int = 10) -> torch.Tensor:
        with torch.no_grad():
            pad = torch.full((1, self.mel_channels, pad_steps),
                             self.padding_val).to(mel.device)
            mel = torch.cat((mel, pad), dim=2)
            audio = self.forward(mel).squeeze()
            audio = audio[:-(self.hop_length * pad_steps)]
        return audio


if __name__ == '__main__':
    model = MelganGenerator(80)

    x = torch.randn(3, 80, 1000)
    print(x.shape)

    y = model(x)
    print(y.shape)
    #assert y.shape == torch.Size([3, 1, 2560])

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
