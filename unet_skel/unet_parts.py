import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualBlock(nn.Module):
    """Conv3x3 → BN → ReLU → Conv3x3 → BN + skip → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class DownBlockCustom(nn.Module):
    """One down-level: repeat (Conv3x3→BN→ReLU + ResidualBlock) five times, then MaxPool."""
    def __init__(self, in_ch: int, out_ch: int, repeats: int = 5):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(repeats):
            # conv3x3 + BN + ReLU
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            # residual block
            layers.append(ResidualBlock(out_ch, out_ch))
            ch = out_ch
        self.block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        skip = x
        x = self.pool(x)
        return skip, x


class UpBlockCustom(nn.Module):
    """One up-level: transposed conv → concat skip → four conv3x3 layers each (ConvBNReLU)"""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, convs: int =4):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        layers = []
        ch = out_ch + skip_ch
        for i in range(convs):
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            ch = out_ch
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.up(x)
        # adjust padding if needed
        if x.shape[-2:] != skip.shape[-2:]:
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [diffX//2, diffX - diffX//2,
                          diffY//2, diffY - diffY//2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv_block(x)
        return x