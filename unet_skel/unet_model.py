""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .unet_parts import *

class UNetSkeleton(nn.Module):
    """
    UNet Skeleton built on the Pytorch-UNet style, but with:
      - five (conv+res) repeats per down level
      - a 1×1 conv bottleneck with two layers + ReLU
      - four convs per up level after concat
    """
    def __init__(self, in_channels: int, out_channels: int,
                 base_ch: int = 32, levels: int = 5):
        super().__init__()
        self.levels = levels

        # Down path
        self.downs = nn.ModuleList()
        ch = in_channels
        for i in range(levels):
            out_ch = base_ch * (2 ** i)
            self.downs.append(DownBlockCustom(ch, out_ch, repeats=5))
            ch = out_ch

        # Bottleneck: two 1×1 conv layers with ReLU
        self.bottleneck_conv1 = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bottleneck_relu1 = nn.ReLU(inplace=True)
        self.bottleneck_conv2 = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.bottleneck_relu2 = nn.ReLU(inplace=True)

        # Up path
        self.ups = nn.ModuleList()
        for i in reversed(range(levels)):
            skip_ch = base_ch * (2 ** i)
            out_ch = base_ch * (2 ** i)
            self.ups.append(UpBlockCustom(ch, skip_ch, out_ch, convs=4))
            ch = out_ch

        # Final 1×1 conv
        self.final = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        cur = x
        # Down
        for down in self.downs:
            skip, cur = down(cur)
            skips.append(skip)

        # Bottleneck
        cur = self.bottleneck_conv1(cur)
        cur = self.bottleneck_relu1(cur)
        cur = self.bottleneck_conv2(cur)
        cur = self.bottleneck_relu2(cur)

        # Up
        for up, skip in zip(self.ups, reversed(skips)):
            cur = up(cur, skip)

        out = self.final(cur)
        return out


# Example usage
if __name__ == "__main__":
    model = UNetSkeleton(in_channels=3, out_channels=1, base_ch=16, levels=4)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)
