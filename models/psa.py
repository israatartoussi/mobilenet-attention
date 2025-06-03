import torch
import torch.nn as nn
import torch.nn.functional as F

class PSA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(PSA, self).__init__()
        reduced_channels = in_channels // reduction

        # Channel-only attention
        self.channel_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial-only attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        channel_att = self.channel_conv(x)
        x_channel = x * channel_att

        # Spatial attention
        spatial_att = self.spatial_conv(x)
        x_spatial = x * spatial_att

        # Combine attentions
        out = x_channel + x_spatial
        return out
