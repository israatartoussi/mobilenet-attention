# bam_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, dilation=4):
        super(BAM, self).__init__()

        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ch_att = self.channel_att(x)
        sp_att = self.spatial_att(x)
        att = self.sigmoid(ch_att * sp_att)
        return x * att
