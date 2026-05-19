import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.FCANet import MultiSpectralAttentionLayer


class Fuse_block(nn.Module):
    def __init__(self, dim, channels):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(dim, channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.down_conv = nn.Sequential(
            nn.Sequential(nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 4, channels * 2, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1), nn.ReLU()),
            nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), nn.Tanh()),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # n,c,h,w
        x = self.encoder(x)
        x = self.down_conv(x)
        return x

class Att_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.att = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid())

    def forward(self, x):
        att = self.att(x)
        x = x * att
        return x


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.convx = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x

class DMRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.att1 = Att_Block(in_channels, out_channels)
        self.grad = Sobelxy(out_channels)

    def forward(self, x):
        x = self.embed(x)
        x1 = self.att1(x)
        x2 = self.grad(x)
        x = x1 + x2
        return x


class MDFM(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        c2hw = dict([(16, 256), (32, 128), (64, 64), (128, 32), (256, 16)])
        self.msal = MultiSpectralAttentionLayer(out_channels, c2hw[out_channels], c2hw[out_channels])
        # self.dmrm = DMRM(out_channels, out_channels)
        self.sa = SpatialAttention()
        # self.fuse = Fuse_block(out_channels*2, out_channels)
        self.conv_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.msal(x) * x
        x = self.sa(x) * x
        return x

if __name__ == '__main__':
    model = MDFM(16)
    x = torch.randn(4, 16, 256, 256)
    # print(x)
    x = model(x)
    print(x.shape)