import torch
from torch import nn
from torch.nn import functional


class Encoder3Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Encoder1Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.conv(x)
        out = torch.sigmoid(x1)
        return out


class DRDB(nn.Module):
    def __init__(self, in_ch=64, growth_rate=32):
        super().__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = functional.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = functional.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = functional.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.Dcov4(x3)
        x4 = functional.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)

        x5 = self.Dcov5(x4)
        x5 = functional.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)

        x6 = self.conv(x5)
        out = x+functional.relu(x6)
        return out


class AttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder1Conv(6, 64)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(6, 64, 3, padding=1),
#            nn.ReLU(inplace=True),)
        self.attention = AttentionModule()

    def forward(self, x1, x2, x3):
        feature1 = self.encoder(x1)
        refer = self.encoder(x2)
        feature2 = self.encoder(x3)
        map1 = self.attention(feature1, refer)
        map2 = self.attention(feature2, refer)
        feature_1 = torch.mul(feature1, map1)
        feature_2 = torch.mul(feature2, map2)
        out = torch.cat([feature_1, refer, feature_2], dim=1)
        return out, refer


class MergingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 64, 3, padding=1)
        self.DRDB1 = DRDB()
        self.DRDB2 = DRDB()
        self.DRDB3 = DRDB()
        self.conv2 = nn.Conv2d(192, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, xskip):
        x1 = self.conv1(x)
        x1 = functional.relu(x1)
        x2 = self.DRDB1(x1)
        x3 = self.DRDB2(x2)
        x4 = self.DRDB3(x3)

        x5 = torch.cat([x2, x3, x4], dim=1)

        x6 = self.conv2(x5)
        x6 = functional.relu(x6)

        x7 = x6 + xskip

        x8 = self.conv3(x7)
        x8 = functional.relu(x8)
        x9 = self.conv4(x8)
        out = functional.relu(x9)

        return out


class AHDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = AttentionNetwork()
        self.M = MergingNetwork()

    def forward(self, x1, x2, x3):
        midout, ref = self.A(x1, x2, x3)
        finalout = self.M(midout, ref)
        return finalout
