# -*- coding: utf-8 -*-
# @Time : 2023/3/2 10:23
# @Author : LiQingCode
# @File : module.py
# @Project : JointUpsamplingUsingCSPN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from cspn import Affinity_Propagate

def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)

class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)

# 搭建unet 网络
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # 用 BN 代替 Dropout
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class Conv1_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1_1, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1))
    def forward(self, x):
        x = self.conv1_1(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.downsampling = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.downsampling(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.upsampling = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsampling(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class JUAL(nn.Module):
    def __init__(self, in_channels=7, out_channels=1, is_depth=True, mode='nearest', scale=8):
        super(JUAL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_depth = is_depth
        self.mode = mode
        self.scale = scale

        self.in_conv = DoubleConv(in_channels, 64)
        self.guidance_conv = Conv1_1(64, out_channels*8)
        self.blur_conv = Conv1_1(64, out_channels)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.cspn = Affinity_Propagate(8*out_channels, 3, norm_type='8sum')

    def forward(self, x_lr, x_hr, gt_down=None):
        if not self.is_depth:
            upsampling = nn.UpsamplingNearest2d([x_hr.size()[2], x_hr.size()[3]])
            x_lr_up = upsampling(x_lr)
            lr_up = upsampling(gt_down)
            x_hr_2 = torch.cat([x_hr, x_lr_up, lr_up], 1)
        else:
            downsampling = nn.Upsample(scale_factor=1 / self.scale, mode=self.mode)
            upsampling = nn.Upsample(size=[x_hr.shape[2], x_hr.shape[3]], mode='bicubic')
            hr_down_up = upsampling(downsampling(x_hr))
            lr_up = x_lr
            x_hr_2 = torch.cat([x_hr, hr_down_up, x_lr], 1)

        x1 = self.in_conv(x_hr_2)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        guidance = self.guidance_conv(x)
        out = self.cspn(guidance, lr_up)
        return out

    def init_lr(self, path):
        self.lr.load_state_dict(torch.load(path))
