import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
import numpy
from torch.nn.parameter import Parameter

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CEM_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CEM_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1) # 1×1的卷积
        self.branch1 = BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)
        self.branch2 = BasicConv2d(out_channel, out_channel, 5, padding=4, dilation=2)
        self.branch3 = BasicConv2d(out_channel, out_channel, 7, padding=6, dilation=2)
        self.conv_cat = BasicConv2d(3 * out_channel, out_channel, 1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x1 = self.branch1(self.branch0(x))
        x2 = self.branch2(self.branch0(x))
        x3 = self.branch3(self.branch0(x))
        x_cat = self.conv_cat(torch.cat((x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CA(nn.Module):
    def __init__(self, channels, reduction=32):
        super(CA, self).__init__()

        mip = 1

        self.conv1 = nn.Conv2d(channels, mip, 1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = torch.nn.functional.adaptive_avg_pool2d(x, (h, 1))
        x_w = torch.nn.functional.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


class CSAM(nn.Module):

    def __init__(self, planes, conv_kernels=[3, 5, 7], stride=1, conv_groups=[1, 4, 8]):
        super(CSAM, self).__init__()
        self.conv_1 = conv(512, planes, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(1024, planes, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(2048, planes, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.sweight = Parameter(torch.zeros(1, planes, 1, 1))
        self.sbias = Parameter(torch.ones(1, planes, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(planes, planes)

        self.ca = CA(32)
        self.split_channel = planes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3):
        x1_3 = self.conv_1(x1)
        x2_3 = self.conv_2(x2)
        x3_3 = self.conv_3(x3)
        x2_3 = F.interpolate(x2_3, scale_factor=2, mode='bilinear')
        x3_3 = F.interpolate(x3_3, scale_factor=4, mode='bilinear')

        batch_size = x1_3.shape[0]

        feats1 = torch.cat((x1_3, x2_3, x3_3), dim=1)
        feats1 = feats1.view(batch_size, 3, self.split_channel, feats1.shape[2], feats1.shape[3])

        xs1 = self.gn(x1_3)
        xs1 = self.sweight * xs1 + self.sbias
        xs1 = x1_3 * self.sigmoid(xs1)
        xs2 = self.gn(x2_3)
        xs2 = self.sweight * xs2 + self.sbias
        xs2 = x2_3 * self.sigmoid(xs2)
        xs3 = self.gn(x3_3)
        xs3 = self.sweight * xs3 + self.sbias
        xs3 = x3_3 * self.sigmoid(xs3)

        x_se = torch.cat((xs1, xs2, xs3), dim=1)
        attention_vectors = x_se.view(batch_size, 3, self.split_channel, x_se.shape[2], x_se.shape[3])
        attention_vectors = self.softmax(attention_vectors)

        x1_ca = self.ca(x1_3)
        x2_ca = self.ca(x2_3)
        x3_ca = self.ca(x3_3)

        xs = torch.cat((x1_ca, x2_ca, x3_ca), dim=1)
        xca2 = xs.view(batch_size, 3, self.split_channel, xs.shape[2], xs.shape[3])

        feats_weight = feats1 * xca2 * attention_vectors
        for i in range(3):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out




class MFFM_modified(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(MFFM_modified, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv5(x3_2)

        return x


class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)

        self.MFFM = MFFM_modified(32)

        self.CEM2 = CEM_modified(512, 32)
        self.CEM3 = CEM_modified(1024, 32)
        self.CEM4 = CEM_modified(2048, 32)

        self.conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.conv2 = BasicConv2d(96, 1, kernel_size=3, padding=1)
        # self.conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.CSAM = CSAM(32)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11

        B2 = self.CEM2(x2)  # 128
        B3 = self.CEM3(x3)  # 256
        B4 = self.CEM4(x4)  # 512
        E = self.MFFM(B4, B3, B2)
        E0 = F.interpolate(E, scale_factor=8, mode='bilinear')

        S = self.CSAM(x2, x3, x4)
        S = self.conv2(S)

        S0 = F.interpolate(S, scale_factor=8, mode='bilinear')
        R0 = E0 + S0

        return E0, S0, R0
