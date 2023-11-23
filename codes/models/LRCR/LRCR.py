# --coding:utf-8--

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

from codes.models.cnns.up_conv_size import Up_conv_size
from codes.models.cnns.conv_depthwise_separable import Conv_DW
from codes.models.transformer.transformer import Trans as Trans
from codes.models.LRCR.Filter_branch import Filter_branch
from functools import partial

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(seed)

random.seed(seed)
np.random.seed(seed)

class LRCR(nn.Module):
    def __init__(self, cfg):
        super(LRCR, self).__init__()
        self.encoder = Encoder(in_ch=3, out_ch=256, in_H=cfg.patch_size, in_W=cfg.patch_size, cfg=cfg)
        self.decoder = Decoder(in_ch=256, out_ch=3, in_H=cfg.patch_size // 8, in_W=cfg.patch_size // 8, cfg=cfg)

        self.filter_processing = Filter_branch(cfg)
        self.cfg = cfg

    def forward(self, x):

        # step1: light restoration
        b3, b2, b1, b0, illu_map = self.encoder(x)
        out1 = self.decoder(b3, b2, b1, b0)

        # step2: color refinement
        out2 = self.filter_processing(x["img_ll_noise"], out1)

        out = {'img_enhance1': out1,
               'clear_low_img':None,
               "low_level_feature":None,
               'noise_map': None,
               'coeffs_out': None,
               'slice_coeffs':None}

        out.update(x)
        out.update(out2)
        out.pop('is_test')

        out2 = {}
        out2['illu_map'] = illu_map
        out2['en_block0'] = b0
        out2['en_block1'] = b1
        out2['en_block2'] = b2
        out2['en_block3'] = b3

        return out, out2


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, in_H, in_W, cfg):
        super(Encoder, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.in_H = in_H
        self.in_W = in_W

        self.input_conv = nn.Sequential(nn.Conv2d(4, 16, 3, 1, 1))
        self.pooling = nn.MaxPool2d(2, 2)
        self.depth_l_list = cfg.depth

        self.block0 = self._make_layer(16, 32, depth_l=self.depth_l_list[0], resolution=[in_H, in_W], ratio=2, compre_coe=4, patch_size=4,
                                    embed_dims=64, num_heads=4, mlp_ratios=4, depth_t=1, sr_ratios=7)
        self.block1 = self._make_layer(32, 64, depth_l=self.depth_l_list[1],resolution=[in_H // 2, in_W // 2], ratio=8, compre_coe=4, patch_size=4,
                                    embed_dims=64, num_heads=4, mlp_ratios=4, depth_t=1, sr_ratios=5)
        self.block2 = self._make_layer(64, 128, depth_l=self.depth_l_list[2],resolution=[in_H // 4, in_W // 4], ratio=16, compre_coe=4, patch_size=4,
                                    embed_dims=64, num_heads=8, mlp_ratios=4, depth_t=1, sr_ratios=3)
        self.block3 = self._make_layer(128, 256, depth_l=self.depth_l_list[3],resolution=[in_H // 8, in_W // 8], ratio=16, compre_coe=4, patch_size=4,
                                    embed_dims=64, num_heads=8, mlp_ratios=4, depth_t=1, sr_ratios=1)

        self.cv_conv = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(True),
                                     nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU(True))

    def forward(self, x):

        input = x["img_ll_noise"]
        illu = x['illu_map']

        if x['is_test']:
            illu = x['illu_map'].unsqueeze(dim=0)
        x_low_levle = self.input_conv(torch.cat((input, illu), 1))    # 将输入图像与照度图拼接

        b0 = self.block0(x_low_levle)
        b0_pool = self.pooling(b0)
        b1 = self.block1(b0_pool)
        b1_pool = self.pooling(b1)
        b2 = self.block2(b1_pool)
        b2_pool = self.pooling(b2)
        b3 = self.block3(b2_pool)

        return b3, b2, b1, b0, illu

    def _make_layer(self, in_ch, out_ch, depth_l, stride=1, compre_coe=4, heads=4, resolution=None, ratio=16,
                    patch_size=4, embed_dims=256, num_heads=4, mlp_ratios=4, depth_t=2, sr_ratios=2):
        conv = nn.Sequential()
        for i in range(depth_l):
            if i == 0:
                conv.add_module(str(i),Bottleneck_en(in_ch, out_ch, stride=stride, heads=heads, resolution=resolution, ratio=ratio,
                                          compre_coe=compre_coe // 2, patch_size=patch_size,
                                          embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
                                          depths=depth_t, sr_ratios=sr_ratios))
            else:
                conv.add_module(str(i),
                    Bottleneck_en(out_ch, out_ch, stride=stride, heads=heads, resolution=resolution, ratio=ratio,
                                  compre_coe=compre_coe, patch_size=patch_size,
                                  embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios, depths=depth_t,
                                  sr_ratios=sr_ratios))

        return conv

    def control_vector(self, x, cv):
        '''
        :param x:
        :param cv:
        :return:
        '''
        B, C, H, W = x.size()
        out = cv.repeat(1, C, H, W)
        out = Variable(out)

        return out

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, in_H, in_W, cfg):
        super(Decoder, self).__init__()
        self.out_conv = nn.Conv2d(32, out_ch, 3, 1, 1)
        self.block = Conv_DW(in_ch, in_ch)

        self.block0 = Conv_DW(in_ch , 128)
        self.block1 = Conv_DW(128, 64)
        self.block2 = Conv_DW(64, 32)

        self.up_0 = Up_conv_size(256, 128)
        self.up_1 = Up_conv_size(128, 64)
        self.up_2 = Up_conv_size(64, 32)

    def forward(self, b3, b2, b1, b0):

        x = self.block(b3)
        up0 = self.up_0(x, b2)
        f0 = torch.cat((up0, b2), 1)
        d0 = self.block0(f0)

        up1 = self.up_1(d0, b1)
        f1 = torch.cat((up1, b1), 1)
        d1 = self.block1(f1)

        up2 = self.up_2(d1, b0)
        f2 = torch.cat((up2, b0), 1)
        d2= self.block2(f2)

        out = self.out_conv(d2)

        return out

    def pixel_upsample(self, x,):
        x = nn.PixelShuffle(2)(x)
        return x

    def ft_size(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, x2.shape[2:])
        return x1


class Bottleneck_en(nn.Module):

    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, compre_coe=4,heads=4, resolution=None,ratio=16,
                 patch_size=4, embed_dims=256, num_heads=4, mlp_ratios=4, depths=2, sr_ratios=2):
        super(Bottleneck_en, self).__init__()

        self.inner_ch = out_ch // compre_coe
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.compre_coe = compre_coe

        self.conv_wx = RCRB(in_ch=self.in_ch, resolution=resolution, patch_size=patch_size, embed_dims=embed_dims,
                         num_heads=num_heads, mlp_ratios=mlp_ratios, depths=depths, sr_ratios=sr_ratios)
        self.conv = nn.Conv2d(self.in_ch, out_ch, 3, 1, 1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x, control_vector=None):

        x_multi = self.conv_wx(x, control_vector)
        x_conv = self.conv(x_multi)

        out = self.shortcut(x) + x_conv
        out = F.relu(out)
        return out


class Up_Size(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(Up_Size, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1, 1)
        self.up_conv = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, x, size):
        if not self.ch_in == self.ch_out:
            x = self.conv1x1(x)
        up = F.interpolate(input=x, size=size, mode='bilinear')
        up = self.up_conv(up)
        return up


class RCRB(nn.Module):
    '''
    region-calibrated residual block (RCRB)
    '''
    def __init__(self, in_ch, patch_size, resolution, embed_dims, num_heads, mlp_ratios, depths, sr_ratios):
        super(RCRB, self).__init__()

        self.in_ch_branch = in_ch // 2
        self.trans = Trans(img_size=resolution[0], in_chans=self.in_ch_branch, patch_size=patch_size, embed_dims=embed_dims,
                           num_heads=num_heads,
                           mlp_ratios=mlp_ratios, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                           depths=depths,
                           sr_ratios=sr_ratios)

        self.conv = nn.Conv2d(self.in_ch_branch, self.in_ch_branch, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self, x, control_vector=None):

        if control_vector is not None:
            x1 = torch.cat((x, control_vector), 1)
            x = self.compre(x1)

        x1, x2 = torch.split(x, self.in_ch_branch, 1)
        trans = self.trans(x1)
        conv = self.conv(x2)

        fusion = torch.cat((trans, conv), 1)
        out = self.relu(fusion)

        return out






















