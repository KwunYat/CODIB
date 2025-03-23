import torch
import torch.nn as nn
import torch.nn.functional as F
from net.PVTv2 import pvt_v2_b2
from net.SA import SpatialAttention_max
from net.VGG import *
from net.PCA import *
from net.Grad import GradientBasedModule
import numpy as np
import os
from torch.nn import Softmax, Dropout
import time
from thop import profile



class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class WaveletTransform(nn.Module):
    def __init__(self, in_channels, channel):
        super(WaveletTransform, self).__init__()
        self.in_channels = in_channels
        self.channel = channel

        harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
        harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

        harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
        harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
        harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
        harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

        self.filter_LL = nn.Parameter(torch.from_numpy(harr_wav_LL).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
        self.filter_LH = nn.Parameter(torch.from_numpy(harr_wav_LH).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
        self.filter_HL = nn.Parameter(torch.from_numpy(harr_wav_HL).float().unsqueeze(0).unsqueeze(0), requires_grad=True)
        self.filter_HH = nn.Parameter(torch.from_numpy(harr_wav_HH).float().unsqueeze(0).unsqueeze(0), requires_grad=True)

        self.merge_conv = Res_block(in_channels * 3, channel)
        self.Conv = Res_block(channel, in_channels)

    def forward(self, x):
        LL = nn.functional.conv2d(x, self.filter_LL.repeat(self.in_channels, 1, 1, 1), stride=2, padding=0, groups=self.in_channels)
        LH = nn.functional.conv2d(x, self.filter_LH.repeat(self.in_channels, 1, 1, 1), stride=2, padding=0, groups=self.in_channels)
        HL = nn.functional.conv2d(x, self.filter_HL.repeat(self.in_channels, 1, 1, 1), stride=2, padding=0, groups=self.in_channels)
        HH = nn.functional.conv2d(x, self.filter_HH.repeat(self.in_channels, 1, 1, 1), stride=2, padding=0, groups=self.in_channels)

        merged = torch.cat((LH, HL, HH), dim=1)

        output = self.Conv(self.merge_conv(merged))
        output_upsampled = nn.functional.interpolate(output, size=x.size()[2:], mode='bilinear', align_corners=False)

        return output_upsampled
    
    def initialize(self):
        weight_init(self)




def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            m.initialize()

class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(FAM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(in_channel_down, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channel_right, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.conv_d1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)


    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True) #256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True) #256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True) #256

        down_1 = self.conv_d1(down)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')


        out = torch.cat((left, down_1, down_2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)


    def initialize(self):
        weight_init(self)

class SRM(nn.Module):
    def __init__(self, in_channel):
        super(SRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, in_channel*2, kernel_size=3, stride=1, padding=1)

    def forward(self, x, in_channel):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True) #128
        out2 = self.conv2(out1)
        w, b = out2[:, :in_channel, :, :], out2[:, in_channel:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class Network(nn.Module):
    def __init__(self, channel):
        super(Network, self).__init__()
        

        self.Wave = WaveletTransform(3, 16)
        self.HF_backbone = VGG()
        self.backbone = pvt_v2_b2()
        path = './models/pvt_v2_b2.pth'
        
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        
        self.conv_HF4 = nn.Conv2d(in_channels=512, out_channels=channel, kernel_size=3, padding=1)
        self.conv_x4 = nn.Conv2d(in_channels=512, out_channels=channel, kernel_size=3, padding=1)
        self.Grad = GradientBasedModule(channel, channel)

        self.crossattention = PCASC(channel, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=8)
        
        self.BottlNeck_1 = SpatialAttention_max(64, 32)
        self.BottlNeck_2 = SpatialAttention_max(64, 16)

        
        self.fam_1 = FAM(256, 320, 32)
        self.fam_2 = FAM(128, 128, 64)
        self.fam_3 = FAM(64, 64, 64)
        
        self.sr_4 = SRM(64)
        self.sr_3 = SRM(64)
        self.sr_2 = SRM(64)
        self.sr_1 = SRM(64)
        
        self.ps = nn.PixelShuffle(2)
        

        self.linear4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear0 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

        self.linearz1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.linearz2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)



        
    def forward(self, image):
        # bs, 64, 96, 96
        # bs, 128, 48, 48
        # bs, 320, 24, 24
        # bs, 512, 12, 12     
        wavelet = self.Wave(image)
        HF0 = self.HF_backbone.conv1(wavelet)
        HF1 = self.HF_backbone.conv2(HF0)
        HF2 = self.HF_backbone.conv3(HF1)
        HF3 = self.HF_backbone.conv4(HF2)
        HF4 = self.HF_backbone.conv5(HF3)
        x = self.backbone(image)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        
        HF_feature = self.conv_HF4(HF4)
        HF_feature = self.Grad(HF_feature)
        x_feature = self.conv_x4(x4)
                
        fusion1 = self.crossattention(x_feature, HF_feature)
        fusion = self.sr_4(fusion1, 64)
        z1 = self.BottlNeck_1(fusion)
        
        fam1 = self.fam_1(HF3, x3, z1)
        fam1 = self.sr_3(fam1, 64)
        fam2 = self.fam_2(HF2, x2, fam1)
        fam2 = self.sr_2(fam2, 64)
        fam3 = self.fam_3(HF1, x1, fam2)
        fam3 = self.sr_1(fam3, 64)
        z2 = self.BottlNeck_2(fam3)
        ps = self.ps(z2)
        out = self.linear0(ps)

        out4 = F.interpolate(self.linear4(fusion), size=image.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.linear3(fam1), size=image.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.linear2(fam2), size=image.size()[2:], mode='bilinear')
        out1 = F.interpolate(self.linear1(fam3), size=image.size()[2:], mode='bilinear')
        out = F.interpolate(out, size=image.size()[2:], mode='bilinear')

        z1 = F.interpolate(self.linearz1(z1), size=image.size()[2:], mode='bilinear')
        z2 = F.interpolate(self.linearz2(z2), size=image.size()[2:], mode='bilinear')


       
        return out, out1, out2, out3, out4, z1, z2, wavelet


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = Network(64).cuda()
    input1 = torch.rand(1,3,384,384).cuda()
    torch.cuda.synchronize()
    start_time = time.time()
    out, out1, out2, out3, out4, z1, z2, wavelet = net(input1)
    end_time = time.time()
    inference_time = end_time - start_time
    print('Latency = '+str(inference_time) + 's')
    flops, params = profile(net, (input1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')