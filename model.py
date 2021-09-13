import torch
from torch import nn
import torch.nn.functional as F

class ESPCN(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ESPCN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(channels_in, 50, 3),
            nn.Conv3d(50, 100, 1),
            nn.Conv3d(100, channels_out * 8, 3)
        )
        self.channels_out = channels_out
    
    def forward(self, x):
        x = self.layers(x)
        batchs, channels, h, w, d = x.size()
        x = x.reshape((batchs, channels, h * 2, w * 2, d * 2))
        return x


class SR_q_DL_S(nn.Module):
    def __init__(self):
        super(SR_q_DL_S, self).__init__()
        self.threshold = nn.Threshold(0.01, 0, inplace=True)
        self.s = nn.Conv3d(201, 201, 1)
    
    def forward(self, x, x_ref):
        x = self.threshold(x)
        x = self.s(x)
        return x + x_ref


class SR_q_DL(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(SR_q_DL, self).__init__()
        self.w = nn.Conv3d(channels_in, 201, 1)
        self.res = nn.ModuleList([SR_q_DL_S() for i in range(8)])
        self.threshold = nn.Threshold(0.01, 0, inplace=True)
        self.h = nn.Sequential(
            nn.Conv3d(201, 200, 3),
            nn.Dropout(0.1),
            nn.Conv3d(200, 400, 1),
            nn.Dropout(0.1),
            nn.Conv3d(400, channels_out * 8, 3)
        )
        self.channels_out = channels_out


    def forward(self, x):
        x = self.w(x)
        x_ref = torch.empty_like(x).copy_(x)
        for i in range(8):
            x = self.res[i](x, x_ref)
        x = self.threshold(x)
        x = self.h(x)

        batchs, channels, h, w, d = x.size()
        x = x.reshape((batchs, self.channels_out, h * 2, w * 2, d * 2))

        return x