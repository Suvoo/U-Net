import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels), # FIND out batchnorm
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
            nn.BatchNorm2d(out_channels), # FIND out batchnorm
            nn.ReLU(inplace=True)
        )
        # ISHAN DUTTA

        def forward(self,x):
            return self.conv(x)

class Unet(nn.Module):
    def __init__(self,in_channels=3,out_channels=1)



