import torch
import torch.nn as nn

from NeuralModels import BaseBlock, ConvLayer,  UpsampleDeConv, BaseNormalBlock
from DeepImageDenoiser import DIMENSION


class RSGUNetGenerator(torch.nn.Module):
    def __init__(self, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(RSGUNetGenerator, self).__init__()
        self.activation = activation
        self.enc1 = BaseBlock(DIMENSION, 16, 3, 2, activation=activation)
        self.enc2 = nn.Sequential(BaseBlock(16,   16, 3, 1, activation=activation),
                                  BaseBlock(16,   16, 3, 1, activation=activation), ConvLayer(16, 32, 3, 2))
        self.enc3 = nn.Sequential(BaseBlock(32,   32, 3, 1, activation=activation),
                                  BaseBlock(32,   32, 3, 1, activation=activation), ConvLayer(32, 64, 3, 2))
        self.enc4 = nn.Sequential(BaseBlock(64,   64, 3, 1, activation=activation),
                                  BaseBlock(64,   64, 3, 1, activation=activation), ConvLayer(64, 128, 3, 2))
        self.enc5 = nn.Sequential(BaseBlock(128, 128, 3, 1, activation=activation),
                                  BaseBlock(128, 128, 3, 1, activation=activation), ConvLayer(128, 128, 3, 2))

        self.center = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BaseBlock(128,128, 1, 1, activation=activation))

        self.dec6   = nn.Sequential(BaseBlock(256,128, 3, 1, activation=activation), BaseBlock(128,128, 3, 1, activation=activation), deconv(128, 64))
        self.dec7   = nn.Sequential(BaseBlock(192, 64, 3, 1, activation=activation), BaseBlock(64,  64, 3, 1, activation=activation), deconv(64,  32))
        self.dec8   = nn.Sequential(BaseBlock(96,  32, 3, 1, activation=activation), BaseBlock(32,  32, 3, 1, activation=activation), deconv(32,  16))
        self.dec9   = nn.Sequential(BaseBlock(48,  16, 3, 1, activation=activation), BaseBlock(16,  16, 3, 1, activation=activation), deconv(16,  16))
        self.deconv1 = deconv(16, 16)
        self.final = ConvLayer(16, DIMENSION, 3, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        c = c.expand_as(enc5)
        up6 = torch.cat([c, enc5], dim = 1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, enc4], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, enc3], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, enc2], dim=1)
        dec9 = torch.mul(self.dec9(up9), enc1)
        dec10 = self.deconv1(dec9)
        out = self.final(dec10)
        return torch.tanh(out)

'''
class RSGUNetGenerator(torch.nn.Module):
    def __init__(self, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(RSGUNetGenerator, self).__init__()
        self.activation = activation
        self.enc1 = BaseBlock(DIMENSION, 16, 3, 2, activation=activation)
        self.enc2 = nn.Sequential(BaseNormalBlock(16,   16, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
                                  BaseNormalBlock(16,   16, 3, 1, activation=activation, normalization=nn.BatchNorm2d), ConvLayer(16, 32, 3, 2))
        self.enc3 = nn.Sequential(BaseNormalBlock(32,   32, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
                                  BaseNormalBlock(32,   32, 3, 1, activation=activation, normalization=nn.BatchNorm2d), ConvLayer(32, 64, 3, 2))
        self.enc4 = nn.Sequential(BaseNormalBlock(64,   64, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
                                  BaseNormalBlock(64,   64, 3, 1, activation=activation, normalization=nn.BatchNorm2d), ConvLayer(64, 128, 3, 2))
        self.enc5 = nn.Sequential(BaseNormalBlock(128, 128, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
                                  BaseNormalBlock(128, 128, 3, 1, activation=activation, normalization=nn.BatchNorm2d), ConvLayer(128, 128, 3, 2))

        self.center = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BaseBlock(128,128, 1, 1, activation=activation))

        self.dec6   = nn.Sequential(BaseNormalBlock(256,128, 3, 1, activation=activation, normalization=nn.BatchNorm2d), BaseNormalBlock(128,128, 3, 1, activation=activation, normalization=nn.BatchNorm2d), deconv(128, 64))
        self.dec7   = nn.Sequential(BaseNormalBlock(192, 64, 3, 1, activation=activation, normalization=nn.BatchNorm2d), BaseNormalBlock(64,  64, 3, 1, activation=activation, normalization=nn.BatchNorm2d), deconv(64,  32))
        self.dec8   = nn.Sequential(BaseNormalBlock(96,  32, 3, 1, activation=activation, normalization=nn.BatchNorm2d), BaseNormalBlock(32,  32, 3, 1, activation=activation, normalization=nn.BatchNorm2d), deconv(32,  16))
        self.dec9   = nn.Sequential(BaseNormalBlock(48,  16, 3, 1, activation=activation, normalization=nn.BatchNorm2d), BaseNormalBlock(16,  16, 3, 1, activation=activation, normalization=nn.BatchNorm2d), deconv(16,  16))
        self.deconv1 = deconv(16, 16)
        self.final = ConvLayer(16, DIMENSION, 3, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        c = self.center(enc5)
        c = c.expand_as(enc5)
        up6 = torch.cat([c, enc5], dim = 1)
        dec6 = self.dec6(up6)
        up7 = torch.cat([dec6, enc4], dim=1)
        dec7 = self.dec7(up7)
        up8 = torch.cat([dec7, enc3], dim=1)
        dec8 = self.dec8(up8)
        up9 = torch.cat([dec8, enc2], dim=1)
        dec9 = torch.mul(self.dec9(up9), enc1)
        dec10 = self.deconv1(dec9)
        out = self.final(dec10)
        return torch.tanh(out)
'''