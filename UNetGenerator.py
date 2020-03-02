import torch
import torch.nn as nn

from NeuralModels import BaseNormalBlock,  ConvLayer, UpsampleDeConv
from DeepImageDenoiser import DIMENSION

LATENT_SPACE   = int(64)


class UNetGenerator(torch.nn.Module):
    def __init__(self, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(UNetGenerator, self).__init__()
        self.enc1   = UnetDoubleBlock(DIMENSION, 64, activation)
        self.enc2   = UnetDoubleBlock(64,  128, activation)
        self.enc3   = UnetDoubleBlock(128, 256, activation)
        self.enc4   = UnetDoubleBlock(256, 512, activation)

        self.center = UnetDoubleBlock(512, 1024)

        self.deconv4 = deconv(1024, 512)
        self.dec4  = UnetSingleBlock(1024, 512, activation = activation)
        self.deconv3 = deconv(512, 256)
        self.dec3  = UnetSingleBlock(512,  256, activation = activation)
        self.deconv2 = deconv(256, 128)
        self.dec2  = UnetSingleBlock(256,  128, activation = activation)
        self.deconv1 = deconv(128, 64)
        self.dec1  = UnetSingleBlock(128,   64, activation = activation)
        self.final  = ConvLayer(64, DIMENSION, 1)
        self.activation = activation
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.enc1(x)
        pool1 = self.max_pool(e1)
        e2 = self.enc2(pool1)
        pool2 = self.max_pool(e2)
        e3 = self.enc3(pool2)
        pool3 = self.max_pool(e3)
        e4 = self.enc4(pool3)
        pool4 = self.max_pool(e4)

        c = self.center(pool4)

        up6 = self.activation(self.deconv4(c))
        merge6 = torch.cat([up6, e4], dim=1)
        conv6 = self.dec4(merge6)
        up7 = (self.deconv3(conv6))
        merge7 = torch.cat([up7, e3], dim=1)
        conv7 = self.dec3(merge7)
        up8 = self.activation(self.deconv2(conv7))
        merge8 = torch.cat([up8, e2], dim=1)
        conv8 = self.dec2(merge8)
        up9 = self.activation(self.deconv1(conv8))
        merge9 = torch.cat([up9, e1], dim=1)
        conv9 = self.dec1(merge9)
        conv = self.activation(conv9)
        y = self.final(conv)
        return torch.tanh(y)


class UnetDoubleBlock(nn.Sequential):
    def __init__(self, in_size, out_size, activation=nn.LeakyReLU(0.2)):
        super(UnetDoubleBlock, self).__init__(
            BaseNormalBlock(in_size,  out_size, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
            BaseNormalBlock(out_size, out_size, 3, 1, activation=activation, normalization=nn.BatchNorm2d),
        )


class UnetSingleBlock(nn.Sequential):
    def __init__(self, in_size, out_size, activation=nn.LeakyReLU(0.2)):
        super(UnetSingleBlock, self).__init__(
            activation,
            BaseNormalBlock(in_size, out_size, 3, 1, activation=activation,normalization=nn.BatchNorm2d),
        )
