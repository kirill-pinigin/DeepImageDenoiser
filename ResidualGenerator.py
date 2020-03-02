import torch
import torch.nn as nn

from NeuralModels import ResidualBlock, ConvLayer, UpsampleDeConv, BaseNormalBlock
from DeepImageDenoiser import DIMENSION

LATENT_SPACE   = int(256)
LATENT_SPACE_2 = int(LATENT_SPACE / 2)
LATENT_SPACE_4 = int(LATENT_SPACE / 4)


class ResidualGenerator(torch.nn.Module):
    def __init__(self, deconv = UpsampleDeConv, activation = nn.LeakyReLU()):
        super(ResidualGenerator, self).__init__()
        self.DEPTH_SIZE = int(9)
        # Initial convolution layers
        self.conv1 = BaseNormalBlock(DIMENSION, LATENT_SPACE_4, kernel_size=9, stride=1, activation = activation)
        self.conv2 = BaseNormalBlock(LATENT_SPACE_4, LATENT_SPACE_2, kernel_size=3, stride=2, activation = activation)
        self.conv3 = BaseNormalBlock(LATENT_SPACE_2, LATENT_SPACE, kernel_size=3, stride=2, activation = activation)

        # Residual layers
        self.residual_blocks = nn.Sequential()

        for i in range(0, self.DEPTH_SIZE):
            self.residual_blocks.add_module(str(i), ResidualBlock(LATENT_SPACE, LATENT_SPACE, stride = 1, activation=activation))

        # Upsampling Layers
        self.deconv1 = deconv(LATENT_SPACE, LATENT_SPACE_2)
        self.norm1 = torch.nn.InstanceNorm2d(LATENT_SPACE_2, affine=True)
        self.deconv2 = deconv(LATENT_SPACE_2, LATENT_SPACE_4)
        self.norm2 = torch.nn.InstanceNorm2d(LATENT_SPACE_4, affine=True)
        self.final = ConvLayer(LATENT_SPACE_4, DIMENSION, kernel_size=9, stride=1)
        # Non-linearities
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_blocks(x)
        x = self.activation(self.norm1(self.deconv1(x)))
        x = self.activation(self.norm2(self.deconv2(x)))
        x = self.final(x)
        return torch.tanh(x)
