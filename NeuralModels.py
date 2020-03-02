import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Padding(torch.nn.Module):
    def __init__(self, padding_size = 1):
        super(Padding, self).__init__()
        self.pad = torch.nn.ReflectionPad2d(padding_size)

    def forward(self, x):
        return self.pad(x)


class ConvLayer(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias = False):
        super(ConvLayer, self).__init__(in_channels, out_channels, kernel_size, stride=stride, bias = bias)
        padding_size = kernel_size // 2
        self.pad = Padding(padding_size)

    def forward(self, x):
        x = self.pad(x)
        x = super(ConvLayer, self).forward(x)
        return x


class BaseBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation = Identity(), bias = False):
        super(BaseBlock, self).__init__(
            ConvLayer(in_channels, out_channels, kernel_size, stride, bias),
            activation,
        )


class BaseNormalBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation = Identity(), bias = False, normalization=  nn.InstanceNorm2d):
        super(BaseNormalBlock, self).__init__(
            ConvLayer(in_channels, out_channels, kernel_size, stride, bias),
            normalization(out_channels, affine=True),
            activation,
        )




class PixelDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelDeConv, self).__init__()
        self.conv2d = ConvLayer(in_channels, out_channels * 4, 3, 1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        return self.upsample(self.conv2d(x))


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, activation = nn.LeakyReLU(0.2)):
        super(ResidualBlock, self).__init__()
        self.conv1 = BaseNormalBlock(in_channels,  out_channels, kernel_size=3, stride=stride, activation = activation)
        self.conv2 = BaseNormalBlock(out_channels, out_channels, kernel_size=3, stride=1)
        self.skip  = BaseBlock(in_channels,  out_channels, kernel_size=1, stride=stride, bias= False)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return torch.add(x, residual)


class UpsampleDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(UpsampleDeConv, self).__init__()
        self.conv2d = ConvLayer(in_channels, out_channels, 3, 1, bias=False)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2)
        x  = self.conv2d(x)
        return x


class TransposedDeConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposedDeConv, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)

    def forward(self, x):
        return self.conv2d(x)


class SILU(torch.nn.Module):
    def __init__(self):
        super(SILU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        return out


class TotalVariation(nn.Module):
    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
