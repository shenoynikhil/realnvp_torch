import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

# weighted norm convolutional 2D normalization 
class WNConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0,
                 bias=True):
        super(WNConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                 bias=bias))

    def forward(self, x):
        return self.conv(x)

# Resnet block using the weighted norm
class ResnetBlock(nn.Module):
    def __init__(self, filters):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            WNConv2d(filters, filters, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            WNConv2d(filters, filters, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            WNConv2d(filters, filters, (1, 1), stride=1, padding=0))

    def forward(self, x):
        return x + self.block(x)

# Activation normalization that adds scale and translation to each channel and using data dependent normalization
class ActNorm(nn.Module):
    def __init__(self, n_channels):
        super(ActNorm, self).__init__()
        self.log_scale = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad = True) # scale factor (s) in paper
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1, 1), requires_grad = True) # translation factor
        self.channels = n_channels
        self.initialized = False

    def forward(self, x, reverse = False):
        if reverse:
            return (x - self.bias) * torch.exp(-self.log_scale), self.log_scale
        if not self.initialized:
            self.log_scale.data = -torch.log(torch.std(x.permute(1, 0, 2, 3).reshape(self.channels, -1), dim = 1)).view(1, self.channels, 1, 1)
            self.bias.data = -torch.mean(x.permute(1, 0, 2, 3).reshape(self.channels, -1), dim = 1).view(1, self.channels, 1, 1)
            self.initialized = True
        return x * torch.exp(self.log_scale) + self.bias, self.log_scale

class Resnet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 6, filters = 128, blocks = 3):
        super(Resnet, self).__init__()
        layers = []
        layers.extend([WNConv2d(in_channels, filters, (3, 3), stride = 1, padding = 1),
            nn.ReLU()])
        for _ in range(blocks):
            layers.append(ResnetBlock(filters))
        layers.extend([nn.ReLU(),
            WNConv2d(filters, out_channels, (3, 3), stride = 1, padding = 1)])
        self.resnet = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)

class AffineCheckerboardTransform(nn.Module):
    def __init__(self, type=1.0):
        super(AffineCheckerboardTransform, self).__init__()
        self.mask = self.build_mask(type=type)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = Resnet()

    def build_mask(self, type=1.0):
        # if type == 1.0, the top left corner will be 1.0 else on type == 0.0 it will be 0.0
        mask = np.arange(32).reshape(-1, 1) + np.arange(32)
        mask = np.mod(type + mask, 2)
        mask = mask.reshape(-1, 1, 32, 32)
        return torch.tensor(mask.astype('float32')).to(device)

    def forward(self, x, reverse=False):
        # returns transform(x), log_det
        batch_size, n_channels, _, _ = x.shape
        mask = self.mask.repeat(batch_size, 1, 1, 1)
        x_ = x * mask
        

        # from pseudo-code provided
        log_s, t = self.resnet(x_).split(n_channels, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift # both scale and scale_shift learnable params
        t = t * (1.0 - mask) # for the other half of the x
        log_s = log_s * (1.0 - mask) # for the other half of the x

        if reverse:  # inverting the transformation
            x = (x - t) * torch.exp(-log_s) # for inverse
        else:
            x = x * torch.exp(log_s) + t # for forward
        return x, log_s

class AffineChannelTransform(nn.Module):
    def __init__(self, modify_top):
        '''
        modify_top : Signifies which half of x is activated
        '''
        super(AffineChannelTransform, self).__init__()
        self.modify_top = modify_top
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.resnet = Resnet(in_channels=6, out_channels=12)

    def forward(self, x, reverse=False):
        batch_size, n_channels, _, _ = x.shape
        if self.modify_top:
            on, off = x.split(n_channels // 2, dim=1)
        else:
            off, on = x.split(n_channels // 2, dim=1)
        log_s, t = self.resnet(off).split(n_channels // 2, dim=1)
        log_s = self.scale * torch.tanh(log_s) + self.scale_shift

        if reverse:  # inverting the transformation
            on = (on - t) * torch.exp(-log_s)
        else:
            on = on * torch.exp(log_s) + t

        if self.modify_top:
            return torch.cat([on, off], dim=1), torch.cat([log_s, torch.zeros_like(log_s)], dim=1)
        else:
            return torch.cat([off, on], dim=1), torch.cat([torch.zeros_like(log_s), log_s], dim=1)

class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()
        self.prior = torch.distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device)) # standard normal distribution
        self.checker_transforms1 = nn.ModuleList([
            AffineCheckerboardTransform(1.0),
            ActNorm(3),
            AffineCheckerboardTransform(0.0),
            ActNorm(3),
            AffineCheckerboardTransform(1.0),
            ActNorm(3),
            AffineCheckerboardTransform(0.0)
        ])

        self.channel_transforms = nn.ModuleList([
            AffineChannelTransform(True),
            ActNorm(12),
            AffineChannelTransform(False),
            ActNorm(12),
            AffineChannelTransform(True),
        ])

        self.checker_transforms2 = nn.ModuleList([
            AffineCheckerboardTransform(1.0),
            ActNorm(3),
            AffineCheckerboardTransform(0.0),
            ActNorm(3),
            AffineCheckerboardTransform(1.0)
        ])

    def squeeze(self, x):
        # C x H x W -> 4C x H/2 x W/2
        B, C, H, W = x.size()
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return x

    def unsqueeze(self, x):
        #  4C x H/2 x W/2  ->  C x H x W
        B, C, H, W = x.size()
        x = x.reshape(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C // 4, H * 2, W * 2)
        return x

    def g(self, z):
        # z -> x (inverse of f)
        x = z
        for op in reversed(self.checker_transforms2):
            x, _ = op.forward(x, reverse=True)
        x = self.squeeze(x)
        for op in reversed(self.channel_transforms):
            x, _ = op.forward(x, reverse=True)
        x = self.unsqueeze(x)
        for op in reversed(self.checker_transforms1):
            x, _ = op.forward(x, reverse=True)
        return x

    def f(self, x):
        # maps x -> z, and returns the log determinant (not reduced)
        z, log_det = x, torch.zeros_like(x)
        for op in self.checker_transforms1:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        z, log_det = self.squeeze(z), self.squeeze(log_det)
        for op in self.channel_transforms:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        z, log_det = self.unsqueeze(z), self.unsqueeze(log_det)
        for op in self.checker_transforms2:
            z, delta_log_det = op.forward(z)
            log_det += delta_log_det
        return z, log_det

    def log_prob(self, x):
        z, log_det = self.f(x)
        # equation 3 RealNVP paper
        return torch.sum(log_det, dim = [1, 2, 3]) + torch.sum(self.prior.log_prob(z), dim = [1, 2, 3]) 

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, 3, 32, 32])
        return self.g(z)

