import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable
import torchvision.transforms.functional as tf


# Prefetcher for Efficient Data Loading
class DataPrefetcher:
    """
    Prefetcher to load data asynchronously to the GPU for efficient training.
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        assert torch.cuda.is_available(), "CUDA is required for this prefetcher"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':  # Ignore metadata fields
                    self.batch[k] = self.batch[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


# Residual Block for Generator
class ResidualBlock(nn.Module):
    """
    Residual block with two convolution layers, Instance Normalization, and ReLU activation.
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)


# Generator Architecture
class Generator(nn.Module):
    """
    Generator network with encoder-decoder structure and residual blocks.
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model_head = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Residual blocks
        model_body = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]

        # Upsampling layers
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2

        # Output layer
        model_tail += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


# PatchGAN Discriminator
class Discriminator(nn.Module):
    """
    Implements the PatchGAN discriminator.
    """
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # Convolutional layers
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output classification layer
        model += [nn.Conv2d(512, 1, kernel_size=4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# Multi-Scale Discriminator
class Discriminator_m(nn.Module):
    """
    Multi-scale discriminator for enhanced adversarial training.
    """
    def __init__(self, input_nc, num_D=1, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(Discriminator_m, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        # Initialize multiple discriminators
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
            setattr(self, f"layer_{i}", netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        results = []
        for i in range(self.num_D):
            netD = getattr(self, f"layer_{i}")
            results.append(netD(x))
            if i < self.num_D - 1:
                x = self.downsample(x)
        return results


# GAN Loss Function
class GANLoss(nn.Module):
    """
    Loss function for adversarial training.
    Supports LSGAN and vanilla GAN loss.
    """
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELoss()

    def forward(self, input, target_is_real):
        target = torch.ones_like(input) if target_is_real else torch.zeros_like(input)
        return self.loss(input, target)


if __name__ == '__main__':
    # Example usage of the Generator and Discriminator
    input_tensor = torch.randn(1, 1, 256, 256)
    gen = Generator(input_nc=1, output_nc=1)
    disc = Discriminator(input_nc=1)

    gen_output = gen(input_tensor)
    disc_output = disc(input_tensor)

    print(f"Generator output shape: {gen_output.shape}")
    print(f"Discriminator output shape: {disc_output.shape}")
