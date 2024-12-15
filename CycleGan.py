import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Residual Block for Generator
class ResidualBlock(nn.Module):
    """
    Defines a residual block for the generator.
    Each block includes two convolution layers with Instance Normalization and ReLU activation.
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),                # Reflection padding to maintain spatial size
            nn.Conv2d(in_features, in_features, 3),  # 3x3 convolution
            nn.InstanceNorm2d(in_features),       # Instance normalization
            nn.ReLU(inplace=True),               # ReLU activation
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        # Add the input (skip connection) to the output of the conv block
        return x + self.conv_block(x)


# Generator Network
class Generator(nn.Module):
    """
    The generator uses an encoder-decoder structure with residual blocks for the transformation.
    """
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial Convolution Block
        model_head = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),  # 7x7 kernel for large receptive field
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling Layers
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Residual Blocks
        model_body = [ResidualBlock(in_features) for _ in range(n_residual_blocks)]

        # Upsampling Layers
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2

        # Output Layer
        model_tail += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()  # Maps the output to the range [-1, 1]
        ]

        # Combine all layers
        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        # Pass input through head, body (residual blocks), and tail
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


# Discriminator Network
class Discriminator(nn.Module):
    """
    Defines the PatchGAN discriminator.
    """
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # Sequence of convolutional layers
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output layer for patch-based discrimination
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten for classification
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


# Test the Models
if __name__ == '__main__':
    input_tensor = torch.Tensor(np.random.rand(1, 1, 512, 512))  # Example input tensor

    # Test the Generator
    generator = Generator(input_nc=1, output_nc=1)
    gen_output = generator(input_tensor)
    print(f"Generator output shape: {gen_output.shape}")

    # Test the Discriminator
    discriminator = Discriminator(input_nc=1)
    disc_output = discriminator(input_tensor)
    print(f"Discriminator output shape: {disc_output.shape}")
