#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import lpips  # Library for perceptual similarity metric
from .utils import LambdaLR, Logger, ReplayBuffer, Resize, ToTensor, smoothing_loss
from .datasets import ImageDataset, ValDataset, TestDataset
from Model.CycleGan import Generator, Discriminator
from torchvision.transforms import RandomAffine, ToPILImage
from skimage import measure
import numpy as np
import shutil
import pydicom
from prefetch_generator import BackgroundGenerator


# DataLoader with prefetching for efficient data loading
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# Helper function for adjusting window level in DICOM data
def to_windowdata(image, WC, WW):
    """
    Convert image to windowed data based on Window Center (WC) and Window Width (WW).

    Args:
        image (numpy.ndarray): Input image array.
        WC (float): Window center value.
        WW (float): Window width value.

    Returns:
        numpy.ndarray: Normalized image data.
    """
    image = (image + 1) * 0.5 * 4095  # Scale to 12-bit range
    image[image == 0] = -2000
    image -= 1024
    center = WC
    width = WW

    try:
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5
    except:
        center = WC[0]
        width = WW[0]
        win_min = (2 * center - width) / 2.0 + 0.5
        win_max = (2 * center + width) / 2.0 + 0.5

    dFactor = 255.0 / (win_max - win_min)
    image -= win_min
    image = np.trunc(image * dFactor)
    image[image > 255] = 255
    image[image < 0] = 0
    image = image / 255
    image = (image - 0.5) / 0.5  # Normalize to range [-1, 1]
    return image


# CycleGAN Trainer Class
class Cyc_Trainer:
    """
    Trainer for CycleGAN with Cycle Consistency Loss and adversarial losses.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netG_B2A = Generator(config['input_nc'], config['output_nc']).cuda()
        self.netD_B = Discriminator(config['input_nc']).cuda()
        self.netD_A = Discriminator(config['input_nc']).cuda()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=config['lr'], betas=(0.5, 0.999)
        )
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Loss functions
        self.MSE_loss = torch.nn.MSELoss()  # For adversarial loss
        self.L1_loss = torch.nn.L1Loss()  # For cycle consistency loss

        # Memory for inputs and targets
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1, 1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1, 1).fill_(0.0), requires_grad=False)

        # Replay buffers for storing fake images
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loaders
        transforms_train = [
            ToPILImage(),
            RandomAffine(degrees=config['noise_level'], translate=[0.02 * config['noise_level'], 0.02 * config['noise_level']],
                         scale=[1 - 0.02 * config['noise_level'], 1 + 0.02 * config['noise_level']], fillcolor=-1),
            ToTensor(),
            Resize(size_tuple=(config['size'], config['size']))
        ]

        val_transforms = [ToTensor(), Resize(size_tuple=(config['size'], config['size']))]

        self.dataloader = DataLoaderX(
            ImageDataset(config['train_list'], transforms_1=transforms_train, transforms_2=transforms_train, unaligned=False),
            batch_size=config['batchSize'], shuffle=True, num_workers=config['n_cpu'], pin_memory=True
        )
        self.val_data = DataLoaderX(
            ValDataset(config['val_list'], transforms_=val_transforms, unaligned=False),
            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu']
        )
        self.test_data = DataLoaderX(
            TestDataset(config['test_list'], transforms_=val_transforms, unaligned=False),
            batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu']
        )

        # Logger for training progress
        self.logger = Logger(config['name'], config['port'], config['n_epochs'] + config['decay_epoch'], len(self.dataloader))

    def update_learning_rate(self):
        """
        Adjust learning rate linearly during the decay phase.
        """
        lrd = self.config['lr'] / self.config['decay_epoch']
        lr = self.config['lr'] - lrd
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.config['lr'] = lr
        print(f"Updated learning rate: {self.config['lr']}")

    def train(self):
        """
        Training function for CycleGAN.
        Includes adversarial loss, cycle consistency loss, and learning rate adjustment.
        """
        for epoch in range(self.config['epoch'] + 1, self.config['n_epochs'] + 1 + self.config['decay_epoch']):
            if epoch > self.config['n_epochs']:
                self.update_learning_rate()

            for i, batch in enumerate(self.dataloader):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))

                # Generator updates
                self.optimizer_G.zero_grad()
                fake_B = self.netG_A2B(real_A)
                loss_GAN_A2B = self.config['Adv_lambda'] * self.MSE_loss(self.netD_B(fake_B), self.target_real)

                fake_A = self.netG_B2A(real_B)
                loss_GAN_B2A = self.config['Adv_lambda'] * self.MSE_loss(self.netD_A(fake_A), self.target_real)

                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.config['Cycle_lambda'] * self.L1_loss(recovered_A, real_A)

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.config['Cycle_lambda'] * self.L1_loss(recovered_B, real_B)

                total_loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                total_loss_G.backward()
                self.optimizer_G.step()

                # Discriminator updates for A and B
                self.optimizer_D_A.zero_grad()
                loss_D_A = self.compute_discriminator_loss(self.netD_A, real_A, self.fake_A_buffer.push_and_pop(fake_A))
                loss_D_A.backward()
                self.optimizer_D_A.step()

                self.optimizer_D_B.zero_grad()
                loss_D_B = self.compute_discriminator_loss(self.netD_B, real_B, self.fake_B_buffer.push_and_pop(fake_B))
                loss_D_B.backward()
                self.optimizer_D_B.step()

                # Logging
                self.logger.log({'loss_G': total_loss_G.item(), 'loss_D_A': loss_D_A.item(), 'loss_D_B': loss_D_B.item()})

            # Validation (every 5 epochs)
            if epoch % 5 == 0:
                self.validate(epoch)

    def compute_discriminator_loss(self, netD, real_data, fake_data):
        """
        Compute the loss for the discriminator.

        Args:
            netD (nn.Module): Discriminator network.
            real_data (Tensor): Real data tensor.
            fake_data (Tensor): Fake data tensor.

        Returns:
            Tensor: Total discriminator loss.
        """
        pred_real = netD(real_data)
        loss_D_real = self.config['Adv_lambda'] * self.MSE_loss(pred_real, self.target_real)

        pred_fake = netD(fake_data.detach())
        loss_D_fake = self.config['Adv_lambda'] * self.MSE_loss(pred_fake, self.target_fake)

        return loss_D_real + loss_D_fake

    def validate(self, epoch):
        """
        Validation function to evaluate model performance.

        Args:
            epoch (int): Current training epoch.
        """
        PSNR, SSIM, num_samples = 0, 0, 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_data):
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B'])).cpu().numpy().squeeze()
                fake_B = self.netG_A2B(real_A).cpu().numpy().squeeze()

                psnr = measure.compare_psnr(real_B, fake_B)
                ssim = measure.compare_ssim(real_B, fake_B)

                PSNR += psnr
                SSIM += ssim
                num_samples += 1

        avg_PSNR = PSNR / num_samples
        avg_SSIM = SSIM / num_samples
        print(f"Epoch {epoch}: Avg PSNR: {avg_PSNR:.2f}, Avg SSIM: {avg_SSIM:.2f}")
