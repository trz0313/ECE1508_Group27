import torch
import torch.nn as nn
import os
import time
import numpy as np
import SimpleITK as sitk
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Translation of Chinese comments included and code structured for better readability.

class Cus_Trainer_x(nn.Module):
    """
    Custom Trainer for GAN-based Image Synthesis.
    Includes training logic, logging, and evaluation functions.
    """

    def __init__(self, config):
        super(Cus_Trainer_x, self).__init__()
        self.config = config

        # Initialize generator and discriminator
        self.netG = config['Generator']
        self.netD = config['Discriminator']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG.to(self.device)
        self.netD.to(self.device)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config['lr'], betas=(0.5, 0.999))

        # Loss Functions
        self.criterion_GAN = nn.MSELoss() if config['use_lsgan'] else nn.BCELoss()
        self.criterion_L1 = nn.L1Loss()

        # Tensorboard Writer
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Checkpoints and Saving Paths
        self.checkpoint_dir = config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, dataloader):
        """
        Train the GAN model on the provided dataset.
        Args:
            dataloader: PyTorch DataLoader providing the training data.
        """
        self.netG.train()
        self.netD.train()

        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            for i, data in enumerate(dataloader):
                # Prepare data and move to device
                real_A = data['A'].to(self.device)
                real_B = data['B'].to(self.device)

                # Generate fake images
                fake_B = self.netG(real_A)

                # Train Discriminator
                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.netD(real_B)
                loss_D_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))

                # Fake loss
                pred_fake = self.netD(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

                # Total Discriminator loss
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()

                # GAN loss
                pred_fake = self.netD(fake_B)
                loss_G_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))

                # L1 loss
                loss_G_L1 = self.criterion_L1(fake_B, real_B) * self.config['lambda_L1']

                # Total Generator loss
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                self.optimizer_G.step()

                # Logging
                if i % self.config['log_interval'] == 0:
                    print(f"Epoch [{epoch}/{self.config['num_epochs']}], Step [{i}/{len(dataloader)}], "
                          f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")
                    self.writer.add_scalar('Loss/Discriminator', loss_D.item(), epoch * len(dataloader) + i)
                    self.writer.add_scalar('Loss/Generator', loss_G.item(), epoch * len(dataloader) + i)

            # Save checkpoint after each epoch
            self.save_checkpoint(epoch)
            print(f"Epoch [{epoch}/{self.config['num_epochs']}] completed in {time.time() - start_time:.2f}s")

    def save_checkpoint(self, epoch):
        """ Save model checkpoints. """
        torch.save(self.netG.state_dict(), os.path.join(self.checkpoint_dir, f'netG_epoch_{epoch}.pth'))
        torch.save(self.netD.state_dict(), os.path.join(self.checkpoint_dir, f'netD_epoch_{epoch}.pth'))
        print(f"Checkpoint saved for epoch {epoch}.")

    def test(self, dataloader):
        """
        Test the GAN model and save generated images.
        Args:
            dataloader: PyTorch DataLoader providing the test data.
        """
        self.netG.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                real_A = data['A'].to(self.device)
                fake_B = self.netG(real_A)

                # Convert to SimpleITK images and save
                fake_B_np = fake_B.squeeze().cpu().numpy()
                sitk_img = sitk.GetImageFromArray(fake_B_np)
                sitk.WriteImage(sitk_img, f"{self.config['output_dir']}/generated_{i}.nii.gz")
                print(f"Generated image saved: {self.config['output_dir']}/generated_{i}.nii.gz")

if __name__ == '__main__':
    # Example configuration
    config = {
        'Generator': None,  # Replace with the generator instance
        'Discriminator': None,  # Replace with the discriminator instance
        'lr': 0.0002,
        'use_lsgan': True,
        'lambda_L1': 10.0,
        'num_epochs': 200,
        'log_dir': './logs',
        'checkpoint_dir': './checkpoints',
        'output_dir': './output',
        'log_interval': 10
    }

    # Initialize trainer (example with dummy generator and discriminator)
    trainer = Cus_Trainer_x(config)

    # Example usage (replace with real DataLoader)
    dummy_dataloader = None  # Replace with your DataLoader
    # trainer.train(dummy_dataloader)
    # trainer.test(dummy_dataloader)
