#!/usr/bin/python3

import argparse
import os
import yaml
import warnings
import torch
import numpy as np
import random
from trainer import P2p_Trainer, Cyc_Trainer, Cus_Trainer_x

# Suppress warnings
warnings.filterwarnings("ignore")

# Set the CUDA device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

def get_config(config_path):
    """
    Load configuration from a YAML file.
    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as stream:
        return yaml.safe_load(stream)

def seed_everything(seed):
    """
    Set random seeds for reproducibility.
    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # Uncomment below if non-deterministic but faster training is preferred:
    # torch.backends.cudnn.benchmark = True

def main():
    """
    Main function to parse arguments, load configurations, and initialize training/testing.
    """
    parser = argparse.ArgumentParser(description="Training script for GAN-based models.")
    parser.add_argument('--config', type=str, default='Yaml/HdGan.yaml',
                        help='Path to the YAML configuration file.')
    opts = parser.parse_args()

    # Load configuration
    config = get_config(opts.config)

    # Initialize the appropriate trainer based on the configuration
    if config['name'] == 'CycleGan':
        trainer = Cyc_Trainer(config)
    elif config['name'] == 'P2p':
        trainer = P2p_Trainer(config)
    elif config['name'] == 'HdGan':
        # Note: During training, change `Hd_Trainer_x1` or `Hd_Trainer_x2` to `Hd_Trainer_x`
        # Options:
        # - `Hd_Trainer_x1`: Adds attention mechanisms.
        # - `Hd_Trainer_x2`: Introduces double input and skip connections.
        trainer = Cus_Trainer_x(config)
    else:
        raise ValueError(f"Unsupported model name: {config['name']}")

    # Execute training or testing
    # Uncomment below for training
    # trainer.train()
    trainer.test()  # Currently set to testing mode

###################################
if __name__ == '__main__':
    # Set seeds for reproducibility
    seed_everything(seed=42)
    main()
