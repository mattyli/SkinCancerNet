from VAE import HairRemovalVAE, train_VAE
from VAEQualityLoss import VAEQualityLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms, datasets, models
import os
from ISICDataset import ISICDataset
import json
import datetime
import argparse


IMG_SIZE = (256, 256)
BATCH_SIZE = 16
SEED = 42

def get_args():
    parser = argparse.ArgumentParser(description="Train VAE for Dermoscopy Hair Removal")

    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--num_epochs", type=int, default=1, 
                        help="Number of epochs (default: 1)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size (default: 16)")
    parser.add_argument("--patience", type=int, default=3, 
                        help="Scheduler patience (default: 3)")
    
    # Loss Function Weights
    parser.add_argument("--lambda_vae", type=float, default=0.8, 
                        help="Weight for VAE Loss (Recon + KL) (default: 0.8)")
    parser.add_argument("--lambda_ssim", type=float, default=0.2, 
                        help="Weight for SSIM Loss (default: 0.2)")

    # System / Logging
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of dataloader workers (default: 0)")
    parser.add_argument("--checkpoint_path", type=str, default="./VAE_checkpoints", 
                        help="Path to save checkpoints")
    parser.add_argument("--run_name", type=str, default="", 
                        help="Name for the run (default: empty string)")

    return parser.parse_args()

def main():
    args = get_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    with open('splits.json', 'rb') as f:
        splits = json.load(f)
    train_list, val_list = splits['training_set'], splits['validation_set']

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    train_dataset = ISICDataset(train_list, transform=train_transform)
    val_dataset = ISICDataset(val_list, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=(device.type == 'cuda')
    )
    
    model = HairRemovalVAE()
    model.to(device)  # CRITICAL: Move model to GPU

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.patience
    )

    # Pass the lambda arguments to your loss function
    criterion = VAEQualityLoss(
        lambda_vae=args.lambda_vae, 
        lambda_ssim=args.lambda_ssim
    )

    train_VAE(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        checkpoint_path=args.checkpoint_path,
        run_name=args.run_name,
        device=device
    )

if __name__ == "__main__":
    main()
