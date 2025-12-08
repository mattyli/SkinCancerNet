from VAE import HairRemovalVAE, train_VAE
from models.HairRemovalVAE.VAEQualityLoss import VAEQualityLoss
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


IMG_SIZE = (256, 256)
BATCH_SIZE = 16
SEED = 42

def main():

    # find device
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    num_workers = 0#min(8, (os.cpu_count() or 1))
    pin_memory = torch.cuda.is_available()
    
    # load the training splits
    with open('splits.json', 'rb') as f:
        splits = json.load(f)

    train, val = splits['training_set'], splits['validation_set']           # train and val are lists of dicts

    # define the transforms
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])

    # create datasets and dataloaders
    train_dataset = ISICDataset(train, transform=train_transform)
    val_dataset = ISICDataset(val, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    
    # instantiate model
    model = HairRemovalVAE()
    
    # define training hyperparameters
    lr = 1e-4
    num_epochs = 1
    scheduler_patience = 3
    checkpoint_path = "./VAE_checkpoints"
    run_name = ""
    criterion = VAEQualityLoss(lambda_vae=0.8, lambda_ssim=0.2)     # default configuration, for verbosity
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=scheduler_patience)

    # execute training function
    train_VAE(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        device=device
    )
