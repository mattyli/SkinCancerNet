import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import os
import json

class HairRemovalVAE(nn.Module):
    def __init__(self):
        super(HairRemovalVAE, self).__init__()
        self.encoder_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.flat_size = 32 * 64 * 64        
        self.fc_mu = nn.Linear(self.flat_size, 50)
        self.fc_logvar = nn.Linear(self.flat_size, 50) # representing sigma (log variance usually used for stability)
        self.decoder_input = nn.Linear(50, self.flat_size)
        
        self.decoder_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )
        self.decoder_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Final reconstruction layer to get RGB output
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid() # Sigmoid forces output to [0, 1] range for images
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # --- Encoder ---
        x = self.encoder_layer_1(x)
        x = self.encoder_layer_2(x)
        
        # Flatten
        x_flat = x.view(x.size(0), -1)
        
        # Latent space
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        
        # Sampling from the distribution
        z = self.reparameterize(mu, logvar)
        
        # --- Decoder ---
        # Unflatten
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(z.size(0), 32, 64, 64)
        
        x_recon = self.decoder_layer_1(z_reshaped)
        x_recon = self.decoder_layer_2(x_recon)
        
        return x_recon, mu, logvar

def train_VAE(model, 
              train_loader, 
              val_loader, 
              criterion, 
              optimizer,
              scheduler, 
              num_epochs,
              checkpoint_path,
              run_name,
              device):
    
    # 1. Setup
    model.to(device).float()
    os.makedirs(checkpoint_path, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    print(f"Starting training on device: {device}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_train_loss = 0.0
        
        for batch in train_loader:
            # Handle cases where loader returns (img, label) or just img
            if isinstance(batch, (list, tuple)):
                imgs = batch[0] # Ignore labels for VAE
            else:
                imgs = batch
            
            imgs = imgs.to(device, dtype=torch.float32)
            optimizer.zero_grad()            
            x_recon, mu, logvar = model(imgs)
            
            # Compute Loss
            loss_output = criterion(imgs, x_recon, mu, logvar)
            
            # Handle if criterion returns a tuple (total_loss, mse, kl...) or just scalar
            if isinstance(loss_output, tuple):
                loss = loss_output[0] # Assume first element is total loss for backprop
            else:
                loss = loss_output
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch
                
                imgs = imgs.to(device, dtype=torch.float32)
                
                # Forward pass
                x_recon, mu, logvar = model(imgs)
                
                # Compute Loss
                loss_output = criterion(imgs, x_recon, mu, logvar)
                
                if isinstance(loss_output, tuple):
                    loss = loss_output[0]
                else:
                    loss = loss_output
                
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        # Step the scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(epoch_val_loss)

        # Logging
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Epoch {epoch+1}/{num_epochs}\t "
              f"Train Loss: {epoch_train_loss:.4f}\t "
              f"Val Loss: {epoch_val_loss:.4f}\t "
              f"[{current_time}]")

        # ==========================
        # 3. Checkpointing
        # ==========================
        # Save best model if validation loss improves
        if epoch_val_loss < best_loss:
            print(f"--> Validation loss improved from {best_loss:.4f} to {epoch_val_loss:.4f}. Saving model.")
            best_loss = epoch_val_loss
            save_name = f"best_{run_name}.pth"
            torch.save(model.state_dict(), os.path.join(checkpoint_path, save_name))
        
        # Optional: Save every 10 epochs as a backup
        if (epoch + 1) % 10 == 0:
             torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{run_name}_epoch_{epoch+1}.pth"))

    print("Training Complete.")

    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_loss": best_loss,
        "epochs": num_epochs,
        "run_name": run_name,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    json_path = os.path.join(checkpoint_path, f"{run_name}_history.json")
    
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    print(f"Training history saved to: {json_path}")
    
    return