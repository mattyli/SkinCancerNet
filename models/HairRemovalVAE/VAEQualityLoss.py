import torch
import torch.nn.functional as F
import kornia.losses
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
from SSIM import SSIMLoss

class VAEQualityLoss(nn.Module):
    """
    Loss is a weighted sum of the VAE Loss (reconstruction loss + KL divergence)
    and the structural similarity (SSIM) loss.

    Change the respective lambdas as needed 
    """
    def __init__(self, lambda_vae, lambda_ssim):
        super().__init__()
        self.lambda_vae = lambda_vae
        self.lambda_ssim = lambda_ssim
        
        self.ssim_loss = SSIMLoss(window_size=3, sigma=1.5)

    def forward(self, x, x_hat, mu, logvar):
        # A. Reconstruction Loss (MSE)
        # We assume inputs are normalized to [0, 1] or similar scale
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')

        # B. KL Divergence Loss
        # Closed form solution for Normal distribution
        # sum over latent dim, mean over batch
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) 

        # Loss_VAE is defined as: mse loss - kl_div loss
        loss_vae = mse_loss - kl_loss

        # C. SSIM Loss
        # Kornia calculates the structural dissimilarity (loss) automatically
        loss_ssim = self.ssim_loss(x_hat, x)

        # D. Total Weighted Loss
        total_loss = (self.lambda_vae * loss_vae) + (self.lambda_ssim * loss_ssim)

        return total_loss, mse_loss, kl_loss, loss_ssim