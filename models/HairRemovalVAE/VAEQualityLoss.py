import torch
import torch.nn.functional as F
import kornia.losses

class VAEQualityLoss(torch.nn.Module):
    """
    Loss is a weighted sum of the VAE Loss (reconstruction loss + KL divergence)
    and the structural similarity (SSIM) loss.

    Change the respective lambdas as needed 
    """
    def __init__(self, lambda_vae=0.8, lambda_ssim=0.2):
        super().__init__()
        self.lambda_vae = lambda_vae
        self.lambda_ssim = lambda_ssim
        
        # Kornia's SSIMLoss computes (1 - SSIM) directly, window_size=11 is standard for SSIM.
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=11, reduction='mean')

    def forward(self, x, x_hat, mu, logvar):
        # A. Reconstruction Loss (MSE)
        # We assume inputs are normalized to [0, 1] or similar scale
        mse_loss = F.mse_loss(x_hat, x, reduction='mean')

        # B. KL Divergence Loss
        # Closed form solution for Normal distribution
        # sum over latent dim, mean over batch
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) 

        # Combine for VAE Loss
        loss_vae = mse_loss + kl_loss

        # C. SSIM Loss
        # Kornia calculates the structural dissimilarity (loss) automatically
        loss_ssim = self.ssim_loss(x_hat, x)

        # D. Total Weighted Loss
        total_loss = (self.lambda_vae * loss_vae) + (self.lambda_ssim * loss_ssim)

        return total_loss, mse_loss, kl_loss, loss_ssim