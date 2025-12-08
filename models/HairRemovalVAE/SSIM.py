import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py


def gaussian(window_size: int, sigma: float, device=None, dtype=None):
    """Create a 1D Gaussian kernel."""
    coords = torch.arange(window_size, device=device, dtype=dtype)
    center = (window_size - 1) / 2.0

    gauss = torch.exp(-((coords - center) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    return gauss


def create_window(window_size: int, sigma: float, channel: int, device=None, dtype=None):
    """Create a 2D Gaussian kernel window expanded along `channel` dimension."""
    _1d = gaussian(window_size, sigma, device=device, dtype=dtype).view(1, -1)
    _2d = (_1d.t() @ _1d)  # outer product → (W, W)

    window = _2d.unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(dim=(1, 2, 3))


class SSIM(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, size_average=True, channel=1):
        """
        SSIM module with configurable Gaussian sigma.
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.size_average = size_average
        self.channel = channel

        # Register the window so it tracks device/dtype moves
        window = create_window(window_size, sigma, channel)
        self.register_buffer("window", window)

    def forward(self, img1, img2):
        _, c, _, _ = img1.size()

        # Rebuild window if:
        # - channel changes
        # - sigma changes
        # - dtype/device changes
        if (
            c != self.channel
            or self.window.device != img1.device
            or self.window.dtype != img1.dtype
        ):
            window = create_window(
                self.window_size,
                self.sigma,
                c,
                device=img1.device,
                dtype=img1.dtype,
            )
            self.window = window
            self.channel = c

        return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.ssim = SSIM(window_size, sigma)

    def forward(self, x, y):
        return 1 - self.ssim(x, y)

# unused
def ssim(img1, img2, window_size=11, sigma=1.5, size_average=True):
    """Stateless SSIM functional API with customizable sigma."""
    _, channel, _, _ = img1.size()
    window = create_window(
        window_size,
        sigma,
        channel,
        device=img1.device,
        dtype=img1.dtype,
    )
    return _ssim(img1, img2, window, window_size, channel, size_average)
