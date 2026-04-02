# ============================================================
# Quality per Joule — WGAN-GP (CelebA 32x32)
# models/wgan_gp.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Architecture adapted from Program_9_WGAN-GP.ipynb:
#   Critic: conv_block(64,5,s=2) -> conv_block(128,5,s=2,dropout=0.3)
#           -> conv_block(256,5,s=2,dropout=0.3) -> conv_block(512,5,s=2)
#           -> Flatten -> Dropout(0.2) -> Dense(1)  (no sigmoid)
#   Generator: Dense(4*4*256,BN,LeakyReLU) -> Reshape(4,4,256)
#              -> upsample(128,BN) -> upsample(64,BN) -> upsample(3,BN,tanh)
#
# Modifications from notebook:
#   - Input: (28,28,1) padded to (32,32,1) -> directly (32,32,3)
#   - Removed ZeroPadding2D/Cropping2D (input already 32x32)
#   - Output channels: 1 -> 3
#   - noise_dim default: 128 (same as notebook)
# ============================================================

import torch
import torch.nn as nn
from .dcgan import Generator, weights_init


# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 256, bias=False),
            nn.BatchNorm1d(4 * 4 * 256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        return self.net(x)


# Critic (Discriminator)
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512 * 2 * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# Gradient penalty for WGAN-GP
def gradient_penalty(critic, real, fake, device, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    c_interp = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=c_interp, inputs=interpolated,
        grad_outputs=torch.ones_like(c_interp),
        create_graph=True, retain_graph=True,
    )[0]
    gradients = gradients.view(B, -1)
    return lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Weights initialization
def critic_weights_init(m):
    cn = m.__class__.__name__
    if "Conv" in cn and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif "Linear" in cn and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif "BatchNorm" in cn and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ============================================================
####################### Sanity Check #########################
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  WGAN-GP (32x32) -- Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    netG = Generator(latent_dim=128).to(device)
    critic = Critic().to(device)
    netG.apply(critic_weights_init); critic.apply(critic_weights_init)
    g_p = sum(p.numel() for p in netG.parameters())
    c_p = sum(p.numel() for p in critic.parameters())
    print(f"  Generator: {g_p:,}  Critic: {c_p:,}  Total: {g_p+c_p:,}")

    z = torch.randn(4, 128).to(device)
    fake = netG(z)
    assert fake.shape == (4, 3, 32, 32)
    print(f"  G output: {fake.shape}  PASS")

    real = torch.randn(4, 3, 32, 32).to(device)
    c_out = critic(real)
    assert c_out.shape == (4,)
    print(f"  C output: {c_out.shape}  PASS (unbounded)")

    gp = gradient_penalty(critic, real, fake, device)
    print(f"  GP: {gp.item():.4f}  PASS")
    print("  All checks passed.")