# ============================================================
# Quality per Joule — DCGAN (CelebA 32x32)
# models/dcgan.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Architecture adapted from Program_8_DCGAN.ipynb:
#   Discriminator: Conv(64,4,s=2) -> Conv(128,4,s=2) -> Conv(128,4,s=2)
#                  -> Flatten -> Dropout(0.2) -> Dense(1, sigmoid)
#   Generator:     Dense(8*8*128) -> Reshape -> ConvT(128,4,s=2) -> ConvT(256,4,s=2)
#                  -> ConvT(512,4,s=2) -> Conv(3,5,s=1,sigmoid)
#
# Modifications from notebook:
#   - Input: (64,64,3) -> (32,32,3) — removed one conv/convT layer
#   - Generator output: sigmoid -> tanh for [-1,1] normalization
#   - latent_dim default: 128 (same as notebook)
# ============================================================

import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 128),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 3, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)
        return self.net(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128 * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)

# Weight initialization
def weights_init(m):
    cn = m.__class__.__name__
    if "Conv" in cn and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    elif "Linear" in cn and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0)


# ============================================================
####################### Sanity Check #########################
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  DCGAN (32x32) -- Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    netG = Generator(latent_dim=128).to(device)
    netD = Discriminator().to(device)
    
    netG.apply(weights_init); netD.apply(weights_init)
    g_p = sum(p.numel() for p in netG.parameters())
    d_p = sum(p.numel() for p in netD.parameters())
    print(f"  G params: {g_p:,}  D params: {d_p:,}  Total: {g_p+d_p:,}")

    z = torch.randn(4, 128).to(device)
    fake = netG(z)
    assert fake.shape == (4, 3, 32, 32), f"G output shape: {fake.shape}"
    print(f"  G output: {fake.shape}  PASS")

    d_out = netD(fake.detach())
    assert d_out.shape == (4,), f"D output shape: {d_out.shape}"
    print(f"  D output: {d_out.shape}  PASS")
    print("  All checks passed.")
