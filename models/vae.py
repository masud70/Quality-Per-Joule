# ============================================================
# Quality per Joule — Variational Autoencoder (CelebA 32x32)
# models/vae.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Architecture adapted from Program_7_VAE_mnist.ipynb:
#   Encoder: Conv(32,3,s=2) -> Conv(64,3,s=2) -> Flatten -> Dense(16,relu)
#            -> Dense(z_mean), Dense(z_log_var), Sampling
#   Decoder: Dense(8*8*64,relu) -> Reshape -> ConvT(64,3,s=2) -> ConvT(32,3,s=2) -> ConvT(3,3,s=1)
#
# Modifications from notebook:
#   - Input: (28,28,1) -> (32,32,3)
#   - Flatten dim: 8*8*64=4096 (was 7*7*64=3136)
#   - Output channels: 3 (was 1), activation: Tanh (was Sigmoid)
#   - latent_dim default: 128 (was 2)
# ============================================================

import torch
import torch.nn as nn


# Encoder
class VAEEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc_mu     = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)   # (B, 4096)
        return self.fc_mu(h), self.fc_logvar(h)

# Decoder
class VAEDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 64, 8, 8)
        return self.deconv_layers(x)


# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder    = VAEEncoder(latent_dim)
        self.decoder    = VAEDecoder(latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def encode(self, x):  return self.encoder(x)
    def decode(self, z):  return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x, recon, mu, logvar):
        recon_loss = ((recon - x) ** 2).mean()
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def sample(self, n, device):
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

# Weights initialization
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
    print("  Variational Autoencoder (32x32) -- Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = VAE(latent_dim=128).to(device)
    model.apply(weights_init)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    
    dummy = torch.randn(4, 3, 32, 32).to(device)
    recon, mu, logvar = model(dummy)
    assert recon.shape == dummy.shape and mu.shape == (4, 128)
    model.loss(dummy, recon, mu, logvar)[0].backward()
    model.eval(); assert model.sample(4, device).shape == (4, 3, 32, 32)
    print("  All checks passed.")
