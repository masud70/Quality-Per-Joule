# ============================================================
# Quality per Joule — Convolutional Autoencoder (CelebA 32x32)
# models/autoencoder.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Architecture adapted from Program_5_AE_mnist.ipynb:
#   Encoder: Conv(32,3,s=2) -> Conv(64,3,s=2) -> Flatten -> Dense(16,relu) -> Dense(z)
#   Decoder: Dense(8*8*64,relu) -> Reshape -> ConvT(64,3,s=2) -> ConvT(32,3,s=2) -> ConvT(3,3,s=1)
#
# Modifications from notebook:
#   - Input changed from (28,28,1) to (32,32,3) for CelebA
#   - Flatten dim: 8*8*64 = 4096 (was 7*7*64 = 3136)
#   - Output channels: 3 (was 1)
#   - Output activation: Tanh (was Sigmoid) to match [-1,1] normalization
#   - latent_dim default: 128 (was 2)
# ============================================================

import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Decoder
class Decoder(nn.Module):
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
        x = self.deconv_layers(x)
        return x


# Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z     = self.encode(x)
        recon = self.decode(z)
        return recon, z


# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight"):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif "Linear" in classname and hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ============================================================
####################### Sanity Check #########################
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  Convolutional Autoencoder (32x32) -- Sanity Check")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    model = ConvAutoencoder(latent_dim=128).to(device)
    model.apply(weights_init)

    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"\n  Encoder params : {encoder_params:,}")
    print(f"  Decoder params : {decoder_params:,}")
    print(f"  Total params   : {total_params:,}")

    dummy = torch.randn(4, 3, 32, 32).to(device)
    recon, z = model(dummy)

    print(f"\n  Input shape    : {dummy.shape}")
    print(f"  Latent shape   : {z.shape}   (expected: [4, 128])")
    print(f"  Output shape   : {recon.shape}  (expected: [4, 3, 32, 32])")
    assert dummy.shape == recon.shape, "Input/output shape mismatch!"
    assert z.shape == (4, 128), "Latent shape mismatch!"
    print(f"  Shape check    : PASS")

    assert recon.min() >= -1.0 and recon.max() <= 1.0, "Output out of [-1,1]!"
    print(f"  Output range   : PASS  (min={recon.min():.3f}, max={recon.max():.3f})")

    loss = ((recon - dummy) ** 2).mean()
    loss.backward()
    print(f"\n  MSE loss       : {loss.item():.4f}")
    print(f"  Backward pass  : PASS")

    print("\n  All checks passed. Autoencoder is ready for training.")
    print("=" * 55)