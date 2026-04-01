# ============================================================
# Quality per Joule — DDPM (CelebA 32x32)
# models/ddpm.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Architecture adapted from ddpm.ipynb
#
# Modifications from notebook:
#   - img_size: 64 -> 32
#   - first_conv_channels: 64 -> 32 (scaled for smaller images)
#   - channel_multiplier: [1,2,4,8] (same as notebook)
#   - widths: [32,64,128,256] (was [64,128,256,512])
#   - has_attention: [False, False, True, True] (same as notebook)
#   - num_res_blocks: 2 (same as notebook)
#   - Keras -> PyTorch translation
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal Timestep Embedding
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# MLP for Time Embedding
class TimeMLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res_conv(x)


# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch)
        self.q = nn.Linear(ch, ch)
        self.k = nn.Linear(ch, ch)
        self.v = nn.Linear(ch, ch)
        self.proj = nn.Linear(ch, ch)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(attn, v)
        out = self.proj(out)
        return x + out.permute(0, 2, 1).view(B, C, H, W)


# U-Net for DDPM
class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        channel_multiplier: list = None,
        has_attention: list = None,
        num_res_blocks: int = 2,
        norm_groups: int = 8,
        time_dim: int = None,
    ):
        super().__init__()
        if channel_multiplier is None:
            channel_multiplier = [1, 2, 4, 8]
        if has_attention is None:
            has_attention = [False, False, True, True]
        if time_dim is None:
            time_dim = base_ch * 4

        widths = [base_ch * m for m in channel_multiplier]
        num_levels = len(widths)

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            TimeMLP(time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Encoder (DownBlocks)
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = base_ch
        skip_channels = [base_ch]

        for i in range(num_levels):
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()
            for _ in range(num_res_blocks):
                block_list.append(ResidualBlock(ch, widths[i], time_dim, norm_groups))
                ch = widths[i]
                if has_attention[i]:
                    attn_list.append(AttentionBlock(ch, norm_groups))
                else:
                    attn_list.append(nn.Identity())
                skip_channels.append(ch)
            self.down_blocks.append(nn.ModuleDict({
                'res': block_list, 'attn': attn_list
            }))
            if i < num_levels - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                skip_channels.append(ch)
            else:
                self.down_samples.append(None)

        # Middle Block
        self.mid_res1 = ResidualBlock(ch, ch, time_dim, norm_groups)
        self.mid_attn = AttentionBlock(ch, norm_groups)
        self.mid_res2 = ResidualBlock(ch, ch, time_dim, norm_groups)

        # Decoder (UpBlocks)
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in reversed(range(num_levels)):
            block_list = nn.ModuleList()
            attn_list = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                block_list.append(ResidualBlock(ch + skip_ch, widths[i], time_dim, norm_groups))
                ch = widths[i]
                if has_attention[i]:
                    attn_list.append(AttentionBlock(ch, norm_groups))
                else:
                    attn_list.append(nn.Identity())
            self.up_blocks.append(nn.ModuleDict({
                'res': block_list, 'attn': attn_list
            }))
            if i > 0:
                self.up_samples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))
            else:
                self.up_samples.append(None)

        # Output
        self.out_norm = nn.GroupNorm(norm_groups, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        x = self.init_conv(x)
        skips = [x]

        # Encoder
        for i, (block, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for res, attn in zip(block['res'], block['attn']):
                x = res(x, t_emb)
                x = attn(x)
                skips.append(x)
            if downsample is not None:
                x = downsample(x)
                skips.append(x)

        # Middle
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        # Decoder
        for block, upsample in zip(self.up_blocks, self.up_samples):
            for res, attn in zip(block['res'], block['attn']):
                x = torch.cat([x, skips.pop()], dim=1)
                x = res(x, t_emb)
                x = attn(x)
            if upsample is not None:
                x = upsample(x)

        return self.out_conv(F.silu(self.out_norm(x)))


# Noise Scheduler for DDPM
class DDPMScheduler:
    """
    Notebook: GaussianDiffusion — linear beta schedule.
    """
    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t_scalar):
        B, device = x_t.size(0), x_t.device
        t_vec = torch.full((B,), t_scalar, dtype=torch.long, device=device)
        beta_t = self.betas[t_scalar]
        alpha_t = self.alphas[t_scalar]
        alpha_bar_t = self.alpha_bar[t_scalar]
        eps_pred = model(x_t, t_vec)
        coef = beta_t / (1 - alpha_bar_t).sqrt()
        mean = (x_t - coef * eps_pred) / alpha_t.sqrt()
        if t_scalar == 0:
            return mean
        return mean + beta_t.sqrt() * torch.randn_like(x_t)

    @torch.no_grad()
    def sample(self, model, shape, device):
        model.eval()
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x


# Weights initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d,)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GroupNorm) and m.weight is not None:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ============================================================
####################### Sanity Check #########################
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  DDPM U-Net (32x32) -- Sanity Check")
    print("=" * 55)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    T = 1000
    model = UNet(base_ch=32, time_dim=128).to(device)
    scheduler = DDPMScheduler(T=T).to(device)
    model.apply(weights_init)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    B = 2
    x0 = torch.randn(B, 3, 32, 32).to(device)
    t = torch.randint(0, T, (B,)).to(device)
    noise = torch.randn_like(x0)
    x_t = scheduler.q_sample(x0, t, noise)

    eps_pred = model(x_t, t)
    assert eps_pred.shape == x0.shape, f"Shape mismatch: {eps_pred.shape}"
    print(f"  Forward: {eps_pred.shape}  PASS")

    loss = F.mse_loss(eps_pred, noise)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}  Backward: PASS")

    samples = scheduler.sample(model, (1, 3, 32, 32), device)
    assert samples.shape == (1, 3, 32, 32)
    print(f"  Sample: {samples.shape}  PASS")
    print("  All checks passed.")