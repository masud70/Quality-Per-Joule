# ============================================================
# Quality per Joule — Evaluation Script (CelebA 32x32)
# evaluate/evaluate.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.celeba_loader import get_test_loader, get_fid_reference_loader, denormalize
from models.autoencoder import ConvAutoencoder
from models.vae import VAE
from models.dcgan import Generator as DCGANGenerator
from models.wgan_gp import Generator as WGANGenerator
from models.ddpm import UNet, DDPMScheduler
from config.config import EVAL_CONFIG as CONFIG


# Utility functions
def to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    return (denormalize(tensor).clamp(0, 1) * 255).to(torch.uint8)

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# FID computation
def compute_fid(real_loader, fake_images: torch.Tensor, device) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    print("  FID: feeding real images...")
    for imgs, _ in tqdm(real_loader, leave=False):
        fid.update(to_uint8(imgs).to(device), real=True)
    print("  FID: feeding generated images...")
    bs = CONFIG["batch_size"]
    for i in tqdm(range(0, len(fake_images), bs), leave=False):
        fid.update(to_uint8(fake_images[i:i+bs]).to(device), real=False)
    return fid.compute().item()


# Model sample generation (AE)
def generate_ae(device, n):
    ckpt = os.path.join(CONFIG["checkpoint_dir"], "autoencoder_final.pt")
    model = ConvAutoencoder(latent_dim=CONFIG["latent_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()
    loader = get_test_loader(batch_size=CONFIG["batch_size"], seed=CONFIG["seed"])
    samples, mse_total, count = [], 0.0, 0
    print(f"  Generating {n} AE reconstructions...")
    with torch.no_grad():
        for imgs, _ in tqdm(loader, leave=False):
            if count >= n:
                break
            imgs = imgs.to(device)
            recon, _ = model(imgs)
            mse_total += ((recon - imgs) ** 2).mean().item() * imgs.size(0)
            samples.append(recon.cpu())
            count += imgs.size(0)
    recon_mse = mse_total / count
    print(f"  AE reconstruction MSE: {recon_mse:.4f}")
    return torch.cat(samples)[:n], recon_mse

# Model sample generation (VAE)
def generate_vae(device, n):
    ckpt = os.path.join(CONFIG["checkpoint_dir"], "vae_final.pt")
    model = VAE(latent_dim=CONFIG["vae_latent_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    model.eval()
    samples, generated = [], 0
    bs = CONFIG["batch_size"]
    print(f"  Generating {n} VAE samples from prior...")
    with torch.no_grad():
        for _ in tqdm(range(0, n, bs), leave=False):
            cur = min(bs, n - generated)
            samples.append(model.sample(cur, device).cpu())
            generated += cur
    return torch.cat(samples)[:n], None

# Model sample generation (DCGAN)
def generate_dcgan(device, n):
    ckpt  = os.path.join(CONFIG["checkpoint_dir"], "dcgan_final.pt")
    netG  = DCGANGenerator(latent_dim=CONFIG["latent_dim"]).to(device)
    netG.load_state_dict(torch.load(ckpt, map_location=device)["netG_state"])
    netG.eval()
    samples, generated = [], 0
    bs = CONFIG["batch_size"]
    print(f"  Generating {n} DCGAN samples...")
    with torch.no_grad():
        for _ in tqdm(range(0, n, bs), leave=False):
            cur = min(bs, n - generated)
            z = torch.randn(cur, CONFIG["latent_dim"]).to(device)
            samples.append(netG(z).cpu())
            generated += cur
    return torch.cat(samples)[:n], None

# Model sample generation (WGAN-GP)
def generate_wgan_gp(device, n):
    ckpt  = os.path.join(CONFIG["checkpoint_dir"], "wgan_gp_final.pt")
    netG  = WGANGenerator(latent_dim=CONFIG["latent_dim"]).to(device)
    netG.load_state_dict(torch.load(ckpt, map_location=device)["netG_state"])
    netG.eval()
    samples, generated = [], 0
    bs = CONFIG["batch_size"]
    print(f"  Generating {n} WGAN-GP samples...")
    with torch.no_grad():
        for _ in tqdm(range(0, n, bs), leave=False):
            cur = min(bs, n - generated)
            z = torch.randn(cur, CONFIG["latent_dim"]).to(device)
            samples.append(netG(z).cpu())
            generated += cur
    return torch.cat(samples)[:n], None

# Model sample generation (DDPM)
def generate_ddpm(device, n):
    ckpt  = os.path.join(CONFIG["checkpoint_dir"], "ddpm_final.pt")
    model = UNet(base_ch=CONFIG["base_ch"], time_dim=CONFIG["time_dim"]).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    scheduler = DDPMScheduler(T=CONFIG["T"]).to(device)
    samples, generated = [], 0
    bs = 64   # smaller batches — reverse diffusion is memory-intensive
    print(f"  Generating {n} DDPM samples (T={CONFIG['T']} steps)...")
    with torch.no_grad():
        for _ in tqdm(range(0, n, bs), leave=False):
            cur = min(bs, n - generated)
            samples.append(scheduler.sample(model, (cur, 3, 32, 32), device))
            generated += cur
    return torch.cat(samples)[:n].cpu(), None


# Main evaluation method
def evaluate():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    print(f"  FID samples per model : {CONFIG['n_fid_samples']}\n")

    fid_ref_loader = get_fid_reference_loader(
        n_samples=CONFIG["n_fid_samples"],
        batch_size=CONFIG["batch_size"],
        seed=CONFIG["seed"]
    )

    models_to_eval = {
        "autoencoder": generate_ae,
        "vae"        : generate_vae,
        "dcgan"      : generate_dcgan,
        "wgan_gp"    : generate_wgan_gp,
        "ddpm"       : generate_ddpm,
    }

    results = {}
    for model_name, generate_fn in models_to_eval.items():
        print("=" * 50)
        print(f"  Evaluating: {model_name.upper()}")
        print("=" * 50)

        energy_log = os.path.join(CONFIG["log_dir"], f"{model_name}_energy.json")
        assert os.path.exists(energy_log), \
            f"Energy log not found: {energy_log}. Train the model first."
        energy_data  = load_json(energy_log)
        total_energy = energy_data["total_energy_j"]
        total_time   = energy_data["total_training_time_s"]
        avg_power    = energy_data["avg_power_w"]

        fake_images, recon_mse = generate_fn(device, CONFIG["n_fid_samples"])

        # Free GPU memory between models
        torch.cuda.empty_cache()

        fid_score = compute_fid(fid_ref_loader, fake_images, device)
        torch.cuda.empty_cache()

        qpj = 1.0 / (fid_score * total_energy) if fid_score > 0 and total_energy > 0 else 0.0

        results[model_name] = {
            "fid"              : round(fid_score, 4),
            "total_energy_j"   : round(total_energy, 2),
            "total_time_s"     : round(total_time, 2),
            "avg_power_w"      : round(avg_power, 2),
            "quality_per_joule": round(qpj, 8),
            "recon_mse"        : round(recon_mse, 6) if recon_mse else None,
            "total_params"     : energy_data.get("total_params"),
        }

        print(f"\n  FID              : {fid_score:.4f}")
        print(f"  Total energy (J) : {total_energy:.2f}")
        print(f"  Quality per Joule: {qpj:.8f}")
        if recon_mse:
            print(f"  Reconstruction MSE: {recon_mse:.4f}")
        print()

    # Display summary table and best model
    print("\n" + "=" * 65)
    print(f"  {'Model':<15} {'FID':>8} {'Energy(kJ)':>12} {'QpJ×10⁻⁷':>14}")
    print("=" * 65)
    for name, r in results.items():
        print(f"  {name:<15} {r['fid']:>8.4f} "
              f"{r['total_energy_j']/1000:>12.1f} "
              f"{r['quality_per_joule']*1e7:>14.6f}")
    print("=" * 65)
    best = max(results, key=lambda k: results[k]["quality_per_joule"])
    print(f"\n  Best Quality per Joule: {best.upper()}")

    os.makedirs(os.path.dirname(CONFIG["output_path"]), exist_ok=True)
    with open(CONFIG["output_path"], "w") as f:
        json.dump({"config": CONFIG, "results": results}, f, indent=2)
    print(f"\n  Results saved to: {CONFIG['output_path']}")
    return results


if __name__ == "__main__":
    evaluate()