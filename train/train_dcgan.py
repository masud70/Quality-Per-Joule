# ============================================================
# Quality per Joule — DCGAN Training Script (CelebA 32x32)
# train/train_dcgan.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.celeba_loader import get_train_loader, denormalize
from models.dcgan import Generator, Discriminator, weights_init
from utils.energy import get_gpu_power_watts, create_tracker, get_total_energy_joules
from utils.training import print_training_config, save_sample_grid
from config.config import DCGAN_CONFIG as CONFIG

# Checkpoint saving function
def save_checkpoint(netG, netD, optG, optD, epoch, path):
    torch.save({"epoch": epoch, "netG_state": netG.state_dict(),
                "netD_state": netD.state_dict(), "optG_state": optG.state_dict(),
                "optD_state": optD.state_dict()}, path)

# Training loop with energy tracking
def train(config=CONFIG):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d in [config["checkpoint_dir"], config["log_dir"], config["sample_dir"]]:
        os.makedirs(d, exist_ok=True)

    train_loader = get_train_loader(batch_size=config["batch_size"], seed=config["seed"])

    netG = Generator(latent_dim=config["latent_dim"]).to(device)
    netD = Discriminator().to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    total_params = sum(p.numel() for p in netG.parameters()) + \
                   sum(p.numel() for p in netD.parameters())

    print_training_config(config, total_params, device)

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=config["lr_g"],
                            betas=(config["beta1"], config["beta2"]))
    optD = torch.optim.Adam(netD.parameters(), lr=config["lr_d"],
                            betas=(config["beta1"], config["beta2"]))

    fixed_noise = torch.randn(16, config["latent_dim"]).to(device)

    project_name = f"qpj_{config['model_name']}"
    tracker = create_tracker(project_name, config["log_dir"])
    epoch_logs = []
    tracker.start()
    training_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        netG.train(); netD.train()
        sum_G = sum_D = 0.0
        epoch_power = []
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False)
        for real_imgs, _ in pbar:
            real_imgs = real_imgs.to(device)
            B = real_imgs.size(0)

            # Train D
            netD.zero_grad()
            labels = torch.full((B,), config["real_label"], dtype=torch.float, device=device)
            loss_D_real = criterion(netD(real_imgs), labels)
            loss_D_real.backward()
            noise     = torch.randn(B, config["latent_dim"]).to(device)
            fake_imgs = netG(noise)
            labels.fill_(config["fake_label"])
            loss_D_fake = criterion(netD(fake_imgs.detach()), labels)
            loss_D_fake.backward()
            optD.step()

            # Train G
            netG.zero_grad()
            labels.fill_(config["real_label"])
            loss_G = criterion(netD(fake_imgs), labels)
            loss_G.backward()
            optG.step()

            sum_D += (loss_D_real + loss_D_fake).item()
            sum_G += loss_G.item()
            pw = get_gpu_power_watts()
            if pw > 0:
                epoch_power.append(pw)
            pbar.set_postfix(lossD=f"{(loss_D_real+loss_D_fake).item():.3f}",
                             lossG=f"{loss_G.item():.3f}")

        epoch_time = time.time() - epoch_start
        avg_power  = float(np.mean(epoch_power)) if epoch_power else 0.0
        energy_j   = avg_power * epoch_time
        n          = len(train_loader)

        epoch_logs.append({
            "epoch": epoch, "avg_loss_G": round(sum_G/n, 6),
            "avg_loss_D": round(sum_D/n, 6), "epoch_time_s": round(epoch_time, 2),
            "avg_power_w": round(avg_power, 2), "epoch_energy_j": round(energy_j, 2),
        })

        print(f"  Epoch {epoch:03d} | G: {sum_G/n:.4f} | D: {sum_D/n:.4f} | "
              f"{epoch_time:.1f}s | {avg_power:.1f}W | {energy_j:.1f}J")

        if epoch % 10 == 0:
            ckpt = os.path.join(config["checkpoint_dir"],
                                f"{config['model_name']}_epoch{epoch:03d}.pt")
            save_checkpoint(netG, netD, optG, optD, epoch, ckpt)
            netG.eval()
            with torch.no_grad():
                fake_imgs = netG(fixed_noise)
            samp = os.path.join(config["sample_dir"],
                                f"{config['model_name']}_samples_epoch{epoch:03d}.png")
            save_sample_grid(fake_imgs, samp,
                             title=f"DCGAN Samples — Epoch {epoch}",
                             n=16, nrow=8, denorm_fn=denormalize)
            netG.train()
            print(f"  Checkpoint: {ckpt}")

    emissions      = tracker.stop()
    total_time     = time.time() - training_start

    cc_energy_j    = get_total_energy_joules(config["log_dir"], project_name)
    nvml_energy_j  = sum(e["epoch_energy_j"] for e in epoch_logs)
    # Prefer CodeCarbon energy if available, otherwise use NVML-based estimate
    total_energy_j = cc_energy_j if cc_energy_j > 0 else nvml_energy_j

    valid_powers   = [e["avg_power_w"] for e in epoch_logs if e["avg_power_w"] > 0]
    avg_power_all  = float(np.mean(valid_powers)) if valid_powers else (
        total_energy_j / total_time if total_time > 0 else 0.0
    )

    summary = {
        "model": config["model_name"], "total_params": total_params,
        "epochs": config["epochs"],
        "total_training_time_s": round(total_time, 2),
        "total_energy_j": round(total_energy_j, 2),
        "avg_power_w": round(avg_power_all, 2),
        "codecarbon_kg_co2": round(emissions, 6) if emissions else None,
        "final_loss_G": epoch_logs[-1]["avg_loss_G"],
        "final_loss_D": epoch_logs[-1]["avg_loss_D"],
        "epoch_logs": epoch_logs, "config": config,
    }

    log_path = os.path.join(config["log_dir"], f"{config['model_name']}_energy.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    save_checkpoint(netG, netD, optG, optD, config["epochs"],
                    os.path.join(config["checkpoint_dir"], f"{config['model_name']}_final.pt"))

    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Total time   : {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Total energy : {total_energy_j:.1f} J")
    print(f"  Avg power    : {avg_power_all:.1f} W")
    print(f"  Final val    : {epoch_logs[-1]['val_loss']:.4f}")
    print(f"  CO2 (kg)     : {emissions:.6f}" if emissions else "  CO2 (kg) : N/A")
    print("=" * 50)

    return netG, netD, summary

if __name__ == "__main__":
    train(CONFIG)
