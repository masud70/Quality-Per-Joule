# ============================================================
# Quality per Joule — WGAN-GP Training Script (CelebA 32x32)
# train/train_wgan_gp.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import os
import sys
import time
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.celeba_loader import get_train_loader, denormalize
from models.wgan_gp import Generator, Critic, critic_weights_init, gradient_penalty
from utils.energy import get_gpu_power_watts, create_tracker, get_total_energy_joules
from utils.training import print_training_config, save_sample_grid
from config.config import WGAN_GP_CONFIG as CONFIG

# Checkpoint saving function
def save_checkpoint(netG, critic, optG, optC, epoch, path):
    torch.save({"epoch": epoch, "netG_state": netG.state_dict(),
                "critic_state": critic.state_dict(), "optG_state": optG.state_dict(),
                "optC_state": optC.state_dict()}, path)

# Training loop with energy tracking
def train(config=CONFIG):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d in [config["checkpoint_dir"], config["log_dir"], config["sample_dir"]]:
        os.makedirs(d, exist_ok=True)

    train_loader = get_train_loader(batch_size=config["batch_size"], seed=config["seed"])

    netG   = Generator(latent_dim=config["latent_dim"]).to(device)
    critic = Critic().to(device)
    netG.apply(critic_weights_init)
    critic.apply(critic_weights_init)
    total_params = sum(p.numel() for p in netG.parameters()) + \
                   sum(p.numel() for p in critic.parameters())

    print_training_config(config, total_params, device)

    optG = torch.optim.Adam(netG.parameters(), lr=config["lr_g"],
                            betas=(config["beta1"], config["beta2"]))
    optC = torch.optim.Adam(critic.parameters(), lr=config["lr_c"],
                            betas=(config["beta1"], config["beta2"]))

    fixed_noise = torch.randn(16, config["latent_dim"]).to(device)

    project_name = f"qpj_{config['model_name']}"
    tracker = create_tracker(project_name, config["log_dir"])
    epoch_logs = []
    tracker.start()
    training_start = time.time()

    data_iter  = iter(train_loader)

    for epoch in range(1, config["epochs"] + 1):
        netG.train(); critic.train()
        sum_C = sum_G = sum_GP = 0.0
        epoch_power = []
        epoch_start = time.time()
        n_batches   = len(train_loader)
        g_steps     = 0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False)
        for _ in pbar:
            # Critic updates
            for _ in range(config["n_critic"]):
                try:
                    real_imgs, _ = next(data_iter)
                except StopIteration:
                    data_iter    = iter(train_loader)
                    real_imgs, _ = next(data_iter)

                real_imgs = real_imgs.to(device)
                B         = real_imgs.size(0)
                noise     = torch.randn(B, config["latent_dim"]).to(device)

                optC.zero_grad()
                fake_imgs = netG(noise).detach()
                gp        = gradient_penalty(critic, real_imgs, fake_imgs,
                                             device, config["lambda_gp"])
                loss_C    = critic(fake_imgs).mean() - critic(real_imgs).mean() + gp
                loss_C.backward()
                optC.step()
                sum_C  += loss_C.item()
                sum_GP += gp.item()

            # Generator update
            optG.zero_grad()
            noise     = torch.randn(B, config["latent_dim"]).to(device)
            fake_imgs = netG(noise)
            loss_G    = -critic(fake_imgs).mean()
            loss_G.backward()
            optG.step()
            sum_G += loss_G.item()
            g_steps += 1

            pw = get_gpu_power_watts()
            if pw > 0:
                epoch_power.append(pw)
            pbar.set_postfix(C=f"{loss_C.item():.3f}",
                             G=f"{loss_G.item():.3f}",
                             GP=f"{gp.item():.3f}")

        epoch_time = time.time() - epoch_start
        avg_power  = float(np.mean(epoch_power)) if epoch_power else 0.0
        energy_j   = avg_power * epoch_time

        epoch_logs.append({
            "epoch"         : epoch,
            "avg_loss_C"    : round(sum_C  / (n_batches * config["n_critic"]), 6),
            "avg_loss_G"    : round(sum_G  / max(g_steps, 1), 6),
            "avg_gp"        : round(sum_GP / (n_batches * config["n_critic"]), 6),
            "epoch_time_s"  : round(epoch_time, 2),
            "avg_power_w"   : round(avg_power,  2),
            "epoch_energy_j": round(energy_j,   2),
        })

        print(f"  Epoch {epoch:03d} | C: {epoch_logs[-1]['avg_loss_C']:.4f} | "
              f"G: {epoch_logs[-1]['avg_loss_G']:.4f} | "
              f"GP: {epoch_logs[-1]['avg_gp']:.4f} | "
              f"{epoch_time:.1f}s | {avg_power:.1f}W | {energy_j:.1f}J")

        if epoch % 10 == 0:
            ckpt = os.path.join(config["checkpoint_dir"],
                                f"{config['model_name']}_epoch{epoch:03d}.pt")
            save_checkpoint(netG, critic, optG, optC, epoch, ckpt)
            netG.eval()
            with torch.no_grad():
                fake_imgs = netG(fixed_noise)
            samp = os.path.join(config["sample_dir"],
                                f"{config['model_name']}_samples_epoch{epoch:03d}.png")
            save_sample_grid(fake_imgs, samp,
                             title=f"WGAN-GP Samples — Epoch {epoch}",
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
        "final_loss_C": epoch_logs[-1]["avg_loss_C"],
        "epoch_logs": epoch_logs, "config": config,
    }

    log_path = os.path.join(config["log_dir"], f"{config['model_name']}_energy.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    save_checkpoint(netG, critic, optG, optC, config["epochs"],
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

    return netG, critic, summary

if __name__ == "__main__":
    train(CONFIG)