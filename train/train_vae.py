# ============================================================
# Quality per Joule — VAE Training Script (CelebA 32x32)
# train/train_vae.py
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
from data.celeba_loader import get_train_loader, get_test_loader, denormalize
from models.vae import VAE, weights_init
from utils.energy import get_gpu_power_watts, create_tracker, get_total_energy_joules
from utils.training import print_training_config, save_sample_grid
from config.config import VAE_CONFIG as CONFIG

# Checkpoint saving function
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({"epoch": epoch, "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(), "loss": loss}, path)

# Training loop with energy tracking
def train(config=CONFIG):
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d in [config["checkpoint_dir"], config["log_dir"], config["sample_dir"]]:
        os.makedirs(d, exist_ok=True)

    train_loader = get_train_loader(batch_size=config["batch_size"], seed=config["seed"])
    test_loader  = get_test_loader(batch_size=config["batch_size"],  seed=config["seed"])

    model = VAE(latent_dim=config["latent_dim"], beta=config["beta"]).to(device)
    model.apply(weights_init)
    total_params = sum(p.numel() for p in model.parameters())

    print_training_config(config, total_params, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    project_name = f"qpj_{config['model_name']}"
    tracker = create_tracker(project_name, config["log_dir"])
    epoch_logs = []
    tracker.start()
    training_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        sum_loss = sum_recon = sum_kl = 0.0
        epoch_power = []
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False)
        for images, _ in pbar:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = model.loss(images, recon, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss  += loss.item()
            sum_recon += recon_loss.item()
            sum_kl    += kl_loss.item()
            pw = get_gpu_power_watts()
            if pw > 0:
                epoch_power.append(pw)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             recon=f"{recon_loss.item():.4f}",
                             kl=f"{kl_loss.item():.4f}")

        scheduler.step()

        # Validation
        model.eval()
        val_loss = val_recon = val_kl = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                recon, mu, logvar = model(images)
                tl, rl, kl = model.loss(images, recon, mu, logvar)
                val_loss  += tl.item()
                val_recon += rl.item()
                val_kl    += kl.item()
        n = len(test_loader)
        val_loss /= n; val_recon /= n; val_kl /= n

        epoch_time = time.time() - epoch_start
        n_train    = len(train_loader)
        avg_power  = float(np.mean(epoch_power)) if epoch_power else 0.0
        energy_j   = avg_power * epoch_time

        epoch_logs.append({
            "epoch"          : epoch,
            "train_loss"     : round(sum_loss  / n_train, 6),
            "train_recon"    : round(sum_recon / n_train, 6),
            "train_kl"       : round(sum_kl    / n_train, 6),
            "val_loss"       : round(val_loss,  6),
            "val_recon"      : round(val_recon, 6),
            "val_kl"         : round(val_kl,    6),
            "epoch_time_s"   : round(epoch_time, 2),
            "avg_power_w"    : round(avg_power,  2),
            "epoch_energy_j" : round(energy_j,   2),
        })

        print(f"  Epoch {epoch:03d} | Loss: {sum_loss/n_train:.4f} | "
              f"Recon: {sum_recon/n_train:.4f} | KL: {sum_kl/n_train:.4f} | "
              f"{epoch_time:.1f}s | {avg_power:.1f}W | {energy_j:.1f}J")

        if epoch % 10 == 0:
            ckpt = os.path.join(config["checkpoint_dir"],
                                f"{config['model_name']}_epoch{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, sum_loss / n_train, ckpt)
            model.eval()
            with torch.no_grad():
                gen_imgs = model.sample(16, device)
            samp = os.path.join(config["sample_dir"],
                                f"{config['model_name']}_samples_epoch{epoch:03d}.png")
            save_sample_grid(gen_imgs, samp,
                             title=f"VAE Samples — Epoch {epoch}",
                             n=16, nrow=8, denorm_fn=denormalize)
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
        "final_train_loss": epoch_logs[-1]["train_loss"],
        "final_val_loss": epoch_logs[-1]["val_loss"],
        "epoch_logs": epoch_logs, "config": config,
    }

    log_path = os.path.join(config["log_dir"], f"{config['model_name']}_energy.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    save_checkpoint(model, optimizer, config["epochs"], epoch_logs[-1]["train_loss"],
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
    
    return model, summary

if __name__ == "__main__":
    train(CONFIG)