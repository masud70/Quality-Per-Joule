# ============================================================
# Quality per Joule — DDPM Training Script (CelebA 32x32)
# train/train_ddpm.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Reduce CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.celeba_loader import get_train_loader, denormalize
from models.ddpm import UNet, DDPMScheduler, weights_init
from utils.energy import get_gpu_power_watts, create_tracker, get_total_energy_joules
from utils.training import print_training_config, save_sample_grid
from config.config import DDPM_CONFIG as CONFIG

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

    model = UNet(base_ch=config["base_ch"], time_dim=config["time_dim"]).to(device)
    model.apply(weights_init)
    scheduler_diff = DDPMScheduler(T=config["T"], beta_start=config["beta_start"],
                                   beta_end=config["beta_end"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print_training_config(config, total_params, device)

    optimizer    = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"])

    project_name = f"qpj_{config['model_name']}"
    tracker = create_tracker(project_name, config["log_dir"])
    epoch_logs = []
    tracker.start()
    training_start = time.time()

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_power = []
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{config['epochs']}", leave=False)
        for x0, _ in pbar:
            x0    = x0.to(device)
            t     = torch.randint(0, config["T"], (x0.size(0),), device=device)
            noise = torch.randn_like(x0)
            x_t   = scheduler_diff.q_sample(x0, t, noise)

            eps_pred = model(x_t, t)
            loss     = F.mse_loss(eps_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pw = get_gpu_power_watts()
            if pw > 0:
                epoch_power.append(pw)
            pbar.set_postfix(loss=f"{loss.item():.4f}", gpu_w=f"{pw:.1f}W")

        scheduler_lr.step()

        epoch_time = time.time() - epoch_start
        avg_loss   = epoch_loss / len(train_loader)
        avg_power  = float(np.mean(epoch_power)) if epoch_power else 0.0
        energy_j   = avg_power * epoch_time

        epoch_logs.append({
            "epoch": epoch, "avg_loss": round(avg_loss, 6),
            "epoch_time_s": round(epoch_time, 2),
            "avg_power_w": round(avg_power, 2),
            "epoch_energy_j": round(energy_j, 2),
        })

        print(f"  Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
              f"{epoch_time:.1f}s | {avg_power:.1f}W | {energy_j:.1f}J")

        if epoch % 10 == 0:
            ckpt = os.path.join(config["checkpoint_dir"],
                                f"{config['model_name']}_epoch{epoch:03d}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt)
            print(f"  Generating 16 samples...")
            gen_imgs = scheduler_diff.sample(model, shape=(16, 3, 32, 32), device=device)
            samp = os.path.join(config["sample_dir"],
                                f"{config['model_name']}_samples_epoch{epoch:03d}.png")
            save_sample_grid(gen_imgs, samp,
                             title=f"DDPM Samples — Epoch {epoch}",
                             n=16, nrow=8, denorm_fn=denormalize)
            model.train()
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
        "T": config["T"], "epochs": config["epochs"],
        "total_training_time_s": round(total_time, 2),
        "total_energy_j": round(total_energy_j, 2),
        "avg_power_w": round(avg_power_all, 2),
        "codecarbon_kg_co2": round(emissions, 6) if emissions else None,
        "final_loss": epoch_logs[-1]["avg_loss"],
        "epoch_logs": epoch_logs, "config": config,
    }

    log_path = os.path.join(config["log_dir"], f"{config['model_name']}_energy.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    save_checkpoint(model, optimizer, config["epochs"], epoch_logs[-1]["avg_loss"],
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
    
    return model, scheduler_diff, summary

if __name__ == "__main__":
    train(CONFIG)
