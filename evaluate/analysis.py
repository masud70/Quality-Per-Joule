# ============================================================
# Quality per Joule — Analysis & Visualization
# evaluate/analysis.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config.config import RESULTS_PATH, LOG_DIR, PLOT_DIR
os.makedirs(PLOT_DIR, exist_ok=True)

# Colors and labels for consistent plotting
COLORS = {
    "autoencoder": "#4C72B0",
    "vae"        : "#C44E52",
    "dcgan"      : "#DD8452",
    "wgan_gp"    : "#8172B3",
    "ddpm"       : "#55A868",
}
LABELS = {
    "autoencoder": "Autoencoder",
    "vae"        : "VAE",
    "dcgan"      : "DCGAN",
    "wgan_gp"    : "WGAN-GP",
    "ddpm"       : "DDPM",
}


# Load results and logs
def load_results():
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return data["results"]

def load_epoch_logs(model_name):
    path = os.path.join(LOG_DIR, f"{model_name}_energy.json")
    with open(path) as f:
        return json.load(f)["epoch_logs"]


# ==============================================================
# Plot 1 — FID vs Energy Scatter
# ==============================================================
def plot_fid_vs_energy(results):
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, r in results.items():
        ax.scatter(
            r["total_energy_j"] / 1000,   # kJ for readability
            r["fid"],
            color=COLORS[name],
            s=220, zorder=5,
            edgecolors="white", linewidths=1.5,
            label=LABELS[name]
        )
        ax.annotate(
            LABELS[name],
            xy=(r["total_energy_j"] / 1000, r["fid"]),
            xytext=(8, 6), textcoords="offset points",
            fontsize=10, color=COLORS[name], fontweight="bold"
        )

    ax.set_xlabel("Total Training Energy (kJ)", fontsize=12)
    ax.set_ylabel("FID Score (lower = better)", fontsize=12)
    ax.set_title("Generative Quality vs. Training Energy", fontsize=13, fontweight="bold")
    ax.invert_yaxis()   # Lower FID is better, so invert so "better" is up
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Annotate ideal quadrant
    ax.text(
        0.03, 0.05,
        "← Less energy\n↑ Better quality\n(ideal region)",
        transform=ax.transAxes, fontsize=8,
        color="gray", va="bottom"
    )

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "fid_vs_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ==============================================================
# Plot 2 — Quality per Joule Bar Chart
# ==============================================================
def plot_quality_per_joule(results):
    names  = list(results.keys())
    qpj    = [results[n]["quality_per_joule"] * 1e7 for n in names]   # scale for readability
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        [LABELS[n] for n in names], qpj,
        color=colors, edgecolor="white", linewidth=1.2, width=0.5
    )

    # Value labels on bars
    for bar, val in zip(bars, qpj):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_ylabel("Quality per Joule  (×10⁻⁷)", fontsize=11)
    ax.set_title("Quality per Joule Comparison\n"
                 r"$\mathbf{QpJ} = \frac{1}{\mathrm{FID} \times \mathrm{Energy}}$",
                 fontsize=12)
    ax.set_ylim(0, max(qpj) * 1.25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "quality_per_joule.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ==============================================================
# Plot 3 — Training Loss Curves
# ==============================================================
def plot_loss_curves():
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)

    configs = [
        ("autoencoder", "val_loss",   "Val Loss (MSE)",       "Autoencoder"),
        ("vae",         "val_loss",   "Val Loss (MSE+KL)",    "VAE"),
        ("dcgan",       "avg_loss_G", "BCE Loss",             "DCGAN — G vs D"),
        ("wgan_gp",     "avg_loss_G", "Wasserstein Loss",     "WGAN-GP — G vs C"),
        ("ddpm",        "avg_loss",   "Noise Pred Loss (MSE)","DDPM"),
    ]

    for ax, (model, loss_key, ylabel, title) in zip(axes, configs):
        logs   = load_epoch_logs(model)
        epochs = [e["epoch"] for e in logs]
        loss   = [e[loss_key] for e in logs]
        ax.plot(epochs, loss, color=COLORS[model], linewidth=2, label=loss_key)

        # For DCGAN, also plot D loss
        if model == "dcgan":
            loss_d = [e["avg_loss_D"] for e in logs]
            ax.plot(epochs, loss_d, color="tomato", linewidth=2,
                    linestyle="--", label="avg_loss_D")
            ax.legend(fontsize=7)

        # For WGAN-GP, also plot Critic loss
        if model == "wgan_gp":
            loss_c = [e["avg_loss_C"] for e in logs]
            ax.plot(epochs, loss_c, color="tomato", linewidth=2,
                    linestyle="--", label="avg_loss_C")
            ax.legend(fontsize=7)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Training Loss Curves", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ==============================================================
# Plot 4 — Energy per Epoch Over Training
# ==============================================================
def plot_energy_over_training():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for model in ["autoencoder", "vae", "dcgan", "wgan_gp", "ddpm"]:
        logs    = load_epoch_logs(model)
        epochs  = [e["epoch"] for e in logs]
        cum_energy = np.cumsum([e["epoch_energy_j"] / 1000 for e in logs])  # kJ
        ax.plot(epochs, cum_energy, color=COLORS[model],
                linewidth=2.5, label=LABELS[model])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cumulative Energy (kJ)", fontsize=12)
    ax.set_title("Cumulative Training Energy over Time", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "cumulative_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ==============================================================
# Plot 5 — Summary Dashboard (all metrics in one figure)
# ==============================================================
def plot_summary_dashboard(results):
    fig = plt.figure(figsize=(14, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    names  = list(results.keys())
    labels = [LABELS[n] for n in names]
    colors = [COLORS[n] for n in names]

    # (0,0) FID scores
    ax1 = fig.add_subplot(gs[0, 0])
    fids = [results[n]["fid"] for n in names]
    bars = ax1.bar(labels, fids, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars, fids):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_title("FID Score\n(lower = better)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("FID", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.spines[["top","right"]].set_visible(False)

    # (0,1) Total energy
    ax2 = fig.add_subplot(gs[0, 1])
    energies = [results[n]["total_energy_j"]/1000 for n in names]
    bars = ax2.bar(labels, energies, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars, energies):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Total Training Energy\n(lower = better)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Energy (kJ)", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.spines[["top","right"]].set_visible(False)

    # (0,2) Quality per Joule
    ax3 = fig.add_subplot(gs[0, 2])
    qpj = [results[n]["quality_per_joule"]*1e7 for n in names]
    bars = ax3.bar(labels, qpj, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars, qpj):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
                 f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("Quality per Joule ×10⁻⁷\n(higher = better)", fontsize=10, fontweight="bold")
    ax3.set_ylabel("QpJ ×10⁻⁷", fontsize=9)
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.spines[["top","right"]].set_visible(False)

    # (1,0-1) FID vs Energy scatter
    ax4 = fig.add_subplot(gs[1, 0:2])
    for name, r in results.items():
        ax4.scatter(
            r["total_energy_j"]/1000, r["fid"],
            color=COLORS[name], s=200, zorder=5,
            edgecolors="white", linewidths=1.5, label=LABELS[name]
        )
        ax4.annotate(LABELS[name],
                     xy=(r["total_energy_j"]/1000, r["fid"]),
                     xytext=(8, 5), textcoords="offset points",
                     fontsize=9, color=COLORS[name], fontweight="bold")
    ax4.set_xlabel("Total Training Energy (kJ)", fontsize=10)
    ax4.set_ylabel("FID Score", fontsize=10)
    ax4.set_title("FID vs. Energy Trade-off", fontsize=10, fontweight="bold")
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.spines[["top","right"]].set_visible(False)

    # (1,2) Training time
    ax5 = fig.add_subplot(gs[1, 2])
    times = [results[n]["total_time_s"]/60 for n in names]
    bars  = ax5.bar(labels, times, color=colors, width=0.5, edgecolor="white")
    for b, v in zip(bars, times):
        ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                 f"{v:.1f}m", ha="center", fontsize=9, fontweight="bold")
    ax5.set_title("Total Training Time\n(lower = better)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Time (minutes)", fontsize=9)
    ax5.grid(axis="y", alpha=0.3, linestyle="--")
    ax5.spines[["top","right"]].set_visible(False)

    fig.suptitle("Quality per Joule — Full Evaluation Dashboard\nECE 7650 | Winter 2026",
                 fontsize=14, fontweight="bold", y=1.01)

    path = os.path.join(PLOT_DIR, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {path}")


# ==============================================================
# Print Results Table
# ==============================================================
def print_results_table(results):
    print("\n" + "=" * 75)
    print(f"  {'Model':<15} {'FID':>8} {'Energy(kJ)':>12} "
          f"{'Time(min)':>10} {'Params':>10} {'QpJ×10⁻⁷':>12}")
    print("=" * 75)
    for name, r in results.items():
        print(
            f"  {LABELS[name]:<15} "
            f"{r['fid']:>8.2f} "
            f"{r['total_energy_j']/1000:>12.1f} "
            f"{r['total_time_s']/60:>10.1f} "
            f"{r['total_params']:>10,} "
            f"{r['quality_per_joule']*1e7:>12.4f}"
        )
    print("=" * 75)

    best_fid = min(results, key=lambda k: results[k]["fid"])
    best_eff = min(results, key=lambda k: results[k]["total_energy_j"])
    best_qpj = max(results, key=lambda k: results[k]["quality_per_joule"])

    print(f"\n  Best FID (quality)          : {LABELS[best_fid]}")
    print(f"  Most energy-efficient       : {LABELS[best_eff]}")
    print(f"  Best Quality per Joule      : {LABELS[best_qpj]}")
    print()


# Main execution method
if __name__ == "__main__":
    print("=" * 50)
    print("  Quality per Joule — Analysis")
    print("=" * 50)

    results = load_results()
    print_results_table(results)

    print("\n  Generating plots...")
    plot_fid_vs_energy(results)
    plot_quality_per_joule(results)
    plot_loss_curves()
    plot_energy_over_training()
    plot_summary_dashboard(results)

    print(f"\n  All plots saved to: {PLOT_DIR}/")
    print("  Analysis complete.")