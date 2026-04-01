# ============================================================
# Quality per Joule — Master Runner
# run_all.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================
# Runs the full pipeline in sequence:
#   1. Train AE
#   2. Train VAE
#   3. Train DCGAN
#   4. Train WGAN-GP
#   5. Train DDPM
#   6. Evaluate all models (FID + Quality per Joule)
#   7. Generate analysis plots
#
# Usage:
#   python run_all.py                  # run everything
#   python run_all.py --skip-training  # evaluate + analyze only
#   python run_all.py --only ddpm      # train one model only
# ============================================================

import argparse
import time
import sys
import traceback

# Argument parsing
parser = argparse.ArgumentParser(description="Quality per Joule — Full Pipeline")
parser.add_argument("--skip-training", action="store_true", help="Skip all training, run evaluation + analysis only.")
parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation, run training + analysis only.")
parser.add_argument("--skip-analysis", action="store_true",help="Skip analysis plots.")
parser.add_argument("--only", type=str, default=None, choices=["ae", "vae", "dcgan", "wgan_gp", "ddpm"], help="Train a single model only.")
args = parser.parse_args()

# Helper function to print section headers
def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# seconds to idle between stages to let GPU/system settle
STAGE_SLEEP_S = 10

# Helper function to run a stage with timing and error handling
def run_stage(name, fn):
    section(name)
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n  ✓  {name} completed in {elapsed/60:.1f} min")
        print(f"  Sleeping {STAGE_SLEEP_S}s to let GPU settle...")
        time.sleep(STAGE_SLEEP_S)
        return True
    except Exception as e:
        print(f"\n  ✗  {name} FAILED: {e}")
        traceback.print_exc()
        print(f"  Sleeping {STAGE_SLEEP_S}s before next stage...")
        time.sleep(STAGE_SLEEP_S)
        return False


# Stage functions — each imports and runs the corresponding training/evaluation code
def stage_train_ae():
    from train.train_ae import train, CONFIG
    train(CONFIG)

def stage_train_vae():
    from train.train_vae import train, CONFIG
    train(CONFIG)

def stage_train_dcgan():
    from train.train_dcgan import train, CONFIG
    train(CONFIG)

def stage_train_wgan_gp():
    from train.train_wgan_gp import train, CONFIG
    train(CONFIG)

def stage_train_ddpm():
    from train.train_ddpm import train, CONFIG
    train(CONFIG)

def stage_evaluate():
    from evaluate.evaluate import evaluate
    evaluate()

# Generates all analysis plots and tables from the evaluation results
def stage_analysis():
    from evaluate.analysis import (
        load_results, print_results_table,
        plot_fid_vs_energy, plot_quality_per_joule,
        plot_loss_curves, plot_energy_over_training,
        plot_summary_dashboard,
    )
    results = load_results()
    print_results_table(results)
    plot_fid_vs_energy(results)
    plot_quality_per_joule(results)
    plot_loss_curves()
    plot_energy_over_training()
    plot_summary_dashboard(results)


# List of all training stages in order, with display names and functions
ALL_TRAIN_STAGES = [
    ("Training — Autoencoder",  stage_train_ae),
    ("Training — VAE",          stage_train_vae),
    ("Training — DCGAN",        stage_train_dcgan),
    ("Training — WGAN-GP",      stage_train_wgan_gp),
    ("Training — DDPM",         stage_train_ddpm),
]

ONLY_MAP = {
    "ae"      : ("Training — Autoencoder", stage_train_ae),
    "vae"     : ("Training — VAE",         stage_train_vae),
    "dcgan"   : ("Training — DCGAN",       stage_train_dcgan),
    "wgan_gp" : ("Training — WGAN-GP",     stage_train_wgan_gp),
    "ddpm"    : ("Training — DDPM",        stage_train_ddpm),
}


# Main function to run the full pipeline
def main():
    pipeline_start = time.time()
    results = {}

    section("Quality per Joule — Full Pipeline")
    print(f"  skip-training : {args.skip_training}")
    print(f"  skip-eval     : {args.skip_eval}")
    print(f"  skip-analysis : {args.skip_analysis}")
    print(f"  only          : {args.only or 'all'}")

    # Training
    if not args.skip_training:
        if args.only:
            name, fn = ONLY_MAP[args.only]
            results[name] = run_stage(name, fn)
        else:
            for name, fn in ALL_TRAIN_STAGES:
                results[name] = run_stage(name, fn)

    # Evaluation
    if not args.skip_eval and args.only is None:
        results["Evaluation"] = run_stage("Evaluation — FID + Quality per Joule",
                                          stage_evaluate)

    # Analysis
    if not args.skip_analysis and args.only is None:
        results["Analysis"] = run_stage("Analysis — Plots & Dashboard",
                                        stage_analysis)

    # Display summary
    total_time = time.time() - pipeline_start
    section("Pipeline Summary")
    for stage, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {stage}")
    print(f"\n  Total wall time : {total_time/60:.1f} min  "
          f"({total_time/3600:.2f} hrs)")
    print("=" * 60)

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} stage(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print("\n  All stages completed successfully.")

if __name__ == "__main__":
    main()