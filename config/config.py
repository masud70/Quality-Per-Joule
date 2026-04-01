# ============================================================
# Quality per Joule — Global Configuration
# config/config.py
# ECE 7650 Generative AI | Winter 2026
# ============================================================

# ----------------
# Shared Constants
# ----------------
SEED        = 42
IMAGE_SIZE  = 32
BATCH_SIZE  = 64
NUM_WORKERS = 8
DATA_DIR    = "./data/celeba"
LATENT_DIM  = 128
EPOCHS         = 30
LEARNING_RATE  = 2e-4

# ----------------
# Directory Paths
# ----------------
CHECKPOINT_DIR = "./results/checkpoints"
LOG_DIR        = "./results/logs"
SAMPLE_DIR     = "./results/samples"
PLOT_DIR       = "./results/plots"
RESULTS_PATH   = "./results/evaluation_results.json"

# ----------------
# Data Pipeline
# ----------------
DATA_CONFIG = {
    "seed"          : SEED,
    "data_dir"      : DATA_DIR,
    "batch_size"    : BATCH_SIZE,
    "num_workers"   : NUM_WORKERS,
    "image_size"    : IMAGE_SIZE,
    "n_samples"     : 10000,
}

# ----------------
# Autoencoder
# ----------------
AE_CONFIG = {
    "latent_dim"    : LATENT_DIM,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "learning_rate" : LEARNING_RATE,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "sample_dir"    : SAMPLE_DIR,
    "model_name"    : "autoencoder",
}

# ----------------
# VAE
# ----------------
VAE_CONFIG = {
    "latent_dim"    : LATENT_DIM,
    "beta"          : 1.0,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "learning_rate" : LEARNING_RATE,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "sample_dir"    : SAMPLE_DIR,
    "model_name"    : "vae",
}

# ----------------
# DCGAN
# ----------------
DCGAN_CONFIG = {
    "latent_dim"    : LATENT_DIM,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "lr_g"          : LEARNING_RATE,
    "lr_d"          : LEARNING_RATE,
    "beta1"         : 0.5,
    "beta2"         : 0.999,
    "real_label"    : 0.9,
    "fake_label"    : 0.0,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "sample_dir"    : SAMPLE_DIR,
    "model_name"    : "dcgan",
}
# ----------------
# WGAN-GP
# ----------------
WGAN_GP_CONFIG = {
    "latent_dim"    : LATENT_DIM,
    "lambda_gp"     : 10.0,
    "n_critic"      : 3,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "lr_g"          : LEARNING_RATE,
    "lr_c"          : LEARNING_RATE,
    "beta1"         : 0.0,
    "beta2"         : 0.9,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "sample_dir"    : SAMPLE_DIR,
    "model_name"    : "wgan_gp",
}

# ----------------
# DDPM
# ----------------
DDPM_CONFIG = {
    "base_ch"       : 32,
    "time_dim"      : 128,
    "T"             : 1000,
    "beta_start"    : 1e-4,
    "beta_end"      : 0.02,
    "epochs"        : EPOCHS,
    "batch_size"    : BATCH_SIZE,
    "learning_rate" : LEARNING_RATE,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "sample_dir"    : SAMPLE_DIR,
    "model_name"    : "ddpm",
}

# ----------------
# Evaluation
# ----------------
EVAL_CONFIG = {
    "latent_dim"    : LATENT_DIM,
    "vae_latent_dim": LATENT_DIM,
    "T"             : DDPM_CONFIG["T"],
    "base_ch"       : DDPM_CONFIG["base_ch"],
    "time_dim"      : DDPM_CONFIG["time_dim"],
    "n_fid_samples" : 10000,
    "batch_size"    : 32,
    "seed"          : SEED,
    "checkpoint_dir": CHECKPOINT_DIR,
    "log_dir"       : LOG_DIR,
    "output_path"   : RESULTS_PATH,
}
