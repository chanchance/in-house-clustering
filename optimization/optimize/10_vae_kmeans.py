"""
10_vae_kmeans.py

Optimize clustering via β-VAE + MiniBatchKMeans using Optuna.
Hyperparameters: latent_dim, n_clusters, epochs, lr, beta_kl
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
try:
    import optuna
except ImportError:
    from optimization.common import optuna_compat as optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from optimization.common.cost import cost_function
from optimization.common.utils import (
    load_preprocessed,
    load_cost_config,
    merge_small_clusters,
    relabel_sequential,
    append_trial_log,
    save_best_result,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKL_PATH = ROOT / "optimization" / "preprocessed.pkl"
CFG_PATH = ROOT / "optimization" / "cost_function.json"

RESULTS_DIR = ROOT / "optimization" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH  = RESULTS_DIR / "10_vae_kmeans_log.jsonl"
BEST_PATH = RESULTS_DIR / "10_vae_kmeans_best.json"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# β-VAE architecture
# ---------------------------------------------------------------------------
class LayoutVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64),        nn.LeakyReLU(0.1),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
        )

    def encode(self, x):
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z


# ---------------------------------------------------------------------------
# ELBO loss
# ---------------------------------------------------------------------------
def vae_loss(recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta_kl: float) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
    kl_loss    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta_kl * kl_loss


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_vae(
    X_tensor: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    beta_kl: float,
) -> np.ndarray:
    """Train β-VAE and return deterministic mu embeddings for all samples."""
    model = LayoutVAE(input_dim, latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    dataset = TensorDataset(X_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, mu, logvar, _ = model(batch)
            loss = vae_loss(recon, batch, mu, logvar, beta_kl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(loader))

    # Use mu (deterministic) for clustering — more stable than sampled z
    model.eval()
    with torch.no_grad():
        mu_all, _ = model.encode(X_tensor.to(DEVICE))
    return mu_all.cpu().numpy()


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    input_dim = X_sel.shape[1]
    X_tensor  = torch.tensor(X_sel, dtype=torch.float32)

    def objective(trial):
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])
        n_clusters = trial.suggest_int("n_clusters", 5, 60)
        epochs     = trial.suggest_categorical("epochs", [30, 50, 100])
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        beta_kl    = trial.suggest_float("beta_kl", 0.5, 4.0, log=True)

        t0 = time.perf_counter()

        # Train β-VAE and extract latent mu
        X_latent = train_vae(X_tensor, input_dim, latent_dim, epochs, lr, beta_kl)

        # Cluster in latent space
        km = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=10000, n_init=5, random_state=42
        )
        raw_labels = km.fit_predict(X_latent)

        merged = merge_small_clusters(raw_labels, X_latent, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(labels, y, ref_median, min_count, lower_pct, upper_pct)

        duration  = time.perf_counter() - t0
        n_actual  = int(labels.max()) + 1
        cost_str  = f"{cost:.4f}" if cost != float("inf") else "inf"

        print(
            f"Trial {trial.number + 1:03d} | "
            f"latent={latent_dim} k={n_clusters} epochs={epochs} "
            f"lr={lr:.4f} beta={beta_kl:.2f} | "
            f"cost={cost_str} | clusters={n_actual} | {duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "latent_dim": latent_dim,
                "n_clusters": n_clusters,
                "epochs":     epochs,
                "lr":         lr,
                "beta_kl":    beta_kl,
            },
            "cost":         cost if cost != float("inf") else None,
            "n_clusters":   n_actual,
            "duration_sec": round(duration, 4),
        }
        append_trial_log(str(LOG_PATH), record)

        return cost

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Optimize β-VAE + KMeans clustering with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Number of Optuna trials (default: 30)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run only 2 trials (for testing)"
    )
    args = parser.parse_args()

    n_trials = 2 if args.dry_run else args.n_trials

    print(f"Device: {DEVICE}")
    print(f"Loading preprocessed data from {PKL_PATH} ...")
    data     = load_preprocessed(str(PKL_PATH))
    X_sel    = data["X_sel"]
    y        = data["y"]
    ref_median       = data["overall_median_cd"]
    baseline_4sigma  = data.get("baseline_4sigma", None)

    print(f"Loading cost config from {CFG_PATH} ...")
    cfg = load_cost_config(str(CFG_PATH))

    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"Fixed: min_count={cfg['min_count']}")
    print(f"Data shape: X={X_sel.shape}, y={y.shape}")
    print("-" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    objective = make_objective(X_sel, y, ref_median, cfg)
    study.optimize(objective, n_trials=n_trials)

    # Best result — retrain to get final labels
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cost   = best_trial.value

    input_dim = X_sel.shape[1]
    X_tensor  = torch.tensor(X_sel, dtype=torch.float32)

    X_latent_best = train_vae(
        X_tensor,
        input_dim,
        best_params["latent_dim"],
        best_params["epochs"],
        best_params["lr"],
        best_params["beta_kl"],
    )
    km_best = MiniBatchKMeans(
        n_clusters=best_params["n_clusters"],
        batch_size=10000,
        n_init=5,
        random_state=42,
    )
    raw_labels   = km_best.fit_predict(X_latent_best)
    merged       = merge_small_clusters(raw_labels, X_latent_best, cfg["min_count"])
    best_labels  = relabel_sequential(merged)
    best_n_clusters = int(best_labels.max()) + 1

    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method":          "vae_kmeans",
        "best_cost":       best_cost if best_cost != float("inf") else None,
        "best_params":     best_params,
        "n_clusters":      best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": improvement_pct,
    }
    save_best_result(str(BEST_PATH), result)

    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print(f"  Best cost          : inf")
    print(
        f"  Best params        : latent_dim={best_params['latent_dim']}, "
        f"n_clusters={best_params['n_clusters']}, epochs={best_params['epochs']}, "
        f"lr={best_params['lr']:.6f}, beta_kl={best_params['beta_kl']:.4f}"
    )
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")


if __name__ == "__main__":
    main()
