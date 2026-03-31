"""
03_autoencoder_kmeans.py

Optimize clustering via Autoencoder + MiniBatchKMeans using Optuna.
Hyperparameters: latent_dim, n_clusters, epochs, lr, dropout
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

from optimization.common.cost import cost_function, compute_cluster_stats
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

LOG_PATH = RESULTS_DIR / "03_autoencoder_kmeans_log.jsonl"
BEST_PATH = RESULTS_DIR / "03_autoencoder_kmeans_best.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
class LayoutAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.LeakyReLU(0.1),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.LeakyReLU(0.1),
            nn.Linear(64, 128),         nn.LeakyReLU(0.1),
            nn.Linear(128, 256),        nn.LeakyReLU(0.1),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_autoencoder(
    X_tensor: torch.Tensor,
    input_dim: int,
    latent_dim: int,
    epochs: int,
    lr: float,
    dropout: float,
) -> np.ndarray:
    """Train autoencoder and return latent embeddings for all samples."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    model = LayoutAutoencoder(input_dim, latent_dim, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        avg_loss = epoch_loss / len(X_tensor)
        scheduler.step(avg_loss)

    # Extract latent embeddings
    model.eval()
    with torch.no_grad():
        z = model.encode(X_tensor.to(DEVICE))
    return z.cpu().numpy()


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=None):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]
    cost_mode                          = cfg.get("cost_mode", "combined")
    lambda_penalty                     = cfg.get("lambda_penalty", 0.3)
    max_cluster_4sigma_threshold_ratio = cfg.get("max_cluster_4sigma_threshold_ratio", 0.8)

    input_dim = X_sel.shape[1]
    X_tensor = torch.tensor(X_sel, dtype=torch.float32)

    def objective(trial):
        latent_dim = trial.suggest_categorical("latent_dim", [8, 16, 32])
        n_clusters = trial.suggest_int("n_clusters", 5, 60)
        epochs = trial.suggest_categorical("epochs", [30, 50, 100])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        t0 = time.perf_counter()

        # Train autoencoder and get latent representation
        z = train_autoencoder(X_tensor, input_dim, latent_dim, epochs, lr, dropout)

        # Cluster in latent space
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=BATCH_SIZE, n_init=3
        )
        raw_labels = kmeans.fit_predict(z)

        merged = merge_small_clusters(raw_labels, z, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(
            labels, y, ref_median, min_count, lower_pct, upper_pct,
            cost_mode=cost_mode,
            lambda_penalty=lambda_penalty,
            baseline_4sigma=baseline_4sigma,
            max_cluster_4sigma_threshold_ratio=max_cluster_4sigma_threshold_ratio,
        )

        duration = time.perf_counter() - t0
        n_actual = int(labels.max()) + 1

        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"latent={latent_dim} k={n_clusters} epochs={epochs} "
            f"lr={lr:.4f} drop={dropout:.2f} | "
            f"cost={cost_str} | clusters={n_actual} | {duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "latent_dim": latent_dim,
                "n_clusters": n_clusters,
                "epochs": epochs,
                "lr": lr,
                "dropout": dropout,
            },
            "cost": cost if cost != float("inf") else None,
            "n_clusters": n_actual,
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
        description="Optimize Autoencoder + KMeans clustering with Optuna"
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
    data = load_preprocessed(str(PKL_PATH))
    X_sel = data["X_sel"]
    y = data["y"]
    ref_median = data["overall_median_cd"]
    baseline_4sigma = data.get("baseline_4sigma", None)

    print(f"Loading cost config from {CFG_PATH} ...")
    cfg = load_cost_config(str(CFG_PATH))

    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"Fixed: min_count={cfg['min_count']}")
    print(f"Cost mode: {cfg.get('cost_mode', 'combined')}")
    print(f"Data shape: X={X_sel.shape}, y={y.shape}")
    print("-" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    n_jobs = cfg.get("optuna_n_jobs", 1)
    objective = make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=baseline_4sigma)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Best result — retrain to get final labels
    best_trial = study.best_trial
    best_params = best_trial.params
    best_cost = best_trial.value

    input_dim = X_sel.shape[1]
    X_tensor = torch.tensor(X_sel, dtype=torch.float32)

    z_best = train_autoencoder(
        X_tensor,
        input_dim,
        best_params["latent_dim"],
        best_params["epochs"],
        best_params["lr"],
        best_params["dropout"],
    )
    kmeans_best = MiniBatchKMeans(
        n_clusters=best_params["n_clusters"],
        random_state=42,
        batch_size=BATCH_SIZE,
        n_init=3,
    )
    raw_labels = kmeans_best.fit_predict(z_best)
    merged = merge_small_clusters(raw_labels, z_best, cfg["min_count"])
    best_labels = relabel_sequential(merged)
    best_n_clusters = int(best_labels.max()) + 1

    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]
    stats = compute_cluster_stats(best_labels, y, ref_median, lower_pct, upper_pct)

    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method": "autoencoder_kmeans",
        "best_cost": best_cost if best_cost != float("inf") else None,
        "best_params": best_params,
        "n_clusters": best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": improvement_pct,
        "cluster_stats": {
            "combined_4sigma_pct": stats["combined_4sigma_pct"],
            "weighted_mean_4spct": stats["weighted_mean_4spct"],
            "max_4spct": stats["max_4spct"],
            "median_per_cluster": {str(k): v for k, v in stats["median_per_cluster"].items()},
            "cluster_counts": {str(k): v for k, v in stats["cluster_counts"].items()},
        },
    }
    save_best_result(str(BEST_PATH), result)

    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print(f"  Best cost          : inf")
    print(f"  Best params        : latent_dim={best_params['latent_dim']}, "
          f"n_clusters={best_params['n_clusters']}, epochs={best_params['epochs']}, "
          f"lr={best_params['lr']:.6f}, dropout={best_params['dropout']:.4f}")
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")


if __name__ == "__main__":
    main()
