"""
12_xgb_leaf_kmeans.py

XGBoost Leaf Embedding + K-Means clustering.

The existing XGBoost CD model assigns each sample to a leaf per tree.
Samples that land in the same leaves frequently share similar CD behavior.
We encode this as a normalized leaf matrix, reduce via TruncatedSVD, then
cluster in the embedding space with MiniBatchKMeans.

Hyperparameters optimised via Optuna:
  n_clusters     : int  [5, 100]
  svd_components : categorical [20, 40, 60, 80, 100]
  n_init         : int  [3, 15]
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import optuna

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
DATA_PKL  = ROOT / "optimization" / "preprocessed.pkl"
COST_CFG  = ROOT / "optimization" / "cost_function.json"
XGB_MODEL = ROOT / "optimization" / "xgb_cd_model.pkl"

RESULTS_DIR = ROOT / "optimization" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH  = RESULTS_DIR / "12_xgb_leaf_kmeans_log.jsonl"
BEST_PATH = RESULTS_DIR / "12_xgb_leaf_kmeans_best.json"


# ---------------------------------------------------------------------------
# Pre-compute leaf embedding (done once, outside Optuna loop)
# ---------------------------------------------------------------------------
def compute_leaf_norm(xgb_model, X_scaled: np.ndarray) -> np.ndarray:
    """Return normalised leaf index matrix, shape (n_samples, n_estimators)."""
    # apply() returns leaf indices: shape (n_samples, n_estimators)
    leaf_indices = xgb_model.apply(X_scaled)          # (n, 200)
    leaf_float   = leaf_indices.astype(np.float32)

    # Normalise each tree's leaf IDs to [0, 1]
    leaf_min   = leaf_float.min(axis=0, keepdims=True)
    leaf_max   = leaf_float.max(axis=0, keepdims=True)
    leaf_range = np.where(leaf_max - leaf_min > 0, leaf_max - leaf_min, 1.0)
    leaf_norm  = (leaf_float - leaf_min) / leaf_range  # (n, n_trees)
    return leaf_norm


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(leaf_norm: np.ndarray, y: np.ndarray, ref_median: float, cfg: dict):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize
        from sklearn.cluster import MiniBatchKMeans

        n_clusters     = trial.suggest_int("n_clusters", 5, 100)
        svd_components = trial.suggest_categorical("svd_components", [20, 40, 60, 80, 100])
        n_init         = trial.suggest_int("n_init", 3, 15)

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # TruncatedSVD dimensionality reduction
        # ------------------------------------------------------------------
        svd   = TruncatedSVD(n_components=svd_components, random_state=42)
        X_emb = svd.fit_transform(leaf_norm)   # (n_samples, svd_components)
        X_emb = normalize(X_emb)               # L2-normalise rows

        # ------------------------------------------------------------------
        # MiniBatchKMeans on embedding
        # ------------------------------------------------------------------
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=10000,
            n_init=n_init,
            random_state=42,
        )
        raw_labels = km.fit_predict(X_emb)

        merged = merge_small_clusters(raw_labels, X_emb, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(labels, y, ref_median, min_count, lower_pct, upper_pct)

        duration          = time.perf_counter() - t0
        n_clusters_actual = int(labels.max()) + 1

        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"k={n_clusters:3d} svd={svd_components:3d} n_init={n_init:2d} | "
            f"cost={cost_str} | "
            f"clusters={n_clusters_actual:3d} | "
            f"{duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "n_clusters":     n_clusters,
                "svd_components": svd_components,
                "n_init":         n_init,
            },
            "cost":       cost if cost != float("inf") else None,
            "n_clusters": n_clusters_actual,
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
        description="Optimize XGBoost Leaf Embedding + MiniBatchKMeans with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=80,
        help="Number of Optuna trials (default: 80)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 3 trials (for testing)",
    )
    args    = parser.parse_args()
    n_trials = 3 if args.dry_run else args.n_trials

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading preprocessed data from {DATA_PKL} ...")
    data            = load_preprocessed(str(DATA_PKL))
    X_scaled        = data["X_scaled"]          # (n, 126) — model was trained on full 126D
    y               = data["y"]
    ref_median      = data["overall_median_cd"]
    baseline_4sigma = data.get("baseline_4sigma", None)

    print(f"Loading cost config from {COST_CFG} ...")
    cfg = load_cost_config(str(COST_CFG))

    # ------------------------------------------------------------------
    # Load XGBoost model and pre-compute leaf embedding (one-time)
    # ------------------------------------------------------------------
    import joblib
    print(f"Loading XGBoost model from {XGB_MODEL} ...")
    xgb_model = joblib.load(str(XGB_MODEL))

    print("Computing leaf embedding (one-time) ...")
    t_emb_start = time.perf_counter()
    leaf_norm   = compute_leaf_norm(xgb_model, X_scaled)
    t_emb_end   = time.perf_counter()
    print(
        f"Leaf embedding computed: shape={leaf_norm.shape}, "
        f"time={t_emb_end - t_emb_start:.1f}s"
    )

    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"Fixed: min_count={cfg['min_count']}")
    print(f"Data shape: X_scaled={X_scaled.shape}, y={y.shape}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study     = optuna.create_study(direction="minimize")
    objective = make_objective(leaf_norm, y, ref_median, cfg)
    study.optimize(objective, n_trials=n_trials)

    # ------------------------------------------------------------------
    # Recompute best result for final summary
    # ------------------------------------------------------------------
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    from sklearn.cluster import MiniBatchKMeans

    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cost   = best_trial.value

    svd_best   = TruncatedSVD(n_components=best_params["svd_components"], random_state=42)
    X_emb_best = svd_best.fit_transform(leaf_norm)
    X_emb_best = normalize(X_emb_best)

    km_best = MiniBatchKMeans(
        n_clusters=best_params["n_clusters"],
        batch_size=10000,
        n_init=best_params["n_init"],
        random_state=42,
    )
    raw_labels_best  = km_best.fit_predict(X_emb_best)
    merged_best      = merge_small_clusters(raw_labels_best, X_emb_best, cfg["min_count"])
    best_labels      = relabel_sequential(merged_best)
    best_n_clusters  = int(best_labels.max()) + 1

    # Improvement over baseline
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost   = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method":          "xgb_leaf_kmeans",
        "best_cost":       best_cost if best_cost != float("inf") else None,
        "best_params":     best_params,
        "n_clusters":      best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": improvement_pct,
    }
    save_best_result(str(BEST_PATH), result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print("  Best cost          : inf")
    print(
        f"  Best params        : n_clusters={best_params['n_clusters']}, "
        f"svd_components={best_params['svd_components']}, "
        f"n_init={best_params['n_init']}"
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
