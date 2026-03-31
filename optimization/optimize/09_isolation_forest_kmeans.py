"""
09_isolation_forest_kmeans.py

Two-stage clustering: IsolationForest anomaly detection followed by
MiniBatchKMeans on inliers.  Outliers are assigned to a separate cluster
before small-cluster merging.

Hyperparameters optimised via Optuna:
  n_clusters    : int  [5, 80]
  contamination : float uniform [0.01, 0.15]
  n_estimators  : categorical [50, 100, 200]
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

LOG_PATH  = RESULTS_DIR / "09_isolation_forest_kmeans_log.jsonl"
BEST_PATH = RESULTS_DIR / "09_isolation_forest_kmeans_best.json"


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=None):
    min_count = cfg["min_count"]
    cost_mode                          = cfg.get("cost_mode", "combined")
    lambda_penalty                     = cfg.get("lambda_penalty", 0.3)
    max_cluster_4sigma_threshold_ratio = cfg.get("max_cluster_4sigma_threshold_ratio", 0.8)
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import MiniBatchKMeans

        n_clusters    = trial.suggest_int("n_clusters", 5, 80)
        contamination = trial.suggest_float("contamination", 0.01, 0.15)
        n_estimators  = trial.suggest_categorical("n_estimators", [50, 100, 200])

        t0 = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1: Isolation Forest anomaly detection
        # ------------------------------------------------------------------
        iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )
        preds        = iso.fit_predict(X_sel)   # 1=inlier, -1=outlier
        inlier_mask  = preds == 1

        X_inliers    = X_sel[inlier_mask]
        n_outliers   = int((~inlier_mask).sum())

        # ------------------------------------------------------------------
        # Stage 2: MiniBatchKMeans on inliers
        # ------------------------------------------------------------------
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=10000,
            n_init=5,
            random_state=42,
        )
        inlier_labels = km.fit_predict(X_inliers)

        # Build full label array: inliers -> cluster id, outliers -> max+1
        full_labels = np.full(len(X_sel), -1, dtype=int)
        full_labels[inlier_mask] = inlier_labels
        max_lbl = int(full_labels[inlier_mask].max())
        full_labels[~inlier_mask] = max_lbl + 1

        merged = merge_small_clusters(full_labels, X_sel, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(
            labels, y, ref_median, min_count, lower_pct, upper_pct,
            cost_mode=cost_mode,
            lambda_penalty=lambda_penalty,
            baseline_4sigma=baseline_4sigma,
            max_cluster_4sigma_threshold_ratio=max_cluster_4sigma_threshold_ratio,
        )

        duration = time.perf_counter() - t0
        n_clusters_actual = int(labels.max()) + 1

        # Print progress
        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"k={n_clusters:2d} contam={contamination:.2f} n_est={n_estimators:3d} | "
            f"cost={cost_str} | "
            f"clusters={n_clusters_actual:3d} outliers={n_outliers:5d} | "
            f"{duration:.1f}s"
        )

        # Log trial
        record = {
            "trial_number": trial.number + 1,
            "params": {
                "n_clusters":    n_clusters,
                "contamination": contamination,
                "n_estimators":  n_estimators,
            },
            "cost":       cost if cost != float("inf") else None,
            "n_clusters": n_clusters_actual,
            "n_outliers": n_outliers,
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
        description="Optimize IsolationForest + MiniBatchKMeans clustering with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=80,
        help="Number of Optuna trials (default: 80)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 3 trials (for testing)",
    )
    args = parser.parse_args()

    n_trials = 3 if args.dry_run else args.n_trials

    # Load data and config
    print(f"Loading preprocessed data from {PKL_PATH} ...")
    data        = load_preprocessed(str(PKL_PATH))
    X_sel       = data["X_sel"]
    y           = data["y"]
    ref_median  = data["overall_median_cd"]
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

    # Create study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    objective = make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=baseline_4sigma)
    study.optimize(objective, n_trials=n_trials)

    # ------------------------------------------------------------------
    # Best result: recompute to get final cluster count and outlier count
    # ------------------------------------------------------------------
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import MiniBatchKMeans

    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cost   = best_trial.value

    iso_best = IsolationForest(
        n_estimators=best_params["n_estimators"],
        contamination=best_params["contamination"],
        random_state=42,
        n_jobs=-1,
    )
    preds_best      = iso_best.fit_predict(X_sel)
    inlier_mask_best = preds_best == 1
    best_n_outliers  = int((~inlier_mask_best).sum())

    km_best = MiniBatchKMeans(
        n_clusters=best_params["n_clusters"],
        batch_size=10000,
        n_init=5,
        random_state=42,
    )
    inlier_labels_best = km_best.fit_predict(X_sel[inlier_mask_best])

    full_labels_best = np.full(len(X_sel), -1, dtype=int)
    full_labels_best[inlier_mask_best] = inlier_labels_best
    max_lbl_best = int(full_labels_best[inlier_mask_best].max())
    full_labels_best[~inlier_mask_best] = max_lbl_best + 1

    merged_best     = merge_small_clusters(full_labels_best, X_sel, cfg["min_count"])
    best_labels     = relabel_sequential(merged_best)
    best_n_clusters = int(best_labels.max()) + 1

    # Improvement over baseline
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method":         "isolation_forest_kmeans",
        "best_cost":      best_cost if best_cost != float("inf") else None,
        "best_params":    best_params,
        "n_clusters":     best_n_clusters,
        "n_outliers":     best_n_outliers,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": improvement_pct,
    }
    save_best_result(str(BEST_PATH), result)

    # Print summary
    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print("  Best cost          : inf")
    print(
        f"  Best params        : n_clusters={best_params['n_clusters']}, "
        f"contamination={best_params['contamination']:.4f}, "
        f"n_estimators={best_params['n_estimators']}"
    )
    print(f"  Best n_clusters    : {best_n_clusters}")
    print(f"  Best n_outliers    : {best_n_outliers}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")


if __name__ == "__main__":
    main()
