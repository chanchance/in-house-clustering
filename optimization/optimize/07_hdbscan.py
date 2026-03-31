"""07_hdbscan.py — HDBSCAN hyperparameter optimization via Optuna.

Usage:
    python optimization/optimize/07_hdbscan.py [--n-trials N] [--dry-run]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import optuna
except ImportError:
    from optimization.common import optuna_compat as optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from optimization.common.cost import cost_function, compute_cluster_stats
from optimization.common.utils import (
    append_trial_log,
    load_cost_config,
    load_preprocessed,
    merge_small_clusters,
    relabel_sequential,
    save_best_result,
)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
PKL_PATH  = ROOT / "optimization" / "preprocessed.pkl"
CFG_PATH  = ROOT / "optimization" / "cost_function.json"
LOG_PATH  = ROOT / "optimization" / "results" / "07_hdbscan_log.jsonl"
BEST_PATH = ROOT / "optimization" / "results" / "07_hdbscan_best.json"


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=None):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]
    cost_mode                          = cfg.get("cost_mode", "combined")
    lambda_penalty                     = cfg.get("lambda_penalty", 0.3)
    max_cluster_4sigma_threshold_ratio = cfg.get("max_cluster_4sigma_threshold_ratio", 0.8)

    def objective(trial):
        import numpy as np
        from sklearn.cluster import HDBSCAN

        min_cluster_size          = trial.suggest_int("min_cluster_size", 20, 500)
        min_samples               = trial.suggest_int("min_samples", 1, 50)
        cluster_selection_epsilon = trial.suggest_float("cluster_selection_epsilon", 0.0, 1.0)
        cluster_selection_method  = trial.suggest_categorical("cluster_selection_method", ["eom", "leaf"])

        t0 = time.perf_counter()

        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        raw_labels = hdb.fit_predict(X_sel)

        # Assign noise points (-1) to nearest cluster centroid
        noise_mask = raw_labels == -1
        if noise_mask.any():
            valid_lbls = np.unique(raw_labels[~noise_mask])
            if len(valid_lbls) == 0:
                # All noise — return inf
                return float("inf")
            centroids  = np.array([X_sel[raw_labels == l].mean(axis=0) for l in valid_lbls])
            noise_X    = X_sel[noise_mask]
            dists      = np.linalg.norm(noise_X[:, None, :] - centroids[None, :, :], axis=2)
            raw_labels[noise_mask] = valid_lbls[dists.argmin(axis=1)]

        merged = merge_small_clusters(raw_labels, X_sel, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(
            labels, y, ref_median, min_count, lower_pct, upper_pct,
            cost_mode=cost_mode,
            lambda_penalty=lambda_penalty,
            baseline_4sigma=baseline_4sigma,
            max_cluster_4sigma_threshold_ratio=max_cluster_4sigma_threshold_ratio,
        )
        elapsed = time.perf_counter() - t0

        n_actual = int(labels.max()) + 1
        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"mcs={min_cluster_size} ms={min_samples} "
            f"eps={cluster_selection_epsilon:.2f} method={cluster_selection_method} | "
            f"cost={cost_str} | k={n_actual} | {elapsed:.1f}s"
        )

        append_trial_log(
            LOG_PATH,
            {
                "trial": trial.number + 1,
                "params": {
                    "min_cluster_size":          min_cluster_size,
                    "min_samples":               min_samples,
                    "cluster_selection_epsilon": cluster_selection_epsilon,
                    "cluster_selection_method":  cluster_selection_method,
                },
                "cost":              cost if cost != float("inf") else None,
                "n_actual_clusters": n_actual,
                "elapsed_sec":       round(elapsed, 3),
            },
        )

        return cost

    return objective


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HDBSCAN clustering optimization via Optuna")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--dry-run",  action="store_true",  help="Run 2 trials for testing")
    args = parser.parse_args()

    n_trials = 2 if args.dry_run else args.n_trials

    # 데이터 및 설정 로드
    print(f"Loading data from {PKL_PATH} ...")
    data = load_preprocessed(PKL_PATH)
    X_sel          = data["X_sel"]
    y              = data["y"]
    ref_median     = data["overall_median_cd"]
    baseline_4sigma = data["baseline_4sigma"]

    cfg = load_cost_config(CFG_PATH)
    n_jobs = cfg.get("optuna_n_jobs", 1)

    print(f"  X_sel shape : {X_sel.shape}")
    print(f"  ref_median  : {ref_median:.4f}")
    print(f"  baseline 4σ : {baseline_4sigma:.4f}")
    print(f"  n_trials    : {n_trials}")
    print(f"  n_jobs      : {n_jobs}")
    print(f"Cost mode: {cfg.get('cost_mode', 'combined')}")
    print()

    # 결과 디렉토리 보장
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=baseline_4sigma),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=False,
    )

    best       = study.best_trial
    best_cost  = best.value
    best_params = best.params

    # Refit with best params to get best_labels
    import numpy as np
    from sklearn.cluster import HDBSCAN

    hdb_best = HDBSCAN(
        min_cluster_size=best_params["min_cluster_size"],
        min_samples=best_params["min_samples"],
        cluster_selection_epsilon=best_params["cluster_selection_epsilon"],
        cluster_selection_method=best_params["cluster_selection_method"],
    )
    raw_labels = hdb_best.fit_predict(X_sel)

    noise_mask = raw_labels == -1
    if noise_mask.any():
        valid_lbls = np.unique(raw_labels[~noise_mask])
        centroids  = np.array([X_sel[raw_labels == l].mean(axis=0) for l in valid_lbls])
        noise_X    = X_sel[noise_mask]
        dists      = np.linalg.norm(noise_X[:, None, :] - centroids[None, :, :], axis=2)
        raw_labels[noise_mask] = valid_lbls[dists.argmin(axis=1)]

    merged      = merge_small_clusters(raw_labels, X_sel, cfg["min_count"])
    best_labels = relabel_sequential(merged)
    best_n_clusters = int(best_labels.max()) + 1

    improvement_pct = (
        (baseline_4sigma - best_cost) / baseline_4sigma * 100.0
        if baseline_4sigma > 0
        else 0.0
    )

    # Cluster stats
    stats = compute_cluster_stats(best_labels, y, ref_median, cfg["lower_pct"], cfg["upper_pct"])

    result = {
        "method": "hdbscan",
        "best_cost": best_cost,
        "best_params": best_params,
        "n_clusters": best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": round(improvement_pct, 4),
        "cluster_stats": {
            "combined_4sigma_pct":  stats["combined_4sigma_pct"],
            "weighted_mean_4spct":  stats["weighted_mean_4spct"],
            "max_4spct":            stats["max_4spct"],
            "median_per_cluster":   {str(k): v for k, v in stats["median_per_cluster"].items()},
            "cluster_counts":       {str(k): v for k, v in stats["cluster_counts"].items()},
        },
    }

    save_best_result(BEST_PATH, result)

    print()
    print("=" * 60)
    print(f"Best cost        : {best_cost:.4f}")
    print(f"Best params      : {best_params}")
    print(f"Best n_clusters  : {best_n_clusters}")
    print(f"Baseline 4σ      : {baseline_4sigma:.4f}")
    print(f"Improvement      : {improvement_pct:.2f}%")
    print(f"Log              : {LOG_PATH}")
    print(f"Best result      : {BEST_PATH}")


if __name__ == "__main__":
    main()
