"""
04_gmm.py

Optimize clustering via GaussianMixture using Optuna.
Hyperparameters: n_components, covariance_type, max_iter
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

LOG_PATH = RESULTS_DIR / "04_gmm_log.jsonl"
BEST_PATH = RESULTS_DIR / "04_gmm_best.json"


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    # Subsample index for fitting (scalability)
    subsample_n = min(50000, len(X_sel))
    idx = np.random.RandomState(42).choice(len(X_sel), subsample_n, replace=False)
    X_fit = X_sel[idx]

    def objective(trial):
        from sklearn.mixture import GaussianMixture

        n_components = trial.suggest_int("n_components", 5, 80)
        covariance_type = trial.suggest_categorical(
            "covariance_type", ["full", "tied", "diag", "spherical"]
        )
        max_iter = trial.suggest_categorical("max_iter", [100, 200, 300])

        t0 = time.perf_counter()

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42,
            n_init=3,
        )
        gmm.fit(X_fit)
        raw_labels = gmm.predict(X_sel)
        merged = merge_small_clusters(raw_labels, X_sel, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(labels, y, ref_median, min_count, lower_pct, upper_pct)

        duration = time.perf_counter() - t0
        n_clusters = int(labels.max()) + 1

        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"n={n_components:3d} cov={covariance_type} max_iter={max_iter} | "
            f"cost={cost_str} | clusters={n_clusters:3d} | {duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "n_components": n_components,
                "covariance_type": covariance_type,
                "max_iter": max_iter,
            },
            "cost": cost if cost != float("inf") else None,
            "n_clusters": n_clusters,
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
        description="Optimize GMM clustering with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=80, help="Number of Optuna trials (default: 80)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run only 3 trials (for testing)"
    )
    args = parser.parse_args()

    n_trials = 3 if args.dry_run else args.n_trials

    # Load data and config
    print(f"Loading preprocessed data from {PKL_PATH} ...")
    data = load_preprocessed(str(PKL_PATH))
    X_sel = data["X_sel"]
    y = data["y"]
    ref_median = data["overall_median_cd"]
    baseline_4sigma = data.get("baseline_4sigma", None)

    print(f"Loading cost config from {CFG_PATH} ...")
    cfg = load_cost_config(str(CFG_PATH))

    subsample_n = min(50000, len(X_sel))
    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"Fixed: min_count={cfg['min_count']}")
    print(f"Data shape: X={X_sel.shape}, y={y.shape}")
    print(f"Fit subsample: {subsample_n} / {len(X_sel)}")
    print("-" * 70)

    # Create study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    objective = make_objective(X_sel, y, ref_median, cfg)
    study.optimize(objective, n_trials=n_trials)

    # Best result
    best_trial = study.best_trial
    best_params = best_trial.params
    best_cost = best_trial.value

    # Recompute n_clusters for best params
    from sklearn.mixture import GaussianMixture

    subsample_n = min(50000, len(X_sel))
    idx = np.random.RandomState(42).choice(len(X_sel), subsample_n, replace=False)

    gmm_best = GaussianMixture(
        n_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        max_iter=best_params["max_iter"],
        random_state=42,
        n_init=3,
    )
    gmm_best.fit(X_sel[idx])
    raw_labels = gmm_best.predict(X_sel)
    merged = merge_small_clusters(raw_labels, X_sel, cfg["min_count"])
    best_labels = relabel_sequential(merged)
    best_n_clusters = int(best_labels.max()) + 1

    # Improvement over baseline
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method": "gmm",
        "best_cost": best_cost if best_cost != float("inf") else None,
        "best_params": best_params,
        "n_clusters": best_n_clusters,
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
    print(f"  Best params        : n_components={best_params['n_components']}, "
          f"covariance_type={best_params['covariance_type']}, "
          f"max_iter={best_params['max_iter']}")
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")


if __name__ == "__main__":
    main()
