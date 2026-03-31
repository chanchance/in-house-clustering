"""
11_dt_kmeans_twostage.py

Two-stage clustering: DecisionTreeRegressor produces coarse interpretable
leaf regions (Stage 1), then MiniBatchKMeans sub-clusters within each leaf
(Stage 2).  Produces human-readable decision rules alongside cluster labels.

Hyperparameters optimised via Optuna:
  dt_max_depth      : int  [2, 6]
  sub_k             : int  [2, 8]   (K-Means sub-clusters per DT leaf)
  min_samples_leaf  : int  [50, 300]
"""

from __future__ import annotations

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
DATA_PKL    = ROOT / "optimization" / "preprocessed.pkl"
COST_CFG    = ROOT / "optimization" / "cost_function.json"

RESULTS_DIR = ROOT / "optimization" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH    = RESULTS_DIR / "11_dt_kmeans_twostage_log.jsonl"
BEST_PATH   = RESULTS_DIR / "11_dt_kmeans_twostage_best.json"
RULES_PATH  = RESULTS_DIR / "11_dt_kmeans_twostage_rules.txt"


# ---------------------------------------------------------------------------
# Two-stage clustering helper
# ---------------------------------------------------------------------------
def run_twostage(X_sel, y, dt_max_depth, sub_k, min_samples_leaf, min_count):
    """Fit DT -> KMeans two-stage; return (labels, dt_model)."""
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cluster import MiniBatchKMeans

    # Stage 1: Decision Tree
    dt = DecisionTreeRegressor(
        max_depth=dt_max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    dt.fit(X_sel, y)
    leaf_ids = dt.apply(X_sel)          # shape (n,), leaf node index per sample

    unique_leaves = np.unique(leaf_ids)
    combined_labels = np.full(len(X_sel), -1, dtype=int)

    # Stage 2: K-Means within each leaf
    for i, leaf in enumerate(unique_leaves):
        mask   = leaf_ids == leaf
        X_leaf = X_sel[mask]
        n_leaf = int(mask.sum())

        # Clamp sub_k so no cluster will be smaller than min_count
        actual_sub_k = max(1, min(sub_k, n_leaf // min_count))

        if actual_sub_k == 1:
            combined_labels[mask] = i * sub_k          # single cluster for leaf
        else:
            km = MiniBatchKMeans(
                n_clusters=actual_sub_k,
                batch_size=max(1000, n_leaf),
                n_init=5,
                random_state=42,
            )
            sub_labels = km.fit_predict(X_leaf)
            combined_labels[mask] = i * sub_k + sub_labels

    merged = merge_small_clusters(combined_labels, X_sel, min_count)
    labels = relabel_sequential(merged)
    return labels, dt


# ---------------------------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=None):
    min_count = cfg["min_count"]
    cost_mode                          = cfg.get("cost_mode", "combined")
    lambda_penalty                     = cfg.get("lambda_penalty", 0.3)
    max_cluster_4sigma_threshold_ratio = cfg.get("max_cluster_4sigma_threshold_ratio", 0.8)
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        dt_max_depth     = trial.suggest_int("dt_max_depth",    2,   6)
        sub_k            = trial.suggest_int("sub_k",           2,   8)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 50, 300)

        t0 = time.perf_counter()

        labels, _ = run_twostage(
            X_sel, y, dt_max_depth, sub_k, min_samples_leaf, min_count
        )

        cost = cost_function(
            labels, y, ref_median, min_count, lower_pct, upper_pct,
            cost_mode=cost_mode,
            lambda_penalty=lambda_penalty,
            baseline_4sigma=baseline_4sigma,
            max_cluster_4sigma_threshold_ratio=max_cluster_4sigma_threshold_ratio,
        )

        duration = time.perf_counter() - t0
        n_clusters_actual = int(labels.max()) + 1

        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"depth={dt_max_depth} sub_k={sub_k} min_leaf={min_samples_leaf:3d} | "
            f"cost={cost_str} | "
            f"clusters={n_clusters_actual:3d} | "
            f"{duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "dt_max_depth":     dt_max_depth,
                "sub_k":            sub_k,
                "min_samples_leaf": min_samples_leaf,
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
        description="Optimize DecisionTree + MiniBatchKMeans two-stage clustering with Optuna"
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
    print(f"Loading preprocessed data from {DATA_PKL} ...")
    data            = load_preprocessed(str(DATA_PKL))
    X_sel           = data["X_sel"]
    y               = data["y"]
    ref_median      = data["overall_median_cd"]
    baseline_4sigma = data.get("baseline_4sigma", None)
    selected_features = data.get("selected_features", None)

    print(f"Loading cost config from {COST_CFG} ...")
    cfg = load_cost_config(str(COST_CFG))

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

    objective = make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=baseline_4sigma)
    study.optimize(objective, n_trials=n_trials)

    # ------------------------------------------------------------------
    # Best result: refit with best params to get final labels + DT rules
    # ------------------------------------------------------------------
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cost   = best_trial.value

    best_labels, dt_best = run_twostage(
        X_sel, y,
        best_params["dt_max_depth"],
        best_params["sub_k"],
        best_params["min_samples_leaf"],
        cfg["min_count"],
    )
    best_n_clusters = int(best_labels.max()) + 1

    # Improvement over baseline
    baseline_cost   = None
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost   = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method":          "dt_kmeans_twostage",
        "best_cost":       best_cost if best_cost != float("inf") else None,
        "best_params":     best_params,
        "n_clusters":      best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "baseline_cost":   baseline_cost,
        "improvement_pct": improvement_pct,
    }
    save_best_result(str(BEST_PATH), result)

    # ------------------------------------------------------------------
    # Export decision tree rules
    # ------------------------------------------------------------------
    try:
        from sklearn.tree import export_text
        feature_names = (
            list(selected_features) if selected_features is not None
            else [f"f{i}" for i in range(X_sel.shape[1])]
        )
        tree_rules = export_text(
            dt_best,
            feature_names=feature_names,
            max_depth=best_params["dt_max_depth"],
        )
        RULES_PATH.write_text(tree_rules, encoding="utf-8")
        print(f"\nDecision tree rules saved to: {RULES_PATH}")
        print("\n--- Decision Tree Rules (truncated to 30 lines) ---")
        for line in tree_rules.splitlines()[:30]:
            print(line)
        if len(tree_rules.splitlines()) > 30:
            print("  ... (see full rules in file)")
    except Exception as exc:
        print(f"\n[Warning] Could not export tree rules: {exc}")

    # Print summary
    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print("  Best cost          : inf")
    print(
        f"  Best params        : dt_max_depth={best_params['dt_max_depth']}, "
        f"sub_k={best_params['sub_k']}, "
        f"min_samples_leaf={best_params['min_samples_leaf']}"
    )
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")
    print(f"  Rules saved to     : {RULES_PATH}")


if __name__ == "__main__":
    main()
