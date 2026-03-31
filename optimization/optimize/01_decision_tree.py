"""
01_decision_tree.py

Optimize clustering via DecisionTreeRegressor using Optuna.
Hyperparameters: max_leaf_nodes, min_samples_leaf
"""

import argparse
import time
import sys
from pathlib import Path

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
PKL_PATH = ROOT / "optimization" / "preprocessed.pkl"
CFG_PATH = ROOT / "optimization" / "cost_function.json"

RESULTS_DIR = ROOT / "optimization" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH = RESULTS_DIR / "01_decision_tree_log.jsonl"
BEST_PATH = RESULTS_DIR / "01_decision_tree_best.json"


# ---------------------------------------------------------------------------
# Trial objective
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg):
    alpha = cfg["alpha"]
    beta = cfg["beta"]
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        from sklearn.tree import DecisionTreeRegressor

        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 5, 150)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 500)

        t0 = time.perf_counter()

        dt = DecisionTreeRegressor(
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        dt.fit(X_sel, y)
        raw_labels = dt.apply(X_sel)
        merged = merge_small_clusters(raw_labels, X_sel, min_count)
        labels = relabel_sequential(merged)

        cost = cost_function(
            labels, y, ref_median, alpha, beta, min_count, lower_pct, upper_pct
        )

        duration = time.perf_counter() - t0
        n_clusters = int(labels.max()) + 1

        # Print progress
        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"max_leaf={max_leaf_nodes:3d} min_leaf={min_samples_leaf:3d} | "
            f"cost={cost_str} | k={n_clusters:3d} | {duration:.1f}s"
        )

        # Log trial
        record = {
            "trial_number": trial.number + 1,
            "params": {
                "max_leaf_nodes": max_leaf_nodes,
                "min_samples_leaf": min_samples_leaf,
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
        description="Optimize DecisionTree clustering with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials (default: 100)"
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

    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(f"Fixed: alpha={cfg['alpha']}, beta={cfg['beta']}, min_count={cfg['min_count']}")
    print(f"Data shape: X={X_sel.shape}, y={y.shape}")
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
    from sklearn.tree import DecisionTreeRegressor

    dt_best = DecisionTreeRegressor(
        max_leaf_nodes=best_params["max_leaf_nodes"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
    )
    dt_best.fit(X_sel, y)
    raw_labels = dt_best.apply(X_sel)
    merged = merge_small_clusters(raw_labels, X_sel, cfg["min_count"])
    best_labels = relabel_sequential(merged)
    best_n_clusters = int(best_labels.max()) + 1

    # Improvement over baseline
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost = (cfg["alpha"] + cfg["beta"]) * baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    result = {
        "method": "decision_tree",
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
    print(f"  Best cost          : {best_cost:.4f}" if best_cost != float("inf") else "  Best cost          : inf")
    print(f"  Best params        : max_leaf_nodes={best_params['max_leaf_nodes']}, min_samples_leaf={best_params['min_samples_leaf']}")
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")


if __name__ == "__main__":
    main()
