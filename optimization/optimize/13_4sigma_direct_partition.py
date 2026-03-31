"""
13_4sigma_direct_partition.py

Optimize clustering via 4σ-Direct Recursive Partition using Optuna.
Hyperparameters: target_k, n_thresholds, min_split_improvement

Method: Greedy recursive binary tree where the splitting criterion IS the cost function.
At each step: find the cluster with highest 4σ range, then find the (feature, threshold)
split that maximally reduces the global combined 4σ range after alignment.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import optuna

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from optimization.common.cost import cost_function, compute_4sigma_range_pct, compute_combined_4sigma_after_alignment
from optimization.common.utils import (
    load_preprocessed,
    load_cost_config,
    relabel_sequential,
    append_trial_log,
    save_best_result,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PKL = ROOT / "optimization" / "preprocessed.pkl"
COST_CFG = ROOT / "optimization" / "cost_function.json"

RESULTS_DIR = ROOT / "optimization" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

LOG_PATH  = RESULTS_DIR / "13_4sigma_direct_partition_log.jsonl"
BEST_PATH = RESULTS_DIR / "13_4sigma_direct_partition_best.json"
TREE_PATH = RESULTS_DIR / "13_4sigma_direct_partition_tree.json"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
class FourSigmaPartition:
    """Greedy recursive 4σ-minimizing binary partition.

    Cost metric: combined 4σ range of all shifted clusters (compute_combined_4sigma_after_alignment).
    Recomputes cost on the full array each split step — O(n) but fast (~1ms for 65k samples).
    """

    def __init__(self, X, y, ref_median, min_count,
                 lower_pct, upper_pct, n_thresholds=20):
        self.X = X
        self.y = y
        self.ref_median = ref_median
        self.min_count = min_count
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.n_thresholds = n_thresholds
        self.n_features = X.shape[1]

    def _4sig(self, y_cl):
        return compute_4sigma_range_pct(y_cl, self.ref_median, self.lower_pct, self.upper_pct)

    def _find_best_split(self, cluster_id, four_sig_dict):
        """Find best (feature, threshold) to split cluster_id.
        Returns (feat_idx, threshold, new_cost) or (None, None, inf).
        """
        mask = self._labels == cluster_id
        X_cl = self.X[mask]

        best_cost = float("inf")
        best_feat, best_thresh = None, None
        next_id_tmp = int(self._labels.max()) + 1

        for feat_idx in range(self.n_features):
            vals = X_cl[:, feat_idx]
            thresholds = np.unique(
                np.percentile(vals, np.linspace(5, 95, self.n_thresholds))
            )

            for thresh in thresholds:
                left_mask_cl = vals <= thresh
                right_mask_cl = ~left_mask_cl
                n_L, n_R = left_mask_cl.sum(), right_mask_cl.sum()

                if n_L < self.min_count or n_R < self.min_count:
                    continue

                # Temporarily apply the split to compute cost
                temp_labels = self._labels.copy()
                temp_labels[mask] = np.where(left_mask_cl, cluster_id, next_id_tmp)
                new_cost = compute_combined_4sigma_after_alignment(
                    temp_labels, self.y, self.ref_median, self.lower_pct, self.upper_pct
                )

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh, best_cost

    def fit(self, target_k, min_split_improvement):
        """Grow the partition until target_k clusters or no improvement."""
        n = len(self.y)
        self._labels = np.zeros(n, dtype=int)

        # Initialize cluster state
        self._cluster_counts = {0: n}
        four_sig_dict = {0: self._4sig(self.y)}
        current_cost = compute_combined_4sigma_after_alignment(
            self._labels, self.y, self.ref_median, self.lower_pct, self.upper_pct
        )

        # Check min_count constraint
        if n < self.min_count:
            return relabel_sequential(self._labels), [{"n_clusters": 1, "cost": float("inf")}]

        history = [{"n_clusters": 1, "cost": current_cost}]

        while int(self._labels.max()) + 1 < target_k:
            # Find splittable clusters (need at least 2×min_count)
            splittable = {
                lbl: four_sig_dict[lbl]
                for lbl, cnt in self._cluster_counts.items()
                if cnt >= 2 * self.min_count
            }
            if not splittable:
                break

            # Try clusters ordered by 4σ (worst first)
            sorted_clusters = sorted(splittable, key=splittable.get, reverse=True)
            found_split = False

            for candidate in sorted_clusters:
                feat, thresh, split_cost = self._find_best_split(
                    candidate, four_sig_dict
                )
                if feat is None:
                    continue

                if current_cost - split_cost < min_split_improvement:
                    # No improvement from this cluster; try next worst
                    continue

                # Apply the split
                next_id = int(self._labels.max()) + 1
                mask = self._labels == candidate
                X_cl = self.X[mask]
                y_cl = self.y[mask]
                left_mask_cl = X_cl[:, feat] <= thresh

                self._labels[mask] = np.where(left_mask_cl, candidate, next_id)

                # Update tracking state
                n_L = int(left_mask_cl.sum())
                n_R = int((~left_mask_cl).sum())
                self._cluster_counts[candidate] = n_L
                self._cluster_counts[next_id] = n_R
                four_sig_dict[candidate] = self._4sig(y_cl[left_mask_cl])
                four_sig_dict[next_id] = self._4sig(y_cl[~left_mask_cl])
                current_cost = split_cost

                n_clusters = int(self._labels.max()) + 1
                history.append({
                    "n_clusters": n_clusters,
                    "cost": current_cost,
                    "split_cluster": int(candidate),
                    "split_feature_idx": int(feat),
                    "split_threshold": float(thresh),
                })
                found_split = True
                break

            if not found_split:
                break

        return relabel_sequential(self._labels), history


# ---------------------------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------------------------
def make_objective(X_sel, y, ref_median, cfg):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        target_k             = trial.suggest_int("target_k", 5, 100)
        n_thresholds         = trial.suggest_categorical("n_thresholds", [10, 20, 50])
        min_split_improvement = trial.suggest_float(
            "min_split_improvement", 1e-4, 0.1, log=True
        )

        t0 = time.perf_counter()

        partitioner = FourSigmaPartition(
            X_sel, y, ref_median, min_count,
            lower_pct, upper_pct, n_thresholds,
        )
        labels, history = partitioner.fit(target_k, min_split_improvement)

        cost = cost_function(
            labels, y, ref_median, min_count, lower_pct, upper_pct
        )

        duration   = time.perf_counter() - t0
        n_clusters = int(labels.max()) + 1

        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"target_k={target_k:3d} n_thresh={n_thresholds:2d} "
            f"min_imp={min_split_improvement:.4f} | "
            f"cost={cost_str} | clusters={n_clusters:3d} | {duration:.1f}s"
        )

        record = {
            "trial_number": trial.number + 1,
            "params": {
                "target_k": target_k,
                "n_thresholds": n_thresholds,
                "min_split_improvement": min_split_improvement,
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
        description="Optimize 4σ-Direct Recursive Partition clustering with Optuna"
    )
    parser.add_argument(
        "--n-trials", type=int, default=60,
        help="Number of Optuna trials (default: 60)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 2 trials (for testing)",
    )
    args = parser.parse_args()

    n_trials = 2 if args.dry_run else args.n_trials

    # Load data and config
    print(f"Loading preprocessed data from {DATA_PKL} ...")
    data = load_preprocessed(str(DATA_PKL))
    X_sel             = data["X_sel"]
    y                 = data["y"]
    ref_median        = data["overall_median_cd"]
    baseline_4sigma   = data.get("baseline_4sigma", None)
    selected_features = data.get("selected_features", None)

    print(f"Loading cost config from {COST_CFG} ...")
    cfg = load_cost_config(str(COST_CFG))

    print(
        f"\nStarting Optuna optimization: {n_trials} trials"
        + (" [DRY RUN]" if args.dry_run else "")
    )
    print(
        f"Fixed: min_count={cfg['min_count']}"
    )
    print(f"Data shape: X={X_sel.shape}, y={y.shape}")
    print("-" * 70)

    # Create study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")

    objective = make_objective(X_sel, y, ref_median, cfg)
    study.optimize(objective, n_trials=n_trials)

    # Best result — rerun to get labels and history
    best_trial  = study.best_trial
    best_params = best_trial.params
    best_cost   = best_trial.value

    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    partitioner = FourSigmaPartition(
        X_sel, y, ref_median, min_count,
        lower_pct, upper_pct, best_params["n_thresholds"],
    )
    best_labels, best_history = partitioner.fit(
        best_params["target_k"], best_params["min_split_improvement"]
    )
    best_n_clusters = int(best_labels.max()) + 1

    # Improvement over baseline
    improvement_pct = None
    if baseline_4sigma is not None and baseline_4sigma > 0 and best_cost != float("inf"):
        baseline_cost   = baseline_4sigma
        improvement_pct = round((baseline_cost - best_cost) / baseline_cost * 100, 4)

    # Save best result
    result = {
        "method": "4sigma_direct_partition",
        "best_cost": best_cost if best_cost != float("inf") else None,
        "best_params": best_params,
        "n_clusters": best_n_clusters,
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": improvement_pct,
    }
    save_best_result(str(BEST_PATH), result)

    # Save split tree as interpretable JSON
    # Convert feat_idx -> feature name where possible
    tree_records = []
    for step_idx, entry in enumerate(best_history):
        if step_idx == 0:
            # Initial single-cluster state — no split info
            continue
        feat_idx = entry.get("split_feature_idx")
        feat_name = (
            selected_features[feat_idx]
            if (selected_features is not None and feat_idx is not None
                and feat_idx < len(selected_features))
            else (f"feat_{feat_idx:03d}" if feat_idx is not None else None)
        )
        tree_records.append({
            "step": step_idx,
            "n_clusters": entry["n_clusters"],
            "cost": entry["cost"],
            "split_feature": feat_name,
            "split_threshold": entry.get("split_threshold"),
        })

    with open(TREE_PATH, "w") as fh:
        json.dump(tree_records, fh, indent=2)

    # Print summary
    print("-" * 70)
    print("Optimization complete.")
    if best_cost != float("inf"):
        print(f"  Best cost          : {best_cost:.4f}")
    else:
        print("  Best cost          : inf")
    print(
        f"  Best params        : target_k={best_params['target_k']}, "
        f"n_thresholds={best_params['n_thresholds']}, "
        f"min_split_improvement={best_params['min_split_improvement']:.6f}"
    )
    print(f"  Best n_clusters    : {best_n_clusters}")
    if baseline_4sigma is not None:
        print(f"  Baseline 4-sigma   : {baseline_4sigma:.4f}")
    if improvement_pct is not None:
        print(f"  Improvement        : {improvement_pct:+.2f}%")
    print(f"  Log saved to       : {LOG_PATH}")
    print(f"  Best result saved  : {BEST_PATH}")
    print(f"  Split tree saved   : {TREE_PATH}")


if __name__ == "__main__":
    main()
