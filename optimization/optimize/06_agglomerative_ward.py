"""06_agglomerative_ward.py — AgglomerativeClustering (Ward) hyperparameter optimization via Optuna.

Usage:
    python optimization/optimize/06_agglomerative_ward.py [--n-trials N] [--dry-run]
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

from optimization.common.cost import cost_function
from optimization.common.utils import (
    append_trial_log,
    load_cost_config,
    load_preprocessed,
    merge_small_clusters,
    relabel_sequential,
    save_best_result,
)

# ── 경로 상수 ─────────────────────────────────────────────────────────────────
DATA_PKL = ROOT / "optimization" / "preprocessed.pkl"
COST_JSON = ROOT / "optimization" / "cost_function.json"
LOG_PATH = ROOT / "optimization" / "results" / "06_agglomerative_ward_log.jsonl"
BEST_PATH = ROOT / "optimization" / "results" / "06_agglomerative_ward_best.json"


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
        from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

        n_clusters = trial.suggest_int("n_clusters", 5, 80)
        subsample_n = trial.suggest_categorical("subsample_n", [5000, 10000, 20000])

        t0 = time.perf_counter()

        # Agglomerative는 O(n²) 메모리 → subsample 후 nearest-centroid assign
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_sel), min(subsample_n, len(X_sel)), replace=False)
        X_sub = X_sel[idx]

        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        sub_labels = agg.fit_predict(X_sub)

        # Assign full dataset via nearest centroid
        centroids = np.array(
            [X_sub[sub_labels == l].mean(axis=0) for l in np.unique(sub_labels)]
        )
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            init=centroids,
            n_init=1,
            max_iter=300,
            random_state=42,
        )
        raw_labels = km.fit_predict(X_sel)
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
            f"k={n_clusters} sub={subsample_n} | "
            f"cost={cost_str} | clusters={n_actual} | {elapsed:.1f}s"
        )

        append_trial_log(
            LOG_PATH,
            {
                "trial": trial.number + 1,
                "n_clusters": n_clusters,
                "subsample_n": subsample_n,
                "cost": cost if cost != float("inf") else None,
                "n_actual_clusters": n_actual,
                "elapsed_sec": round(elapsed, 3),
            },
        )

        return cost

    return objective


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgglomerativeClustering (Ward) optimization via Optuna")
    parser.add_argument("--n-trials", type=int, default=60, help="Number of Optuna trials")
    parser.add_argument("--dry-run", action="store_true", help="Run 3 trials for testing")
    args = parser.parse_args()

    n_trials = 3 if args.dry_run else args.n_trials

    # 데이터 및 설정 로드
    print(f"Loading data from {DATA_PKL} ...")
    data = load_preprocessed(DATA_PKL)
    X_sel = data["X_sel"]
    y = data["y"]
    ref_median = data["overall_median_cd"]
    baseline_4sigma = data["baseline_4sigma"]

    cfg = load_cost_config(COST_JSON)

    print(f"  X_sel shape : {X_sel.shape}")
    print(f"  ref_median  : {ref_median:.4f}")
    print(f"  baseline 4σ : {baseline_4sigma:.4f}")
    print(f"  n_trials    : {n_trials}")
    print(f"Cost mode: {cfg.get('cost_mode', 'combined')}")
    print()

    # 결과 디렉토리 보장
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(X_sel, y, ref_median, cfg, baseline_4sigma=baseline_4sigma),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best = study.best_trial
    best_cost = best.value
    best_params = best.params

    baseline_cost = baseline_4sigma
    improvement_pct = (
        (baseline_cost - best_cost) / baseline_cost * 100.0
        if baseline_4sigma > 0
        else 0.0
    )

    result = {
        "method": "AgglomerativeWard",
        "best_cost": best_cost,
        "best_params": best_params,
        "n_clusters": best_params["n_clusters"],
        "baseline_4sigma": baseline_4sigma,
        "improvement_pct": round(improvement_pct, 4),
    }

    save_best_result(BEST_PATH, result)

    print()
    print("=" * 60)
    print(f"Best cost        : {best_cost:.4f}")
    print(f"Best params      : {best_params}")
    print(f"Baseline 4σ      : {baseline_4sigma:.4f}")
    print(f"Improvement      : {improvement_pct:.2f}%")
    print(f"Log              : {LOG_PATH}")
    print(f"Best result      : {BEST_PATH}")


if __name__ == "__main__":
    main()
