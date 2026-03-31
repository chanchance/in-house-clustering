"""02_kmeans_minibatch.py — MiniBatchKMeans hyperparameter optimization via Optuna.

Usage:
    python optimization/optimize/02_kmeans_minibatch.py [--n-trials N] [--dry-run]
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
LOG_PATH = ROOT / "optimization" / "results" / "02_kmeans_minibatch_log.jsonl"
BEST_PATH = ROOT / "optimization" / "results" / "02_kmeans_minibatch_best.json"


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_sel, y, ref_median, cfg):
    min_count = cfg["min_count"]
    lower_pct = cfg["lower_pct"]
    upper_pct = cfg["upper_pct"]

    def objective(trial):
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = trial.suggest_int("n_clusters", 5, 100)
        n_init = trial.suggest_int("n_init", 3, 20)
        batch_size = trial.suggest_categorical("batch_size", [1000, 5000, 10000, 20000])

        t0 = time.perf_counter()
        km = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            n_init=n_init,
            random_state=42,
        )
        raw_labels = km.fit_predict(X_sel)
        merged = merge_small_clusters(raw_labels, X_sel, min_count)
        labels = relabel_sequential(merged)
        cost = cost_function(labels, y, ref_median, min_count, lower_pct, upper_pct)
        elapsed = time.perf_counter() - t0

        n_actual = int(labels.max()) + 1
        cost_str = f"{cost:.4f}" if cost != float("inf") else "inf"
        print(
            f"Trial {trial.number + 1:03d} | "
            f"k={n_clusters} n_init={n_init} batch={batch_size} | "
            f"cost={cost_str} | clusters={n_actual} | {elapsed:.1f}s"
        )

        append_trial_log(
            LOG_PATH,
            {
                "trial": trial.number + 1,
                "n_clusters": n_clusters,
                "n_init": n_init,
                "batch_size": batch_size,
                "cost": cost if cost != float("inf") else None,
                "n_actual_clusters": n_actual,
                "elapsed_sec": round(elapsed, 3),
            },
        )

        return cost

    return objective


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MiniBatchKMeans optimization via Optuna")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
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
    print()

    # 결과 디렉토리 보장
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(X_sel, y, ref_median, cfg),
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
        "method": "MiniBatchKMeans",
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
