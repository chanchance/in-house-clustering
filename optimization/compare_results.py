"""compare_results.py — 전체 최적화 결과 비교 및 랭킹 출력."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "optimization" / "results"

SCRIPTS = [
    ("01", "decision_tree"),
    ("02", "kmeans_minibatch"),
    ("03", "autoencoder_kmeans"),
    ("04", "gmm"),
    ("05", "bisecting_kmeans"),
    ("06", "agglomerative_ward"),
    ("07", "hdbscan"),
    ("08", "spectral"),
    ("09", "isolation_forest_kmeans"),
    ("10", "vae_kmeans"),
    ("11", "dt_kmeans_twostage"),
    ("12", "xgb_leaf_kmeans"),
    ("13", "4sigma_direct_partition"),
]


def load_results():
    results = []
    for num, name in SCRIPTS:
        path = RESULTS_DIR / f"{num}_{name}_best.json"
        if not path.exists():
            results.append({"num": num, "name": name, "available": False})
            continue
        with open(path) as f:
            data = json.load(f)
        results.append({
            "num": num,
            "name": name,
            "available": True,
            "best_cost": data.get("best_cost"),
            "n_clusters": data.get("n_clusters"),
            "improvement_pct": data.get("improvement_pct"),
            "baseline_4sigma": data.get("baseline_4sigma"),
        })
    return results


def print_ranking(results):
    # Sort by improvement_pct descending (None last)
    available   = [r for r in results if r["available"] and r["best_cost"] is not None]
    unavailable = [r for r in results if not r["available"] or r["best_cost"] is None]
    available.sort(key=lambda x: x.get("improvement_pct") or -999, reverse=True)

    print("=" * 75)
    print(f"{'Rank':<5} {'Num':<4} {'Method':<35} {'Best Cost':>10} {'Improve':>8} {'k':>5}")
    print("-" * 75)
    for rank, r in enumerate(available, 1):
        imp = f"{r['improvement_pct']:+.2f}%" if r["improvement_pct"] is not None else "N/A"
        print(
            f"{rank:<5} {r['num']:<4} {r['name']:<35} "
            f"{r['best_cost']:>10.4f} {imp:>8} {r['n_clusters']:>5}"
        )

    if unavailable:
        print("-" * 75)
        for r in unavailable:
            print(f"{'--':<5} {r['num']:<4} {r['name']:<35} {'(not run)':>10}")

    print("=" * 75)
    if available:
        best = available[0]
        print(f"\nBest method: [{best['num']}] {best['name']}")
        if best["improvement_pct"] is not None:
            print(
                f"  Improvement: {best['improvement_pct']:+.2f}%  "
                f"(baseline 4\u03c3={best.get('baseline_4sigma', 'N/A')})"
            )
        print(f"  Best cost: {best['best_cost']:.4f}  n_clusters={best['n_clusters']}")


if __name__ == "__main__":
    results = load_results()
    print_ranking(results)
