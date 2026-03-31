"""Cost function 및 클러스터 통계 계산 모듈."""

from __future__ import annotations
import numpy as np


def compute_4sigma_range_pct(
    cd_values: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """4σ range % — 전체 median 기준으로 정규화.

    각 클러스터의 median을 전체 데이터 median(ref_median)으로 이동시킨 후
    4σ range를 측정하는 것과 수학적으로 동일합니다.

    이동(shift)은 range에 영향을 주지 않으므로:
      range(arr - cluster_median + ref_median) == range(arr)
    따라서:
      4σ range % = (P_upper(arr) - P_lower(arr)) / ref_median × 100
                 = (P_upper(arr/ref_median) - P_lower(arr/ref_median)) × 100
    """
    arr = np.asarray(cd_values, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    if ref_median == 0.0:
        return 0.0
    cd_norm = arr / ref_median
    lower = np.percentile(cd_norm, lower_pct)
    upper = np.percentile(cd_norm, upper_pct)
    return (upper - lower) * 100.0


def compute_cluster_stats(
    labels: np.ndarray,
    cd: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> dict:
    """클러스터별 CD 통계 계산 (4σ range % 기준).

    ref_median은 API 호환성을 위해 유지되나, 각 클러스터는 자체 median으로 정규화합니다.
    (ref_median은 cluster_median=0일 때 fallback으로만 사용됩니다.)

    Returns
    -------
    dict:
        four_sigma_range_pct   : {lbl: 4σ range %}
        cluster_counts         : {lbl: count}
        median_per_cluster     : {lbl: median CD}
        weighted_mean_4spct    : float
        unweighted_mean_4spct  : float
        max_4spct              : float
        n_clusters             : int
        min_count              : int
    """
    unique_labels = np.unique(labels)
    four_sigma: dict[int, float] = {}
    counts: dict[int, int] = {}
    medians: dict[int, float] = {}

    for lbl in unique_labels:
        mask = labels == lbl
        cd_cl = cd[mask]
        four_sigma[lbl] = compute_4sigma_range_pct(cd_cl, ref_median, lower_pct, upper_pct)
        counts[lbl] = int(mask.sum())
        medians[lbl] = float(np.median(cd_cl))

    ranges = np.array([four_sigma[l] for l in unique_labels])
    ns = np.array([counts[l] for l in unique_labels], dtype=float)

    return {
        "four_sigma_range_pct": four_sigma,
        "cluster_counts": counts,
        "median_per_cluster": medians,
        "weighted_mean_4spct": float(np.average(ranges, weights=ns)),
        "unweighted_mean_4spct": float(ranges.mean()),
        "max_4spct": float(ranges.max()),
        "n_clusters": len(unique_labels),
        "min_count": int(ns.min()),
    }


def cost_function(
    labels: np.ndarray,
    cd: np.ndarray,
    ref_median: float,
    alpha: float = 1.0,
    beta: float = 0.5,
    min_count: int = 100,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """Cost = α × weighted_mean(4σ range %) + β × max(4σ range %).

    각 클러스터는 자체 median으로 정규화 → OPC 레시피 보정 후 잔존 variation 기준.
    min_count 제약 위반 시 inf 반환.
    """
    stats = compute_cluster_stats(labels, cd, ref_median, lower_pct, upper_pct)
    if stats["min_count"] < min_count:
        return float("inf")
    return alpha * stats["weighted_mean_4spct"] + beta * stats["max_4spct"]
