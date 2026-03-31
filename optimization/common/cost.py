"""Cost function 및 클러스터 통계 계산 모듈.

핵심 지표: 클러스터 median 정렬 후 합산 분포의 4σ range %

각 클러스터의 median을 전체 데이터 median(ref_median)으로 이동시킨 후
전체 합산 분포의 4σ range를 측정합니다.

  분리 전: 4σ range % of all CD data
  분리 후: 각 cluster median → ref_median 정렬 → 합산 분포의 4σ range %
  개선율 = (분리 전 - 분리 후) / 분리 전 × 100
"""

from __future__ import annotations
import numpy as np


def compute_4sigma_range_pct(
    cd_values: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """단일 배열의 4σ range % (ref_median 기준 정규화).

    4σ range % = (P_upper(arr/ref_median) - P_lower(arr/ref_median)) × 100
    """
    arr = np.asarray(cd_values, dtype=np.float64)
    if len(arr) < 2 or ref_median == 0.0:
        return 0.0
    cd_norm = arr / ref_median
    lower = np.percentile(cd_norm, lower_pct)
    upper = np.percentile(cd_norm, upper_pct)
    return (upper - lower) * 100.0


def compute_combined_4sigma_after_alignment(
    labels: np.ndarray,
    cd: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """각 클러스터 median → ref_median 정렬 후 합산 분포의 4σ range %.

    OPC 레시피가 각 클러스터의 평균 CD를 ref_median으로 보정할 때
    전체 웨이퍼에 걸쳐 잔존하는 CD 산포를 측정합니다.

    Steps:
      1. cluster i의 모든 CD 값을 (ref_median - median_i) 만큼 shift
      2. 모든 클러스터의 shifted CD를 합산
      3. 합산 분포의 4σ range % 계산
    """
    labels = np.asarray(labels)
    cd = np.asarray(cd, dtype=np.float64)
    if ref_median == 0.0:
        return 0.0

    parts = []
    for lbl in np.unique(labels):
        cd_cl = cd[labels == lbl]
        if len(cd_cl) == 0:
            continue
        cluster_median = float(np.median(cd_cl))
        parts.append(cd_cl - cluster_median + ref_median)

    if not parts:
        return 0.0

    combined = np.concatenate(parts)
    cd_norm = combined / ref_median
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
    """클러스터별 CD 통계 + 합산 정렬 후 4σ range.

    Returns
    -------
    dict:
        combined_4sigma_pct    : float  — 실제 cost 지표 (클러스터 정렬 후 합산 4σ range %)
        four_sigma_range_pct   : {lbl: 단일 클러스터 4σ range %}  (참고용)
        cluster_counts         : {lbl: count}
        median_per_cluster     : {lbl: median CD}
        weighted_mean_4spct    : float  (참고용)
        max_4spct              : float  (참고용)
        n_clusters             : int
        min_count              : int
    """
    labels = np.asarray(labels)
    cd = np.asarray(cd, dtype=np.float64)
    unique_labels = np.unique(labels)

    four_sigma: dict[int, float] = {}
    counts: dict[int, int] = {}
    medians: dict[int, float] = {}

    for lbl in unique_labels:
        mask = labels == lbl
        cd_cl = cd[mask]
        four_sigma[int(lbl)] = compute_4sigma_range_pct(cd_cl, ref_median, lower_pct, upper_pct)
        counts[int(lbl)] = int(mask.sum())
        medians[int(lbl)] = float(np.median(cd_cl))

    combined_4sigma = compute_combined_4sigma_after_alignment(
        labels, cd, ref_median, lower_pct, upper_pct
    )

    ranges = np.array([four_sigma[int(l)] for l in unique_labels])
    ns = np.array([counts[int(l)] for l in unique_labels], dtype=float)

    return {
        "combined_4sigma_pct": combined_4sigma,
        "four_sigma_range_pct": four_sigma,
        "cluster_counts": counts,
        "median_per_cluster": medians,
        "weighted_mean_4spct": float(np.average(ranges, weights=ns)),
        "max_4spct": float(ranges.max()),
        "n_clusters": len(unique_labels),
        "min_count": int(ns.min()),
    }


def cost_function(
    labels: np.ndarray,
    cd: np.ndarray,
    ref_median: float,
    min_count: int = 100,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
    **kwargs,
) -> float:
    """Cost = 클러스터 median 정렬 후 합산 분포의 4σ range %.

    각 클러스터의 median을 ref_median으로 이동 → 합산 → 4σ range % 측정.
    min_count 제약 위반 시 inf 반환.

    **kwargs: alpha, beta 등 이전 인자 무시 (backward compatibility)
    """
    labels = np.asarray(labels)
    unique_labels, ns = np.unique(labels, return_counts=True)
    if ns.min() < min_count:
        return float("inf")
    return compute_combined_4sigma_after_alignment(labels, cd, ref_median, lower_pct, upper_pct)
