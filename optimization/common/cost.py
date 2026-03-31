"""Cost function 및 클러스터 통계 계산 모듈.

핵심 지표: 클러스터 median 정렬 후 합산 분포의 4σ range %

3가지 cost mode 지원:
  - "combined"        : combined 4σ range after alignment (default)
  - "soft_penalty"    : combined_4σ + λ × weighted_mean(per_cluster_4σ%)
  - "hard_constraint" : max(per_cluster_4σ%) > baseline × ratio → inf, else combined_4σ
"""

from __future__ import annotations
import numpy as np


def compute_4sigma_range_pct(
    cd_values: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """단일 배열의 4σ range % (ref_median 기준 정규화)."""
    arr = np.asarray(cd_values, dtype=np.float64)
    if len(arr) < 2 or ref_median == 0.0:
        return 0.0
    cd_norm = arr / ref_median
    return (np.percentile(cd_norm, upper_pct) - np.percentile(cd_norm, lower_pct)) * 100.0


def compute_combined_4sigma_after_alignment(
    labels: np.ndarray,
    cd: np.ndarray,
    ref_median: float,
    lower_pct: float = 0.00315,
    upper_pct: float = 99.99685,
) -> float:
    """각 클러스터 median → ref_median 정렬 후 합산 분포의 4σ range %.

    OPC 레시피가 각 클러스터의 median CD를 ref_median으로 보정할 때
    전체 웨이퍼에 잔존하는 CD 산포를 측정합니다.
    """
    if ref_median == 0.0:
        return 0.0
    parts = []
    for lbl in np.unique(labels):
        cd_cl = cd[labels == lbl]
        if len(cd_cl) == 0:
            continue
        cl_med = float(np.median(cd_cl))
        parts.append(cd_cl - cl_med + ref_median)
    if not parts:
        return 0.0
    combined = np.concatenate(parts)
    cd_norm = combined / ref_median
    return (np.percentile(cd_norm, upper_pct) - np.percentile(cd_norm, lower_pct)) * 100.0


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
        combined_4sigma_pct    : float  — mode="combined" cost 지표
        four_sigma_range_pct   : {lbl: 단일 클러스터 4σ range %}
        cluster_counts         : {lbl: count}
        median_per_cluster     : {lbl: median CD}
        weighted_mean_4spct    : float
        max_4spct              : float
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
        "unweighted_mean_4spct": float(ranges.mean()),
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
    cost_mode: str = "combined",
    lambda_penalty: float = 0.3,
    baseline_4sigma: float | None = None,
    max_cluster_4sigma_threshold_ratio: float = 0.8,
    **kwargs,
) -> float:
    """3-mode cost function.

    Parameters
    ----------
    cost_mode : str
        "combined"        — combined 4σ range after alignment (기본값)
        "soft_penalty"    — combined_4σ + lambda_penalty × weighted_mean(per_cluster_4σ%)
        "hard_constraint" — max(per_cluster_4σ%) > baseline_4σ × ratio → inf
    lambda_penalty : float
        soft_penalty 모드 가중치 (default 0.3)
    baseline_4sigma : float | None
        hard_constraint 모드의 threshold 계산 기준.
        None이면 threshold 체크 없이 combined_4σ 반환.
    max_cluster_4sigma_threshold_ratio : float
        hard_constraint 모드: max(per_cluster_4σ%) > baseline × ratio → inf
    **kwargs : ignored
        backward compatibility용 (alpha, beta 등 이전 인자 무시)
    """
    labels = np.asarray(labels)
    unique_labels, ns = np.unique(labels, return_counts=True)
    if ns.min() < min_count:
        return float("inf")

    combined = compute_combined_4sigma_after_alignment(
        labels, cd, ref_median, lower_pct, upper_pct
    )

    if cost_mode == "combined":
        return combined

    # 공통: per-cluster 4σ 계산
    per_cluster_4sigma = np.array([
        compute_4sigma_range_pct(cd[labels == lbl], ref_median, lower_pct, upper_pct)
        for lbl in unique_labels
    ])

    if cost_mode == "soft_penalty":
        weighted_mean = float(np.average(per_cluster_4sigma, weights=ns))
        return combined + lambda_penalty * weighted_mean

    elif cost_mode == "hard_constraint":
        max_4sigma = float(per_cluster_4sigma.max())
        if baseline_4sigma is not None:
            threshold = baseline_4sigma * max_cluster_4sigma_threshold_ratio
            if max_4sigma > threshold:
                return float("inf")
        return combined

    else:
        raise ValueError(
            f"Unknown cost_mode: {cost_mode!r}. "
            "Use 'combined', 'soft_penalty', or 'hard_constraint'."
        )
