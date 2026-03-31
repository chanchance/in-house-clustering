"""전처리 유틸리티: 소형 클러스터 병합, pkl/json 로드."""

from __future__ import annotations
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


# ── 클러스터 후처리 ────────────────────────────────────────────────────────────

def merge_small_clusters(labels: np.ndarray, X: np.ndarray, min_count: int) -> np.ndarray:
    """min_count 미만 클러스터를 centroid 거리 기준 가장 가까운 클러스터에 반복 병합."""
    labels = labels.copy()
    for _ in range(1000):
        counter = Counter(labels.tolist())
        small = sorted(
            [l for l, c in counter.items() if c < min_count],
            key=lambda l: counter[l],
        )
        if not small:
            break
        tgt = small[0]
        valid = [l for l in np.unique(labels) if l != tgt]
        if not valid:
            break
        mask = labels == tgt
        centroid = X[mask].mean(axis=0, keepdims=True)
        valid_centroids = np.array([X[labels == l].mean(axis=0) for l in valid])
        nearest = valid[np.linalg.norm(valid_centroids - centroid, axis=1).argmin()]
        labels[mask] = nearest
    return labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """레이블을 0부터 연속 정수로 재매핑."""
    mapping = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.vectorize(mapping.get)(labels)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_preprocessed(pkl_path: str | Path) -> dict:
    """전처리된 pkl 파일 로드.

    Returns dict with keys:
        X_sel, y, X_scaled, selected_features,
        scaler, overall_median_cd, baseline_4sigma,
        shap_importance (optional)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    required = {"X_sel", "y", "overall_median_cd", "baseline_4sigma"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"preprocessed.pkl에 필수 키 누락: {missing}")
    return data


def load_cost_config(json_path: str | Path) -> dict:
    """cost_function.json 로드. _comment 키는 제거 후 반환."""
    with open(json_path) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    return cfg


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def append_trial_log(log_path: str | Path, record: dict) -> None:
    """각 trial 결과를 jsonl 형식으로 누적 저장."""
    with open(log_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_best_result(result_path: str | Path, result: dict) -> None:
    """최적 결과를 JSON으로 저장."""
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Best result saved → {result_path}")
