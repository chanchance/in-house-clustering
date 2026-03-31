"""전처리 스크립트 — SHAP 분석 포함, preprocessed.pkl 생성.

Usage
-----
    python optimization/preprocess.py                        # 합성 데이터 사용
    python optimization/preprocess.py --data data/layout_features.csv
    python optimization/preprocess.py --output optimization/preprocessed.pkl
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import shap
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from optimization.common.cost import compute_4sigma_range_pct

# ── 기본 경로 ──────────────────────────────────────────────────────────────────
DEFAULT_CFG  = ROOT / "optimization" / "cost_function.json"
DEFAULT_OUT  = ROOT / "optimization" / "preprocessed.pkl"
DEFAULT_SEED = 42

CD_COLUMN = "CD (nm)"
N_ROWS    = 64_955
N_FEAT    = 126


# ── 합성 데이터 생성 ───────────────────────────────────────────────────────────

def generate_synthetic_data(n_rows: int, n_feat: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    group = rng.choice([0, 1, 2, 3, 4], size=n_rows, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    X     = rng.standard_normal((n_rows, n_feat)).astype(np.float32)
    for g in range(5):
        mask  = group == g
        X[mask] = X[mask] * rng.uniform(0.5, 1.5, n_feat) + rng.uniform(-2, 2, n_feat)
    X[rng.random((n_rows, n_feat)) < 0.01] = np.nan

    feat_names = [f"feat_{i:03d}" for i in range(n_feat)]
    df   = pd.DataFrame(X, columns=feat_names)
    coef = rng.uniform(-1, 1, n_feat)
    cd_base     = df.fillna(0).values @ coef * 0.3
    group_offset = np.array([50, 70, 90, 110, 130])[group]
    noise_std    = np.array([3,   5,  4,   6,   8])[group]
    df[CD_COLUMN]    = cd_base + group_offset + rng.normal(0, noise_std, n_rows)
    df["_true_group"] = group
    return df


# ── 전처리 파이프라인 ──────────────────────────────────────────────────────────

def preprocess(
    data_path: str | None,
    cfg_path: Path,
    output_path: Path,
    seed: int = DEFAULT_SEED,
) -> None:
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)

    lower_pct  = cfg["lower_pct"]
    upper_pct  = cfg["upper_pct"]
    shap_top_k = cfg["shap_top_k"]
    shap_n     = cfg["shap_sample_n"]

    # 1. 데이터 로드
    print("=" * 60)
    if data_path:
        print(f"[1/6] 데이터 로드: {data_path}")
        df_raw = pd.read_csv(data_path)
    else:
        print(f"[1/6] 합성 데이터 생성 ({N_ROWS:,} rows × {N_FEAT} features)")
        df_raw = generate_synthetic_data(N_ROWS, N_FEAT, seed)

    feat_cols = [c for c in df_raw.columns if c not in [CD_COLUMN, "_true_group"]]
    print(f"      feature 수: {len(feat_cols)}, 행 수: {len(df_raw):,}")

    # 2. 전처리
    print("[2/6] 전처리 (저분산 제거, 결측 대체, RobustScaler)")
    feat_var = df_raw[feat_cols].var()
    feat_cols_clean = feat_var[feat_var >= 1e-6].index.tolist()
    print(f"      저분산 제거 후: {len(feat_cols_clean)}개 (제거: {len(feat_cols)-len(feat_cols_clean)}개)")

    df_clean = df_raw.dropna(subset=[CD_COLUMN]).copy()
    X_df     = df_clean[feat_cols_clean].fillna(df_clean[feat_cols_clean].median())
    y        = df_clean[CD_COLUMN].values.astype(np.float64)

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X_df).astype(np.float32)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feat_cols_clean, index=X_df.index)

    overall_median_cd  = float(np.median(y))
    # baseline: 전체 데이터를 하나의 클러스터로 볼 때의 4σ range %
    # (cluster_median = overall_median이므로 ref_median 사용과 동일)
    baseline_4sigma    = compute_4sigma_range_pct(y, overall_median_cd, lower_pct, upper_pct)
    print(f"      전체 median CD: {overall_median_cd:.4f} nm")
    print(f"      기준선 4σ range %: {baseline_4sigma:.4f} %")

    # 3. XGBoost 학습
    print("[3/6] XGBoost CD 예측 모델 학습")
    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=seed)
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=seed, verbosity=0,
        eval_metric="rmse", early_stopping_rounds=20,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = xgb_model.predict(X_val)
    rmse   = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    r2     = float(1 - np.sum((y_pred - y_val) ** 2) / np.sum((y_val - y_val.mean()) ** 2))
    print(f"      XGBoost RMSE={rmse:.3f} nm, R²={r2:.4f}")

    xgb_path = output_path.parent / "xgb_cd_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"      모델 저장 → {xgb_path}")

    # 4. SHAP 분석
    print(f"[4/6] SHAP 분석 (sample {shap_n:,}개, Top-{shap_top_k} 선택)")
    rng_shap  = np.random.RandomState(seed)
    shap_idx  = rng_shap.choice(len(X_scaled), min(shap_n, len(X_scaled)), replace=False)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_scaled[shap_idx])

    shap_importance = pd.Series(
        np.abs(shap_vals).mean(axis=0), index=feat_cols_clean
    ).sort_values(ascending=False)

    selected_features = shap_importance.head(shap_top_k).index.tolist()
    X_sel = X_scaled_df[selected_features].values.astype(np.float32)

    cumsum_pct = shap_importance.cumsum() / shap_importance.sum() * 100
    print(f"      Top-{shap_top_k} feature → 전체 SHAP의 {cumsum_pct.iloc[shap_top_k-1]:.1f}% 커버")
    print(f"      선택된 feature: {selected_features[:5]} ...")

    # 5. 저장
    print(f"[5/6] pkl 저장 → {output_path}")
    payload = {
        "X_sel":             X_sel,
        "y":                 y,
        "X_scaled":          X_scaled,
        "X_scaled_df":       X_scaled_df,
        "selected_features": selected_features,
        "feat_cols_clean":   feat_cols_clean,
        "scaler":            scaler,
        "overall_median_cd": overall_median_cd,
        "baseline_4sigma":   baseline_4sigma,
        "shap_importance":   shap_importance.to_dict(),
        "shap_values":       shap_vals,
        "shap_sample_idx":   shap_idx,
        "cfg":               cfg,
        "seed":              seed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"      파일 크기: {size_mb:.1f} MB")

    # 6. 요약
    print("[6/6] 완료 요약")
    print(f"      행 수: {len(y):,}  |  특징 수(선택): {len(selected_features)} / {len(feat_cols_clean)}")
    print(f"      기준선 4σ range %: {baseline_4sigma:.4f}%")
    print(f"      cost_function: {cfg.get('cost_mode', 'combined')}  (min_count≥{cfg['min_count']})")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="전처리 + SHAP 분석 → preprocessed.pkl 생성")
    parser.add_argument("--data",   default=None,              help="입력 CSV 경로 (없으면 합성 데이터)")
    parser.add_argument("--config", default=str(DEFAULT_CFG),  help="cost_function.json 경로")
    parser.add_argument("--output", default=str(DEFAULT_OUT),  help="출력 pkl 경로")
    parser.add_argument("--seed",   default=DEFAULT_SEED, type=int)
    args = parser.parse_args()

    preprocess(
        data_path   = args.data,
        cfg_path    = Path(args.config),
        output_path = Path(args.output),
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
