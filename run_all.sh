#!/bin/bash
# run_all.sh — 전체 최적화 파이프라인 실행
# Usage: bash run_all.sh [--dry-run] [--skip-preprocess] [--parallel N]

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DRY_RUN=""
SKIP_PREPROCESS=0
PARALLEL=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --skip-preprocess) SKIP_PREPROCESS=1; shift ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPTS=(
    "optimization/optimize/01_decision_tree.py"
    "optimization/optimize/02_kmeans_minibatch.py"
    "optimization/optimize/03_autoencoder_kmeans.py"
    "optimization/optimize/04_gmm.py"
    "optimization/optimize/05_bisecting_kmeans.py"
    "optimization/optimize/06_agglomerative_ward.py"
    "optimization/optimize/07_hdbscan.py"
    "optimization/optimize/08_spectral.py"
    "optimization/optimize/09_isolation_forest_kmeans.py"
    "optimization/optimize/10_vae_kmeans.py"
    "optimization/optimize/11_dt_kmeans_twostage.py"
    "optimization/optimize/12_xgb_leaf_kmeans.py"
    "optimization/optimize/13_4sigma_direct_partition.py"
)

echo "=========================================="
echo " in-house-clustering 전체 최적화 파이프라인"
echo "=========================================="
[[ -n "$DRY_RUN" ]] && echo " [DRY RUN 모드]"
echo ""

# Step 1: Preprocess
if [[ $SKIP_PREPROCESS -eq 0 ]]; then
    echo "[Preprocess] optimization/preprocess.py 실행..."
    python3 optimization/preprocess.py
    echo ""
fi

# Step 2: Run optimization scripts
TOTAL=${#SCRIPTS[@]}
SUCCESS=0
FAILED=0
FAILED_LIST=()

run_script() {
    local script="$1"
    local name
    name=$(basename "$script" .py)
    echo "──────────────────────────────────────────"
    echo "  실행: $name $DRY_RUN"
    echo "──────────────────────────────────────────"
    if python3 "$script" $DRY_RUN; then
        echo "  ✓ $name 완료"
        return 0
    else
        echo "  ✗ $name 실패"
        return 1
    fi
}

if [[ $PARALLEL -eq 1 ]]; then
    # Sequential
    for script in "${SCRIPTS[@]}"; do
        if run_script "$script"; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$(basename "$script" .py)")
        fi
    done
else
    # Parallel (N at a time)
    pids=()
    names=()
    count=0
    for script in "${SCRIPTS[@]}"; do
        name=$(basename "$script" .py)
        python3 "$script" $DRY_RUN > "optimization/results/${name}.log" 2>&1 &
        pids+=($!)
        names+=("$name")
        count=$((count + 1))
        if [[ $count -ge $PARALLEL ]]; then
            for i in "${!pids[@]}"; do
                if wait "${pids[$i]}"; then
                    SUCCESS=$((SUCCESS + 1))
                    echo "✓ ${names[$i]}"
                else
                    FAILED=$((FAILED + 1))
                    FAILED_LIST+=("${names[$i]}")
                    echo "✗ ${names[$i]}"
                fi
            done
            pids=()
            names=()
            count=0
        fi
    done
    # Wait remaining
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            SUCCESS=$((SUCCESS + 1))
            echo "✓ ${names[$i]}"
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("${names[$i]}")
            echo "✗ ${names[$i]}"
        fi
    done
fi

# Step 3: Compare results
echo ""
echo "=========================================="
echo " 결과 비교"
echo "=========================================="
python3 optimization/compare_results.py

echo ""
echo "=========================================="
echo " 완료: $SUCCESS/$TOTAL 성공"
[[ $FAILED -gt 0 ]] && echo " 실패: ${FAILED_LIST[*]}"
echo "=========================================="
