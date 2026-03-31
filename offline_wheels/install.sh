#!/bin/bash
# 폐쇄망 환경 optuna 오프라인 설치 스크립트
# 대상: Linux x86_64, Python 3.12.x (cp312)
# 사용: bash offline_wheels/install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo " optuna 오프라인 설치 (Linux x86_64 cp312)"
echo "=========================================="
echo "wheel 경로: $SCRIPT_DIR"
echo ""

pip install --no-index --find-links "$SCRIPT_DIR" \
    greenlet \
    sqlalchemy \
    PyYAML \
    alembic \
    colorlog \
    packaging \
    tqdm \
    optuna

echo ""
echo "설치 완료. 버전 확인:"
python3 -c "import optuna; print(f'  optuna {optuna.__version__}')"
