#!/usr/bin/env bash
# 複数ドメイン × 5 seed の trend benchmark をシーケンシャル実行。
#
# 使用:
#   bash pnode_patent_runner/run_multidomain_5seed.sh [domain1 domain2 ...]
#
# 環境変数:
#   PNODE_EPOCHS      (default: 15)
#   PNODE_YEAR_START  (default: 2010)
#   PNODE_YEAR_END    (default: 2021)
#   PNODE_MIN_EVENTS  (default: 200)
#   PNODE_USE_A       (default: 1)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DOMAINS=("$@")
if [ ${#DOMAINS[@]} -eq 0 ]; then
    DOMAINS=(energy agrifood construction pharma semiconductor computing)
fi

SEEDS=(0 1 42 123 999)

EPOCHS="${PNODE_EPOCHS:-15}"
YEAR_START="${PNODE_YEAR_START:-2010}"
YEAR_END="${PNODE_YEAR_END:-2021}"
MIN_EVENTS="${PNODE_MIN_EVENTS:-200}"
USE_A="${PNODE_USE_A:-1}"

echo "===================================================================="
echo "  Multi-domain 5-seed trend benchmark"
echo "===================================================================="
echo "  Domains: ${DOMAINS[@]}"
echo "  Seeds:   ${SEEDS[@]}"
echo "  Years:   ${YEAR_START}-${YEAR_END}  Epochs: ${EPOCHS}  MinEvents: ${MIN_EVENTS}"
echo "  USE_A:   ${USE_A}"
echo "===================================================================="

for D in "${DOMAINS[@]}"; do
    for S in "${SEEDS[@]}"; do
        echo ""
        echo "--- Domain=$D Seed=$S ---"
        PNODE_DOMAIN="$D" PNODE_SEED="$S" PNODE_USE_A="$USE_A" \
        PNODE_EPOCHS="$EPOCHS" PNODE_YEAR_START="$YEAR_START" \
        PNODE_YEAR_END="$YEAR_END" PNODE_MIN_EVENTS="$MIN_EVENTS" \
            python -m pnode_patent_runner.run_multidomain_trend 2>&1 \
            | grep -E "pnode_pc.*Link|Saved|Error|Traceback" | tail -3 || true
    done
done

echo ""
echo "===================================================================="
echo "  全実行完了"
echo "===================================================================="
