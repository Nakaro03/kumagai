#!/usr/bin/env bash
# Paper + Patent Energy + Patent Construction の 3 ドメイン × 残り 4 seed (0/1/123/999) × alltime
# seed=42 は既に完了済み
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 123 999)
EPOCHS="${PNODE_EPOCHS:-200}"

echo "===================================================================="
echo "  X1 3-domain × 4-seed (alltime)"
echo "  epochs=$EPOCHS  seeds=${SEEDS[*]}"
echo "===================================================================="

# Patent Energy top-50
for S in "${SEEDS[@]}"; do
    echo ""
    echo "--- Patent Energy top-50 seed=$S ---"
    PNODE_DOMAIN_TARGET=patent_energy_top50 PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS \
        python -m pnode_patent_runner.run_pisde_x1 2>&1 \
        | grep -E "^  [0-9]+ |Saved" | tail -3 || true
done

# Patent Construction top-50
for S in "${SEEDS[@]}"; do
    echo ""
    echo "--- Patent Construction top-50 seed=$S ---"
    PNODE_DOMAIN_TARGET=patent_construction_top50 PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS \
        python -m pnode_patent_runner.run_pisde_x1 2>&1 \
        | grep -E "^  [0-9]+ |Saved" | tail -3 || true
done

echo ""
echo "===================================================================="
echo "  Complete"
echo "===================================================================="
