#!/usr/bin/env bash
# PI-SDE + X1 (Topic-Anchor) を 5 seed × 4 condition で実行。
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=(0 1 42 123 999)
EPOCHS="${PNODE_EPOCHS:-300}"   # 効率化のため 500 → 300

echo "===================================================================="
echo "  PI-SDE + X1  5-seed × 4-condition"
echo "  epochs=$EPOCHS  seeds=${SEEDS[*]}"
echo "===================================================================="

# 1) alltime
for S in "${SEEDS[@]}"; do
    echo ""
    echo "--- X1 alltime seed=$S ---"
    PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS PNODE_LEAVEOUT_T="" \
        python -m pnode_patent_runner.run_pisde_x1 2>&1 \
        | grep -E "^  [123] |Saved" | tail -5 || true
done

# 2) leaveout 1, 2, 3
for LO in 1 2 3; do
    for S in "${SEEDS[@]}"; do
        echo ""
        echo "--- X1 leaveout$LO seed=$S ---"
        PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS PNODE_LEAVEOUT_T=$LO \
            python -m pnode_patent_runner.run_pisde_x1 2>&1 \
            | grep -E "^  [123] |Saved" | tail -5 || true
    done
done

echo ""
echo "===================================================================="
echo "  X1 5-seed runs complete"
echo "===================================================================="
