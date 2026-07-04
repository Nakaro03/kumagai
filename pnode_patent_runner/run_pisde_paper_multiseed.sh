#!/usr/bin/env bash
# PI-SDE 論文ドメイン: 5 seed × {alltime, leaveout_1, leaveout_2, leaveout_3}
# 既存の seed=42 (alltime / leaveout3) は再実行をスキップ
set -uo pipefail

cd "$(dirname "$0")/.."

SEEDS=(0 1 42 123 999)
EPOCHS=300                       # 効率化のため 500 → 300
DOMAIN=paper

echo "===================================================================="
echo "  PI-SDE 5-seed × 4-condition  paper domain"
echo "  epochs=$EPOCHS  seeds=${SEEDS[*]}"
echo "===================================================================="

# 1) alltime
for S in "${SEEDS[@]}"; do
    echo ""
    echo "--- alltime seed=$S ---"
    PNODE_DOMAIN=$DOMAIN PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS PNODE_LEAVEOUT_T="" \
        python -m pnode_patent_runner.run_pisde_eval 2>&1 \
        | grep -E "split|PI-SDE.*Naive|^  [123] " | tail -6 || true
done

# 2) leaveout_1, _2, _3
for LO in 1 2 3; do
    for S in "${SEEDS[@]}"; do
        echo ""
        echo "--- leaveout$LO seed=$S ---"
        PNODE_DOMAIN=$DOMAIN PNODE_SEED=$S PNODE_EPOCHS=$EPOCHS PNODE_LEAVEOUT_T=$LO \
            python -m pnode_patent_runner.run_pisde_eval 2>&1 \
            | grep -E "split|PI-SDE.*Naive|^  [123] " | tail -6 || true
    done
done

echo ""
echo "===================================================================="
echo "  All runs complete"
echo "===================================================================="
