#!/usr/bin/env bash
# X5 full sweep: 3 domain × 5 seed × 200 epoch (LOTO enabled, A0 config).
#
# Usage:
#   pnode_patent_runner/scripts/run_x5_sweep.sh                # all 15 runs sequentially
#   PNODE_SEEDS=42,0 DOMAINS=patent_energy_top50 \
#     pnode_patent_runner/scripts/run_x5_sweep.sh              # custom subset
#
# Output: RESULTS_X5/{DATA_NAME}/seed_{S}/loto/{config,evaluation,train.*}
# Logs:   logs_x5/x5_{domain}_seed{S}.log

set -u
cd /home/nakamuraroi/kumagai

DOMAINS="${DOMAINS:-patent_energy_top50 arxiv_construction jp_construction}"
SEEDS="${PNODE_SEEDS:-42 0 1 123 999}"
EPOCHS="${PNODE_EPOCHS:-200}"
WARMUP="${PNODE_WARMUP:-40}"

mkdir -p logs_x5

idx=0
for D in $DOMAINS; do
  for S in $(echo "$SEEDS" | tr ',' ' '); do
    idx=$((idx+1))
    LOG="logs_x5/x5_${D}_seed${S}.log"
    DATA_NAME=$(python -c "from pnode_patent_runner.x5.config import DOMAIN_TABLE; print(DOMAIN_TABLE['${D}'][0])")
    EVAL="RESULTS_X5/${DATA_NAME}/seed_${S}/loto/evaluation.json"
    CFG="RESULTS_X5/${DATA_NAME}/seed_${S}/loto/config.json"
    if [[ -f "$EVAL" ]] && [[ -f "$CFG" ]] && [[ "${FORCE:-0}" != "1" ]]; then
      EXISTING_EP=$(python -c "import json; print(json.load(open('${CFG}')).get('epochs', 0))" 2>/dev/null || echo 0)
      if (( EXISTING_EP >= EPOCHS )); then
        echo "  [skip $idx] $D seed=$S — evaluation.json (epochs=$EXISTING_EP) ≥ requested $EPOCHS"
        continue
      fi
    fi
    echo "[$idx] domain=$D seed=$S epochs=$EPOCHS  →  $LOG"
    PNODE_DOMAIN_TARGET="$D" PNODE_SEED="$S" PNODE_EPOCHS="$EPOCHS" PNODE_WARMUP="$WARMUP" \
      PNODE_LOTO=1 \
      python pnode_patent_runner/run_pisde_x5.py > "$LOG" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "  [FAIL rc=$rc] tail of log:"
      tail -20 "$LOG"
    fi
  done
done
echo "Done. Runs attempted: $idx"
