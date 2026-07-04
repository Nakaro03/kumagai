#!/usr/bin/env bash
# Run PRESCIENT- and MIOFlow-style reimplementations across the 3 domains × 5 seeds.
# Output paths: RESULTS_X5/{DATA}/{PRESCIENT|MIOFLOW}/seed_{S}/alltime/

set -u
cd /home/nakamuraroi/kumagai

DOMAINS="${DOMAINS:-patent_energy_top50 arxiv_construction jp_construction}"
SEEDS="${PNODE_SEEDS:-42 0 1 123 999}"
EPOCHS="${PNODE_EPOCHS:-200}"
METHODS="${METHODS:-prescient mioflow}"

mkdir -p logs_x5_baselines

idx=0
for M in $METHODS; do
  for D in $DOMAINS; do
    for S in $(echo "$SEEDS" | tr ',' ' '); do
      idx=$((idx+1))
      LOG="logs_x5_baselines/${M}_${D}_seed${S}.log"
      echo "[$idx] $M  domain=$D seed=$S  →  $LOG"
      PNODE_DOMAIN_TARGET="$D" PNODE_SEED="$S" PNODE_EPOCHS="$EPOCHS" \
        python pnode_patent_runner/baseline_trajectory_methods.py --method "$M" \
        > "$LOG" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "  [FAIL rc=$rc] tail of log:"
        tail -20 "$LOG"
      fi
    done
  done
done
echo "Done. Trajectory-baseline runs attempted: $idx"
