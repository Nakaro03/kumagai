#!/usr/bin/env bash
# X5 ablation sweep (Pattern C from X5_DESIGN.md §4).
#
# Ablations:
#   A0 — Full X5 (LOTO + 4-term composite)                         [reference]
#   A1 — no LOTO  (PNODE_LOTO=0)
#   A2 — no L_phys   (PNODE_LAM_PHYS=0)
#   A3 — no L_geom   (PNODE_LAM_GEOM=0)
#   A4 — no L_smooth (PNODE_LAM_SMOOTH=0)
#   A5 — Fourier K=0 (raw scalar t)             via PNODE_FOURIER_K=0
#   A6 — LOTO only (phys=geom=smooth=0)
#
# Per ablation: 3 domain × 5 seed = 15 runs.
# Total: 7 × 15 = 105 runs.

set -u
cd /home/nakamuraroi/kumagai
mkdir -p logs_x5_ablation

DOMAINS="${DOMAINS:-patent_energy_top50 arxiv_construction jp_construction}"
SEEDS="${PNODE_SEEDS:-42 0 1 123 999}"
EPOCHS="${PNODE_EPOCHS:-200}"
WARMUP="${PNODE_WARMUP:-40}"

# ablation_id : env_overrides : suffix
declare -A SETUPS=(
  [A0]=""
  [A1]="PNODE_LOTO=0"
  [A2]="PNODE_LAM_PHYS=0"
  [A3]="PNODE_LAM_GEOM=0"
  [A4]="PNODE_LAM_SMOOTH=0"
  [A5]="PNODE_FOURIER_K=0"
  [A6]="PNODE_LAM_PHYS=0 PNODE_LAM_GEOM=0 PNODE_LAM_SMOOTH=0"
)

ABL_ORDER="${ABL_ORDER:-A0 A1 A2 A3 A4 A5 A6}"

idx=0
for ABL in $ABL_ORDER; do
  ENV_OVR="${SETUPS[$ABL]}"
  export PNODE_ABL="$ABL"
  for D in $DOMAINS; do
    for S in $(echo "$SEEDS" | tr ',' ' '); do
      idx=$((idx+1))
      LOG="logs_x5_ablation/x5_${ABL}_${D}_seed${S}.log"
      echo "[$idx] $ABL  domain=$D seed=$S  (overrides: $ENV_OVR)  →  $LOG"
      env $ENV_OVR \
        PNODE_DOMAIN_TARGET="$D" PNODE_SEED="$S" \
        PNODE_EPOCHS="$EPOCHS" PNODE_WARMUP="$WARMUP" \
        python pnode_patent_runner/run_pisde_x5.py > "$LOG" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "  [FAIL rc=$rc] tail of log:"
        tail -20 "$LOG"
      fi
    done
  done
done
echo "Done. Ablation runs attempted: $idx"
