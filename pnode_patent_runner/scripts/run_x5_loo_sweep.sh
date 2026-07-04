#!/usr/bin/env bash
# X5 TRUE leave-one-out sweep.
#
# 3 domain × 1 holdout (middle timepoint) × 5 seed × 3 ablations = 45 runs
# Ablations:
#   full         X5 with anchor + Fourier + geom + smooth (no LOTO mask needed; held-out is real)
#   no_anchor    A2-equivalent under held-out
#   prescient    PRESCIENT-equivalent under held-out (no anchor, no Fourier, no geom, no smooth)
#
# At ~7 min/run × 45 = ~5.3 hours.

set -u
cd /home/nakamuraroi/kumagai
mkdir -p logs_x5_loo

# Pick a "middle" held-out timepoint per domain
declare -A HOLDOUT
HOLDOUT[patent_energy_top50]=5
HOLDOUT[arxiv_construction]=5
HOLDOUT[jp_construction]=5

DOMAINS="${DOMAINS:-patent_energy_top50 arxiv_construction jp_construction}"
SEEDS="${PNODE_SEEDS:-42 0 1 123 999}"
EPOCHS="${PNODE_EPOCHS:-200}"
ABLATIONS="${ABLATIONS:-full no_anchor prescient}"

idx=0
for ABL in $ABLATIONS; do
  for D in $DOMAINS; do
    HT="${HOLDOUT[$D]}"
    for S in $(echo "$SEEDS" | tr ',' ' '); do
      idx=$((idx+1))
      LOG="logs_x5_loo/loo_${ABL}_${D}_h${HT}_seed${S}.log"

      # Skip if already done
      DATA_NAME=$(python -c "from pnode_patent_runner.x5.config import DOMAIN_TABLE; print(DOMAIN_TABLE['${D}'][0])")
      EVAL="RESULTS_X5_LOO/${DATA_NAME}/h${HT}/${ABL}/seed_${S}/evaluation.json"
      if [[ -f "$EVAL" ]] && [[ "${FORCE:-0}" != "1" ]]; then
        echo "  [skip $idx] $ABL $D seed=$S — already exists"
        continue
      fi

      echo "[$idx] $ABL  $D h=$HT seed=$S  →  $LOG"
      PNODE_DOMAIN_TARGET="$D" PNODE_SEED="$S" PNODE_EPOCHS="$EPOCHS" \
        PNODE_HOLDOUT_T="$HT" PNODE_ABLATION="$ABL" \
        python pnode_patent_runner/run_pisde_x5_loo.py > "$LOG" 2>&1
      rc=$?
      if [[ $rc -ne 0 ]]; then
        echo "  [FAIL rc=$rc] tail of log:"
        tail -20 "$LOG"
      fi
    done
  done
done
echo "Done. LOO sweep runs attempted: $idx"
