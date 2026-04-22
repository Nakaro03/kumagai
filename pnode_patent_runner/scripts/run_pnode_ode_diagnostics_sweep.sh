#!/usr/bin/env bash
# 同一 SEED / EPOCHS で ODE 積分法を切り替え、出力 JSON の val_auc / last_epoch_train_breakdown.grad_phi_l2 等を比較する例。
# 使い方:
#   export DATA=/path/to/data.csv
#   export DATA_DOMAIN=author_topic   # または patent / arxiv_paper
#   # 任意: YEAR_RANGE="2015 2019" SEED=42 EPOCHS=5 METHODS=pnode_explicit
#   bash pnode_patent_runner/scripts/run_pnode_ode_diagnostics_sweep.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
: "${DATA:?set DATA to your CSV path}"
DATA_DOMAIN="${DATA_DOMAIN:-author_topic}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-5}"
METHODS="${METHODS:-pnode_explicit}"
if [ -n "${YEAR_RANGE:-}" ]; then
  read -r -a _yr_split <<< "$YEAR_RANGE"
  YR=(--year-range "${_yr_split[0]}" "${_yr_split[1]}")
else
  YR=(--year-range 2010 2020)
fi

run_one() {
  local name="$1"
  local ode_method="$2"
  local nstep="$3"
  local out="${REPO_ROOT}/pnode_patent_runner/outputs/cope_benchmark/ode_diag_${name}_seed${SEED}.json"
  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain "$DATA_DOMAIN" \
    --data "$DATA" \
    "${YR[@]}" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --methods "$METHODS" \
    --pnode-ode-method "$ode_method" \
    --pnode-ode-n-steps "$nstep" \
    --output-json "$out"
  echo "Wrote $out"
}

run_one "dopri5" "dopri5" 4
run_one "rk4_8" "rk4" 8
echo "Compare: best_val_auc, last_epoch_train_breakdown.grad_phi_l2, train_components_per_epoch.grad_phi_l2 in the two JSON files."
