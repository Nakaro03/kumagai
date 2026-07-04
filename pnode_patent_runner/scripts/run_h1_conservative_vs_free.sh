#!/usr/bin/env bash
# =============================================================================
# H1 検証: 保存場 (P-NODE) vs 自由場 (Neural ODE) の多ステップ予測比較
#
# 仮説:
#   dz/dt = -α∇Φ(z)  (保存場) は k ステップ先 AUC の劣化が
#   dz/dt = f_θ(z)   (自由場) より小さい
#
# 実験仕様:
#   ドメイン : energy, agrifood, construction (中小規模 3 ドメイン)
#   シード   : 3 seeds (42, 43, 44)
#   手法     : static, rnn, neural_ode, pnode
#   ホライズン: k = 1, 2, 3
#   学習期間 : 2010–2019 (holdout=2020)
#
# 使い方:
#   bash pnode_patent_runner/scripts/run_h1_conservative_vs_free.sh
#
# オーバーライド例:
#   EPOCHS=20 SEEDS="42 43" bash .../run_h1_conservative_vs_free.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------- 設定 ----------
SEEDS="${SEEDS:-42 43 44}"
DOMAINS="${DOMAINS:-energy agrifood construction}"
YEAR_START="${YEAR_START:-2010}"
YEAR_END="${YEAR_END:-2020}"
HOLDOUT="${HOLDOUT:-2020}"
MIN_P="${MIN_P:-50}"
EPOCHS="${EPOCHS:-10}"
HIDDEN="${HIDDEN:-32}"
LATENT="${LATENT:-8}"
METHODS="${METHODS:-static,rnn,neural_ode,pnode}"
HORIZON_GAPS="${HORIZON_GAPS:-1,2,3}"
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/h1_conservative_vs_free}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " H1: 保存場 vs 自由場  多ステップ AUC 比較"
echo "============================================================"
echo " domains  : $DOMAINS"
echo " year     : $YEAR_START–$YEAR_END  holdout=$HOLDOUT"
echo " seeds    : $SEEDS"
echo " epochs   : $EPOCHS  hidden=$HIDDEN  latent=$LATENT"
echo " methods  : $METHODS"
echo " horizons : k = $HORIZON_GAPS"
echo " output   : $OUT_DIR"
echo "============================================================"
echo ""

FAIL_COUNT=0

for DOMAIN in $DOMAINS; do
  for SEED in $SEEDS; do
    OUT_JSON="$OUT_DIR/h1_${DOMAIN}_seed${SEED}.json"

    if [ -f "$OUT_JSON" ]; then
      echo "[SKIP] 既存: $OUT_JSON"
      continue
    fi

    echo ""
    echo "--- domain=$DOMAIN  seed=$SEED ---"

    python -m pnode_patent_runner.run_benchmark_comparison \
      --data-domain "$DOMAIN" \
      --year-range "$YEAR_START" "$YEAR_END" \
      --holdout-test-year "$HOLDOUT" \
      --min-patents "$MIN_P" \
      --epochs "$EPOCHS" \
      --hidden-dim "$HIDDEN" \
      --latent-dim "$LATENT" \
      --seed "$SEED" \
      --methods "$METHODS" \
      --pnode-potential-feature mlp \
      --pnode-ode-method rk4 \
      --pnode-ode-n-steps 4 \
      --eval-horizon-gaps "$HORIZON_GAPS" \
      --output-json "$OUT_JSON" \
      && echo "[OK] $OUT_JSON" \
      || { echo "[FAIL] domain=$DOMAIN seed=$SEED"; FAIL_COUNT=$((FAIL_COUNT+1)); }

  done
done

echo ""
echo "============================================================"
echo " Phase 1 完了  failures=$FAIL_COUNT"
echo "============================================================"
echo ""
echo "次のステップ:"
echo "  python pnode_patent_runner/analyze_h1_results.py --out-dir $OUT_DIR"
echo ""
