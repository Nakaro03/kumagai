#!/usr/bin/env bash
# ============================================================================
# 仮説検証実験: 「PNODEは長期トレンド予測において既存手法より優れる」
#
# 検証計画: docs/HYPOTHESIS_LONG_HORIZON_VERIFICATION.md
# 統計手順: docs/STATS_PREREGISTRATION.md
#           docs/LONG_HORIZON_PREREGISTRATION.md
#
# Phase 1: 5シード × 4手法 × ホライズン k={1,2,3} のベンチマーク実行
# Phase 2: 集約・検定（H1-H5）
#
# 使い方:
#   bash pnode_patent_runner/scripts/run_hypothesis_long_horizon.sh
#
# 環境変数でオーバーライド可能:
#   SEEDS="42 43 44"  EPOCHS=10  DOMAIN=patent  ...
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------- 設定（環境変数でオーバーライド可能） ----------
SEEDS="${SEEDS:-42 43 44 45 46}"
DATA="${DATA:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
DOMAIN="${DOMAIN:-author_topic}"
TOPIC_COL="${TOPIC_COL:-topic}"
Y0="${Y0:-2022}"
Y1="${Y1:-2025}"
HOLDOUT="${HOLDOUT:-2025}"
MIN_P="${MIN_P:-5}"
EPOCHS="${EPOCHS:-20}"
HIDDEN="${HIDDEN:-128}"
LATENT="${LATENT:-2}"
METHODS="${METHODS:-static,rnn,neural_ode,pnode}"
HORIZON_GAPS="${HORIZON_GAPS:-1,2,3}"
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/hypothesis_long_horizon}"

mkdir -p "$OUT_DIR"

echo "============================================================"
echo " 仮説検証: PNODE 長期 future-link 優位"
echo "============================================================"
echo " domain=$DOMAIN  data=$DATA"
echo " year_range=$Y0–$Y1  holdout=$HOLDOUT"
echo " seeds=($SEEDS)  epochs=$EPOCHS"
echo " methods=$METHODS"
echo " horizon_gaps=$HORIZON_GAPS"
echo " output=$OUT_DIR"
echo "============================================================"
echo ""

# ============================================================
# Phase 1: 各シードでベンチマーク実行
# ============================================================
echo ">>> Phase 1: ベンチマーク実行"

for SEED in $SEEDS; do
  OUT_JSON="$OUT_DIR/benchmark_${DOMAIN}_seed${SEED}.json"
  if [ -f "$OUT_JSON" ]; then
    echo "[SKIP] $OUT_JSON は既に存在 (再実行は rm して再度実行)"
    continue
  fi
  echo ""
  echo "--- seed=$SEED ---"
  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain "$DOMAIN" \
    --data "$DATA" \
    --topic-column "$TOPIC_COL" \
    --year-range "$Y0" "$Y1" \
    --holdout-test-year "$HOLDOUT" \
    --min-patents "$MIN_P" \
    --epochs "$EPOCHS" \
    --hidden-dim "$HIDDEN" \
    --latent-dim "$LATENT" \
    --seed "$SEED" \
    --methods "$METHODS" \
    --loss-aux-warmup-epochs 0 \
    --pnode-potential-feature mlp \
    --pnode-history-len 4 \
    --pnode-hist-fuse-mode gru \
    --eval-horizon-gaps "$HORIZON_GAPS" \
    --save-checkpoint-dir "$OUT_DIR/ckpt" \
    --output-json "$OUT_JSON"
  echo "[DONE] $OUT_JSON"
done

echo ""
echo ">>> Phase 1 完了"
echo ""

# ============================================================
# Phase 2: 集約・仮説検定
# ============================================================
echo "============================================================"
echo " Phase 2: 集約・仮説検定"
echo "============================================================"

GLOB_PAT="$OUT_DIR/benchmark_${DOMAIN}_seed*.json"

# --- H1: PNODE vs Neural ODE (k=2, holdout AUC) ---
echo ""
echo "=========================================="
echo " H1: PNODE vs Neural ODE (k=2, AUC)"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 2 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode

# --- H1 補足: k=1 参考値 ---
echo ""
echo "=========================================="
echo " 参考: k=1 AUC（短期）"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 1 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode

# --- H1 補足: k=3 参考値 ---
echo ""
echo "=========================================="
echo " 参考: k=3 AUC（最長期）"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 3 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode

# --- H1 + AP ---
echo ""
echo "=========================================="
echo " 副指標: k=2 AP"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 2 \
  --horizon-field ap \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode

# --- H3: PNODE vs Neural ODE (通常 holdout AUC, ホライズンなし) ---
echo ""
echo "=========================================="
echo " 基本: holdout AUC（従来指標）"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --markdown \
  --paired-pnode-vs neural_ode

# --- PNODE vs RNN ---
echo ""
echo "=========================================="
echo " 参考: PNODE vs RNN (k=2, AUC)"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 2 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs rnn

# --- PNODE vs Static ---
echo ""
echo "=========================================="
echo " 参考: PNODE vs Static (k=2, AUC)"
echo "=========================================="
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "$GLOB_PAT" \
  --horizon-gap 2 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs static

# --- H2 / H4 / H5: 劣化率・RNN逆転・Staticマージン拡大 ---
echo ""
echo "=========================================="
echo " H2 / H4 / H5: ペア差の二次処理検定"
echo "=========================================="
python -m pnode_patent_runner.evaluate_hypothesis_h2_h5 \
  --glob "$GLOB_PAT" \
  --horizon-split final \
  --markdown

echo ""
echo "============================================================"
echo " 全 Phase 完了"
echo "============================================================"
echo ""
echo "判定フローチャート:"
echo "  docs/HYPOTHESIS_LONG_HORIZON_VERIFICATION.md §7 を参照し、"
echo "  H1–H5 の結果に基づいて論文の主張を確定する"
echo ""
