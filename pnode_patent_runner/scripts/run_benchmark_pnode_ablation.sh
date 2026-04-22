#!/usr/bin/env bash
# pnode の B/D アブレーション（legacy / Bのみ / Dのみ / B+D）を 4 回実行し、
# 各回ともに baseline（static, rnn, neural_ode）を入れて同じ表で比較できるようにする。
# pnode 列だけが設定により変化する（baseline 列は実行間でブレる場合あり＝複数シード推奨）。
#
# - legacy_rff_K1:   Φ=rff 固定B, K=1
# - B_mlp_K1:        Φ=mlp, K=1（D オフ）
# - D_gru_K4_legacy: Φ=rff 固定B, K=4 + GRU
# - BD_mlp_gru_K4:   Φ=mlp, K=4 + GRU
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
DOMAIN="${DOMAIN:-author_topic}"
TOPIC_COL="${TOPIC_COL:-topic}"
Y0="${Y0:-2022}"
Y1="${Y1:-2025}"
MIN_P="${MIN_P:-5}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/ablation_pnode}"
METHODS="${METHODS:-static,rnn,neural_ode,pnode}"

mkdir -p "$OUT_DIR"

run() {
  local name="$1"
  shift
  local out="$OUT_DIR/benchmark_${name}_seed${SEED}.json"
  echo "=== [$name] -> $out ==="
  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain "$DOMAIN" \
    --data "$DATA" \
    --topic-column "$TOPIC_COL" \
    --year-range "$Y0" "$Y1" \
    --min-patents "$MIN_P" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --methods "$METHODS" \
    --loss-aux-warmup-epochs 0 \
    "$@" \
    --output-json "$out"
  echo ""
}

run "legacy_rff_K1_linear" \
  --pnode-potential-feature rff \
  --pnode-rff-frozen-basis \
  --pnode-history-len 1 \
  --pnode-hist-fuse-mode linear

run "B_mlp_phi_K1" \
  --pnode-potential-feature mlp \
  --pnode-history-len 1 \
  --pnode-hist-fuse-mode gru

run "D_gru_K4_legacy_phi" \
  --pnode-potential-feature rff \
  --pnode-rff-frozen-basis \
  --pnode-history-len 4 \
  --pnode-hist-fuse-mode gru

run "BD_mlp_gru_K4" \
  --pnode-potential-feature mlp \
  --pnode-history-len 4 \
  --pnode-hist-fuse-mode gru

echo "=== 横比較（final_val_auc）pnode 列がアブレーション ==="
python -m pnode_patent_runner.compare_benchmark_ablations \
  "$OUT_DIR/benchmark_legacy_rff_K1_linear_seed${SEED}.json" \
  "$OUT_DIR/benchmark_B_mlp_phi_K1_seed${SEED}.json" \
  "$OUT_DIR/benchmark_D_gru_K4_legacy_phi_seed${SEED}.json" \
  "$OUT_DIR/benchmark_BD_mlp_gru_K4_seed${SEED}.json" \
  --labels legacy_rff_K1 B_mlp_K1 D_gru_K4_legacy BD_mlp_gru_K4 \
  --markdown

echo ""
echo "AP: --metric final_val_ap --markdown を付けて同じ JSON 4 つを指定。"
