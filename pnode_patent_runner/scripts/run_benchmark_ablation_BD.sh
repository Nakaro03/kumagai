#!/usr/bin/env bash
# B / D を個別に検証し、既存ベースライン（static, rnn, neural_ode）および P-NODE 系と同条件で比較する。
#
# - legacy: 変更前に近い P-NODE 設定（Φ=rff・固定 B・履歴 K=1）
# - B_only:  D を無効化（K=1 のみ）。Φ を mlp に変更した効果のみを見る
# - D_only:  B を legacy のまま。履歴 K>1 + GRU 融合の効果のみを見る
# - BD: B+D 併用（Φ=mlp、履歴 K=4 + GRU 融合）
#
# static / rnn / neural_ode は各 run で同一ハイパラのため指標は一致する想定。
#
# 使い方（リポジトリルートで）:
#   bash pnode_patent_runner/scripts/run_benchmark_ablation_BD.sh
#   DATA=data/processed/arxiv_cs_embedded_2020-2026_full.csv SEED=42 EPOCHS=20 bash ...
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
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/ablation_BD}"
# 比較する手法（カンマ区切り）。P-NODE 用フラグは pnode / pnode_residual / neural_ode のみ影響
METHODS="${METHODS:-static,rnn,neural_ode,pnode,pnode_residual}"

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

# 1) 既存設定に近い P-NODE（レガシー相当: rff + 固定 B, K=1）
run "legacy_rff_K1_linear" \
  --pnode-potential-feature rff \
  --pnode-rff-frozen-basis \
  --pnode-history-len 1 \
  --pnode-hist-fuse-mode linear

# 2) B のみ: Φ=mlp、履歴は 1（融合は使われない）
run "B_mlp_phi_K1" \
  --pnode-potential-feature mlp \
  --pnode-history-len 1 \
  --pnode-hist-fuse-mode gru

# 3) D のみ: Φ はレガシー、履歴 K=4 + GRU 融合
run "D_gru_K4_legacy_phi" \
  --pnode-potential-feature rff \
  --pnode-rff-frozen-basis \
  --pnode-history-len 4 \
  --pnode-hist-fuse-mode gru

# 4) B+D: Φ=mlp かつ履歴 K=4 + GRU（P-NODE 系は B・D 両方）
run "BD_mlp_gru_K4" \
  --pnode-potential-feature mlp \
  --pnode-history-len 4 \
  --pnode-hist-fuse-mode gru

echo "=== 横比較（final_val_auc）==="
python -m pnode_patent_runner.compare_benchmark_ablations \
  "$OUT_DIR/benchmark_legacy_rff_K1_linear_seed${SEED}.json" \
  "$OUT_DIR/benchmark_B_mlp_phi_K1_seed${SEED}.json" \
  "$OUT_DIR/benchmark_D_gru_K4_legacy_phi_seed${SEED}.json" \
  "$OUT_DIR/benchmark_BD_mlp_gru_K4_seed${SEED}.json" \
  --labels legacy_rff_K1 B_mlp_K1 D_gru_K4_legacy BD_mlp_gru_K4 \
  --markdown

echo ""
echo "AP も見る場合:"
echo "  python -m pnode_patent_runner.compare_benchmark_ablations \\"
echo "    $OUT_DIR/benchmark_legacy_rff_K1_linear_seed${SEED}.json \\"
echo "    $OUT_DIR/benchmark_B_mlp_phi_K1_seed${SEED}.json \\"
echo "    $OUT_DIR/benchmark_D_gru_K4_legacy_phi_seed${SEED}.json \\"
echo "    $OUT_DIR/benchmark_BD_mlp_gru_K4_seed${SEED}.json \\"
echo "    --labels legacy_rff_K1 B_mlp_K1 D_gru_K4_legacy BD_mlp_gru_K4 --metric final_val_ap --markdown"
