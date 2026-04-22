#!/usr/bin/env bash
# pnode（B+D）と baseline（static, rnn, neural_ode）の 4 手法だけを 1 本の JSON で比較。
# B+D: Φ=mlp, 履歴 K=4, GRU 融合。
#
# pnode の legacy / B / D の効きは `run_benchmark_pnode_ablation.sh` を参照。
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
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/pnode_BD_vgae_compare}"
METHODS="${METHODS:-static,rnn,neural_ode,pnode}"

mkdir -p "$OUT_DIR"

OUT_JSON="$OUT_DIR/benchmark_pnode_BD_vs_baselines_seed${SEED}.json"

echo "=== static / rnn / neural_ode / pnode(B+D) 4 手法比較 ==="
echo "methods=$METHODS"
echo "Output: $OUT_JSON"

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
  --pnode-potential-feature mlp \
  --pnode-history-len 4 \
  --pnode-hist-fuse-mode gru \
  --output-json "$OUT_JSON"

echo ""
echo "結果: results の key=static,rnn,neural_ode,pnode を比較。"
