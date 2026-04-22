#!/usr/bin/env bash
# B: P-NODE / Neural ODE ともに直近 K=4 年の z を融合（--pnode-history-len 4）。
# 補助損失ウォームアップはオフ（A との差分を明確にする）。
#
# リポジトリ kumagai ルートから:
#   chmod +x pnode_patent_runner/scripts/run_benchmark_B_author_topic_K4.sh
#   ./pnode_patent_runner/scripts/run_benchmark_B_author_topic_K4.sh
#
# 上書き例:
#   EPOCHS=30 SEED=43 OUT=/tmp/b.json ./pnode_patent_runner/scripts/run_benchmark_B_author_topic_K4.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
Y0="${Y0:-2022}"
Y1="${Y1:-2025}"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
MIN_PAPERS="${MIN_PAPERS:-5}"
TOPIC_COLUMN="${TOPIC_COLUMN:-topic}"
OUT="${OUT:-pnode_patent_runner/outputs/abc_runs/benchmark_B_author_topic_K4_seed${SEED}.json}"

extra=()
if [[ -n "${EVAL_HORIZON_GAPS:-}" ]]; then
  extra+=(--eval-horizon-gaps "${EVAL_HORIZON_GAPS}")
fi

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain author_topic \
  --data "${DATA}" \
  --topic-column "${TOPIC_COLUMN}" \
  --year-range "${Y0}" "${Y1}" \
  --min-patents "${MIN_PAPERS}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --methods static,rnn,neural_ode,pnode \
  --loss-aux-warmup-epochs 0 \
  --pnode-history-len 4 \
  "${extra[@]}" \
  --output-json "${OUT}"

echo "Wrote: ${OUT}"
