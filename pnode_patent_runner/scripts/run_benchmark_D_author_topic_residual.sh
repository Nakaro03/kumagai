#!/usr/bin/env bash
# D: P-NODE+Res（勾配流 + 残差場 h(z)）を、Static / RNN / Neural ODE / P-NODE と同一条件で比較。
#
#   chmod +x pnode_patent_runner/scripts/run_benchmark_D_author_topic_residual.sh
#   ./pnode_patent_runner/scripts/run_benchmark_D_author_topic_residual.sh
#
# Static+RNN+Res のみに絞る例:
#   METHODS=static,rnn,pnode_residual OUT=/tmp/narrow.json \\
#     ./pnode_patent_runner/scripts/run_benchmark_D_author_topic_residual.sh

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
METHODS="${METHODS:-static,rnn,neural_ode,pnode,pnode_residual}"
OUT="${OUT:-pnode_patent_runner/outputs/abc_runs/benchmark_D_vs_baselines_seed${SEED}.json}"

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain author_topic \
  --data "${DATA}" \
  --topic-column "${TOPIC_COLUMN}" \
  --year-range "${Y0}" "${Y1}" \
  --min-patents "${MIN_PAPERS}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --methods "${METHODS}" \
  --loss-aux-warmup-epochs 0 \
  --pnode-history-len 1 \
  --output-json "${OUT}"

echo "Wrote: ${OUT}"
