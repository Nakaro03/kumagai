#!/usr/bin/env bash
# C: P-NODE は Φ(z,year) 勾配流、Neural ODE は年埋め込み付きベクトル場（同一 --time-dependent-potential）。
# K=1（履歴融合なし）。Static / RNN は従来どおり。
#
# リポジトリ kumagai ルートから:
#   chmod +x pnode_patent_runner/scripts/run_benchmark_C_author_topic_time_dep.sh
#   ./pnode_patent_runner/scripts/run_benchmark_C_author_topic_time_dep.sh
#
# 上書き例:
#   EPOCHS=30 SEED=43 OUT=/tmp/c.json ./pnode_patent_runner/scripts/run_benchmark_C_author_topic_time_dep.sh

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
OUT="${OUT:-pnode_patent_runner/outputs/abc_runs/benchmark_C_author_topic_timeDep_K1_seed${SEED}.json}"

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
  --time-dependent-potential \
  --loss-aux-warmup-epochs 0 \
  --pnode-history-len 1 \
  "${extra[@]}" \
  --output-json "${OUT}"

echo "Wrote: ${OUT}"
