#!/usr/bin/env bash
# 特許・著者–論文（arxiv）・著者–トピック（author_topic）を同一条件でベンチマークし、
# cope_benchmark/benchmark_<domain>_seed<seed>.json を更新する。
# リポジトリルート（kumagai）で実行すること。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OPTUNA_JSON="pnode_patent_runner/outputs/optuna/best_params_unified_vgae_cope.json"
EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain patent \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --methods all \
  --optuna-best-json "$OPTUNA_JSON"

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain arxiv \
  --min-patents 5 \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --methods all \
  --optuna-best-json "$OPTUNA_JSON"

python -m pnode_patent_runner.run_benchmark_comparison \
  --data-domain author_topic \
  --min-patents 5 \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --methods all \
  --optuna-best-json "$OPTUNA_JSON"
