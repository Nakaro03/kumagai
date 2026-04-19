#!/usr/bin/env bash
# 対称 HPO（B）: 各手法で同じ --n-trials・同じ目的・同じデータ条件で Optuna を回し、
# --optuna-best-json-map で run_benchmark_comparison に流し込む。
#
# 使い方（kumagai ルート）:
#   chmod +x pnode_patent_runner/scripts/run_symmetric_hpo_benchmark.example.sh
#   ./pnode_patent_runner/scripts/run_symmetric_hpo_benchmark.example.sh
#
# 環境変数で上書き可能:
#   N_TRIALS=30 EPOCHS=10 PATENT_CSV=data/processed/topic_info3.csv \
#   HOLDOUT_TEST_YEAR=2020 ./pnode_patent_runner/scripts/run_symmetric_hpo_benchmark.example.sh
#
# スモーク（最短・パイプライン確認）:
#   SMOKE=1 ./pnode_patent_runner/scripts/run_symmetric_hpo_benchmark.example.sh
#
# 速いプロトタイプ（対称 HPO の流れを保ちつつ軽量化。本番の主表用ではない）:
#   PROTOTYPE=1 ./pnode_patent_runner/scripts/run_symmetric_hpo_benchmark.example.sh
#   # 上書き例: PROTOTYPE=1 N_TRIALS=8 EPOCHS=4 NO_TUNE_HIDDEN=1 ...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --- 編集用（既定は特許・topic_info3 が data/processed にある前提）---
N_TRIALS="${N_TRIALS:-30}"
EPOCHS="${N_TRIALS_EPOCHS:-${EPOCHS:-10}}"
BENCH_EPOCHS="${BENCH_EPOCHS:-$EPOCHS}"
SEED="${SEED:-42}"
COPE_LINK_SCORE="${COPE_LINK_SCORE:-distance}"
MIN_PATENTS="${MIN_PATENTS:-2}"
PATENT_CSV="${PATENT_CSV:-data/processed/topic_info3.csv}"
YEAR_START="${YEAR_START:-2010}"
YEAR_END="${YEAR_END:-2020}"
# 空ならホールドアウト無し。例: 2020
HOLDOUT_TEST_YEAR="${HOLDOUT_TEST_YEAR:-}"
# Optuna: default | minimal | wide（minimal で探索が軽くなりがち）
OPTUNA_SPACE="${OPTUNA_SPACE:-default}"
# 1 で hidden_dim 固定（--hidden-dim）、探索次元を削減
NO_TUNE_HIDDEN="${NO_TUNE_HIDDEN:-0}"

METHODS=(cope static rnn neural_ode pnode)

if [[ "${SMOKE:-0}" == "1" ]]; then
  N_TRIALS=1
  EPOCHS=2
  BENCH_EPOCHS=2
  echo "[SMOKE] N_TRIALS=$N_TRIALS EPOCHS=$EPOCHS BENCH_EPOCHS=$BENCH_EPOCHS"
elif [[ "${PROTOTYPE:-0}" == "1" ]]; then
  N_TRIALS=5
  EPOCHS=3
  BENCH_EPOCHS=5
  if [[ "$OPTUNA_SPACE" == "default" ]]; then
    OPTUNA_SPACE="minimal"
  fi
  echo "[PROTOTYPE] N_TRIALS=$N_TRIALS EPOCHS=$EPOCHS BENCH_EPOCHS=$BENCH_EPOCHS space=$OPTUNA_SPACE"
fi

if [[ ! -f "$PATENT_CSV" ]]; then
  echo "CSV が見つかりません: $PATENT_CSV（PATENT_CSV を指定）" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-symmetric_patent_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="pnode_patent_runner/outputs/optuna/${RUN_ID}"
mkdir -p "$OUT_DIR"

OPTUNA_DB="${OPTUNA_DB:-pnode_patent_runner/outputs/optuna/${RUN_ID}.db}"
STORAGE="sqlite:///${REPO_ROOT}/${OPTUNA_DB}"

echo "=== 対称 HPO: methods=${METHODS[*]} n_trials=$N_TRIALS epochs=$EPOCHS space=$OPTUNA_SPACE ==="
echo "OUT_DIR=$OUT_DIR STORAGE=$STORAGE"

optuna_args=(
  --data-domain patent
  --data "$PATENT_CSV"
  --year-range "$YEAR_START" "$YEAR_END"
  --min-patents "$MIN_PATENTS"
  --n-trials "$N_TRIALS"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --cope-link-score "$COPE_LINK_SCORE"
  --storage "$STORAGE"
  --space "$OPTUNA_SPACE"
)
if [[ "$NO_TUNE_HIDDEN" == "1" ]]; then
  optuna_args+=(--no-tune-hidden)
fi
if [[ -n "$HOLDOUT_TEST_YEAR" ]]; then
  optuna_args+=(--holdout-test-year "$HOLDOUT_TEST_YEAR")
fi

for m in "${METHODS[@]}"; do
  echo "--- Optuna method=$m ---"
  python -m pnode_patent_runner.run_optuna_unified_vgae \
    --method "$m" \
    "${optuna_args[@]}" \
    --output-json "${OUT_DIR}/best_params_${m}.json"
done

MAP_JSON="${OUT_DIR}/optuna_paths_by_method.json"
export OUT_DIR MAP_JSON
python3 -c "
import json, os
out = os.environ['OUT_DIR']
path = os.environ['MAP_JSON']
methods = ['cope', 'static', 'rnn', 'neural_ode', 'pnode']
d = {m: f'{out}/best_params_{m}.json' for m in methods}
with open(path, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
"
echo "Wrote map: $MAP_JSON"

BENCH_OUT="${BENCH_OUT:-pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_symmetric_${RUN_ID}_seed${SEED}.json}"

bench_args=(
  --data-domain patent
  --data "$PATENT_CSV"
  --year-range "$YEAR_START" "$YEAR_END"
  --min-patents "$MIN_PATENTS"
  --epochs "$BENCH_EPOCHS"
  --seed "$SEED"
  --methods all
  --cope-link-score "$COPE_LINK_SCORE"
  --optuna-best-json-map "$MAP_JSON"
  --output-json "$BENCH_OUT"
)
if [[ -n "$HOLDOUT_TEST_YEAR" ]]; then
  bench_args+=(--holdout-test-year "$HOLDOUT_TEST_YEAR")
fi

echo "=== run_benchmark_comparison（対称 HPO 適用）==="
python -m pnode_patent_runner.run_benchmark_comparison "${bench_args[@]}"

echo "Done. Optuna JSON: $OUT_DIR/best_params_*.json"
echo "Benchmark JSON: $BENCH_OUT"
