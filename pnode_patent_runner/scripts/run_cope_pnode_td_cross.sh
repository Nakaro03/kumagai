#!/usr/bin/env bash
# CoPE と P-NODE について --time-dependent-potential の組み合わせ（最大 4 パターン）を
# run_benchmark_comparison で順に実行する。
#
# パターン
#   (1) 両方 TD:     --methods cope,pnode --time-dependent-potential
#   (2) 両方 非TD:   --methods cope,pnode（フラグなし）
#   (3) CoPE のみ TD / P-NODE は非TD: 2 回に分割（CLI は全体で TD が1つしか取れないため）
#   (4) CoPE は非TD / P-NODE のみ TD: 2 回に分割
#
# 使い方（リポジトリ kumagai のルートで）:
#   chmod +x pnode_patent_runner/scripts/run_cope_pnode_td_cross.sh
#   ./pnode_patent_runner/scripts/run_cope_pnode_td_cross.sh
#
# 例（著者–論文・ホールドアウト 2026）:
#   DATA_DOMAIN=arxiv HOLDOUT_TEST_YEAR=2026 \
#   ARXIV_YEAR_START=2020 ARXIV_YEAR_END=2026 \
#   ./pnode_patent_runner/scripts/run_cope_pnode_td_cross.sh
#
# ドライラン（コマンド表示のみ）:
#   DRY_RUN=1 ./pnode_patent_runner/scripts/run_cope_pnode_td_cross.sh
#
# 環境変数（主なもの）:
#   DATA_DOMAIN=patent|arxiv|author_topic
#   EPOCHS SEED COPE_LINK_SCORE MIN_PATENTS
#   PATENT_CSV PATENT_YEAR_START PATENT_YEAR_END
#   ARXIV_CSV ARXIV_YEAR_MIN ARXIV_YEAR_MAX ARXIV_YEAR_START ARXIV_YEAR_END
#   HOLDOUT_TEST_YEAR（省略可。arxiv で最古年がホールドアウト年だと失敗するので注意）
#   OPTUNA_MAP_JSON（省略可。指定時は --optuna-best-json-map）
#   OUT_DIR（既定: pnode_patent_runner/outputs/cope_benchmark/cope_pnode_td_cross/<domain>）
#   RUN_PATTERN（1|2|3|4|all  既定 all）
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DOMAIN="${DATA_DOMAIN:-patent}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
COPE_LINK_SCORE="${COPE_LINK_SCORE:-distance}"
MIN_PATENTS="${MIN_PATENTS:-5}"
DRY_RUN="${DRY_RUN:-0}"
RUN_PATTERN="${RUN_PATTERN:-all}"

PATENT_CSV="${PATENT_CSV:-notebooks/work/dataset/topic_info3.csv}"
PATENT_YEAR_START="${PATENT_YEAR_START:-2010}"
PATENT_YEAR_END="${PATENT_YEAR_END:-2020}"

ARXIV_CSV="${ARXIV_CSV:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
ARXIV_YEAR_MIN="${ARXIV_YEAR_MIN:-2020}"
ARXIV_YEAR_MAX="${ARXIV_YEAR_MAX:-2026}"
ARXIV_YEAR_START="${ARXIV_YEAR_START:-}"
ARXIV_YEAR_END="${ARXIV_YEAR_END:-}"

HOLDOUT_TEST_YEAR="${HOLDOUT_TEST_YEAR:-}"
TOPIC_COLUMN="${TOPIC_COLUMN:-topic}"

OPTUNA_MAP_JSON="${OPTUNA_MAP_JSON:-}"

OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/cope_benchmark/cope_pnode_td_cross/${DATA_DOMAIN}}"
mkdir -p "$OUT_DIR"

common_py_args=(
  python -m pnode_patent_runner.run_benchmark_comparison
  --data-domain "$DATA_DOMAIN"
  --epochs "$EPOCHS"
  --seed "$SEED"
  --cope-link-score "$COPE_LINK_SCORE"
  --min-patents "$MIN_PATENTS"
)

if [[ -n "$OPTUNA_MAP_JSON" ]]; then
  common_py_args+=(--optuna-best-json-map "$OPTUNA_MAP_JSON")
fi

if [[ -n "$HOLDOUT_TEST_YEAR" ]]; then
  common_py_args+=(--holdout-test-year "$HOLDOUT_TEST_YEAR")
fi

data_args=()
case "$DATA_DOMAIN" in
  patent)
    data_args+=(--data "$PATENT_CSV" --year-range "$PATENT_YEAR_START" "$PATENT_YEAR_END")
    ;;
  arxiv|author_topic)
    data_args+=(--data "$ARXIV_CSV" --arxiv-year-min "$ARXIV_YEAR_MIN" --arxiv-year-max "$ARXIV_YEAR_MAX")
    if [[ -n "$ARXIV_YEAR_START" && -n "$ARXIV_YEAR_END" ]]; then
      data_args+=(--year-range "$ARXIV_YEAR_START" "$ARXIV_YEAR_END")
    fi
    if [[ "$DATA_DOMAIN" == "author_topic" ]]; then
      data_args+=(--topic-column "$TOPIC_COLUMN")
    fi
    ;;
  *)
    echo "不明な DATA_DOMAIN: $DATA_DOMAIN（patent|arxiv|author_topic）" >&2
    exit 1
    ;;
esac

run_one() {
  local label=$1
  shift
  local out_json=$1
  shift
  local -a cmd=(
    "${common_py_args[@]}"
    "${data_args[@]}"
    --methods "$1"
    --output-json "$out_json"
  )
  shift
  while [[ $# -gt 0 ]]; do
    cmd+=("$1")
    shift
  done
  echo ""
  echo "=== ${label} -> ${out_json} ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    return 0
  fi
  "${cmd[@]}"
}

should_run() {
  local p=$1
  [[ "$RUN_PATTERN" == "all" || "$RUN_PATTERN" == "$p" ]]
}

# --- (1) 両方 TD ---
if should_run 1; then
  run_one \
    "P1 both TD (cope+pnode)" \
    "${OUT_DIR}/p1_both_td_seed${SEED}.json" \
    "cope,pnode" \
    --time-dependent-potential
fi

# --- (2) 両方 非TD ---
if should_run 2; then
  run_one \
    "P2 both non-TD (cope+pnode)" \
    "${OUT_DIR}/p2_both_notd_seed${SEED}.json" \
    "cope,pnode"
fi

# --- (3) CoPE のみ TD / P-NODE 非TD ---
if should_run 3; then
  run_one \
    "P3a CoPE TD only" \
    "${OUT_DIR}/p3_cope_td_pnode_notd_cope_seed${SEED}.json" \
    "cope" \
    --time-dependent-potential
  run_one \
    "P3b P-NODE non-TD (pair for P3)" \
    "${OUT_DIR}/p3_cope_td_pnode_notd_pnode_seed${SEED}.json" \
    "pnode"
fi

# --- (4) CoPE 非TD / P-NODE のみ TD ---
if should_run 4; then
  run_one \
    "P4a CoPE non-TD (pair for P4)" \
    "${OUT_DIR}/p4_cope_notd_pnode_td_cope_seed${SEED}.json" \
    "cope"
  run_one \
    "P4b P-NODE TD only" \
    "${OUT_DIR}/p4_cope_notd_pnode_td_pnode_seed${SEED}.json" \
    "pnode" \
    --time-dependent-potential
fi

echo ""
echo "Done. Outputs under ${OUT_DIR}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "(dry-run: 実際の学習は行っていません)"
fi
