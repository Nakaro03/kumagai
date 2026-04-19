#!/usr/bin/env bash
# 3 ドメイン × 3 HPO モード × 2（時間依存 Φ あり/なし）の因子実験を連続実行する。
# リポジトリ kumagai のルートで実行すること。
#
# 環境変数（例）:
#   DOMAINS="patent arxiv author_topic"
#   HPO_MODE=fixed| cope_only | all
#   TD=0|1   （1 のとき --time-dependent-potential）
#   SEED=42 EPOCHS=10
#   OPTUNA_COPE_JSON=pnode_patent_runner/outputs/optuna/best_params_unified_vgae_cope.json
#   OPTUNA_MAP_JSON=pnode_patent_runner/outputs/optuna/symmetric_patent_20260411_234808/optuna_paths_by_method.json
#   PATENT_CSV ARXIV_CSV / 年範囲はドメインごとに下の case で編集
#   HOLDOUT_TEST_YEAR=2020  … 指定時は --holdout-test-year（年範囲にその年を含めること）
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DOMAINS="${DOMAINS:-patent}"
HPO_MODE="${HPO_MODE:-fixed}"
TD="${TD:-0}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-10}"
COPE_LINK_SCORE="${COPE_LINK_SCORE:-distance}"
MIN_PAPERS="${MIN_PAPERS:-5}"
OUT_ROOT="${OUT_ROOT:-pnode_patent_runner/outputs/cope_benchmark/factorial}"
mkdir -p "$OUT_ROOT"

OPTUNA_COPE_JSON="${OPTUNA_COPE_JSON:-pnode_patent_runner/outputs/optuna/best_params_unified_vgae_cope.json}"
OPTUNA_MAP_JSON="${OPTUNA_MAP_JSON:-}"

PATENT_CSV="${PATENT_CSV:-notebooks/work/dataset/topic_info3.csv}"
PATENT_YEAR_START="${PATENT_YEAR_START:-2010}"
PATENT_YEAR_END="${PATENT_YEAR_END:-2020}"

ARXIV_CSV="${ARXIV_CSV:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
ARXIV_YEAR_MIN="${ARXIV_YEAR_MIN:-2020}"
ARXIV_YEAR_MAX="${ARXIV_YEAR_MAX:-2026}"
ARXIV_YEAR_START="${ARXIV_YEAR_START:-}"
ARXIV_YEAR_END="${ARXIV_YEAR_END:-}"

HOLDOUT_TEST_YEAR="${HOLDOUT_TEST_YEAR:-}"
holdout_args=()
if [[ -n "${HOLDOUT_TEST_YEAR:-}" ]]; then
  holdout_args=(--holdout-test-year "$HOLDOUT_TEST_YEAR")
fi

td_flag=()
if [[ "$TD" == "1" ]]; then
  td_flag=(--time-dependent-potential)
fi

hpo_args=()
hpo_slug="fixed"
case "$HPO_MODE" in
  fixed)
    hpo_slug="fixed"
    ;;
  cope_only)
    hpo_slug="cope_only"
    hpo_args=(--optuna-best-json "$OPTUNA_COPE_JSON")
    ;;
  all)
    hpo_slug="all"
    if [[ -z "$OPTUNA_MAP_JSON" ]]; then
      echo "HPO_MODE=all には OPTUNA_MAP_JSON を設定してください。" >&2
      exit 1
    fi
    hpo_args=(--optuna-best-json-map "$OPTUNA_MAP_JSON")
    ;;
  *)
    echo "不明な HPO_MODE: $HPO_MODE（fixed|cope_only|all）" >&2
    exit 1
    ;;
esac

for domain in $DOMAINS; do
  data_args=()
  extra_args=()
  case "$domain" in
    patent)
      data_args=(--data "$PATENT_CSV" --year-range "$PATENT_YEAR_START" "$PATENT_YEAR_END")
      ;;
    arxiv|author_topic)
      if [[ -n "${ARXIV_CSV:-}" && -f "$ARXIV_CSV" ]]; then
        data_args=(--data "$ARXIV_CSV")
      fi
      extra_args=(
        --min-patents "$MIN_PAPERS"
        --arxiv-year-min "$ARXIV_YEAR_MIN"
        --arxiv-year-max "$ARXIV_YEAR_MAX"
      )
      if [[ -n "$ARXIV_YEAR_START" && -n "$ARXIV_YEAR_END" ]]; then
        data_args+=(--year-range "$ARXIV_YEAR_START" "$ARXIV_YEAR_END")
      fi
      ;;
    *)
      echo "不明なドメイン: $domain" >&2
      exit 1
      ;;
  esac

  ho_suffix=""
  if [[ -n "${HOLDOUT_TEST_YEAR:-}" ]]; then
    ho_suffix="_holdout${HOLDOUT_TEST_YEAR}"
  fi
  out_json="${OUT_ROOT}/${domain}_hpo${hpo_slug}_td${TD}${ho_suffix}_seed${SEED}.json"

  echo "=== domain=$domain HPO=$hpo_slug TD=$TD holdout=${HOLDOUT_TEST_YEAR:-none} -> $out_json ==="
  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain "$domain" \
    "${data_args[@]}" \
    "${extra_args[@]}" \
    "${td_flag[@]}" \
    "${hpo_args[@]}" \
    "${holdout_args[@]}" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --methods all \
    --cope-link-score "$COPE_LINK_SCORE" \
    --output-json "$out_json"
done

echo "Done. Outputs under $OUT_ROOT"
