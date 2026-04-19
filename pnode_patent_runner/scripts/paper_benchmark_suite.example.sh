#!/usr/bin/env bash
# 論文用: run_benchmark_comparison を複数シード・複数ドメインで回す雛形。
# 使用前先に REPO_ROOT / CSV / 年範囲を環境に合わせて編集すること。
#
# 実行例（リポジトリ kumagai ルート）:
#   chmod +x pnode_patent_runner/scripts/paper_benchmark_suite.example.sh
#   ./pnode_patent_runner/scripts/paper_benchmark_suite.example.sh
#
# 著者–論文 / 著者–トピックも評価する例:
#   DOMAINS="patent arxiv author_topic" ARXIV_CSV=data/processed/arxiv_cs_embedded_2020-2026.csv \
#   ARXIV_YEAR_START=2020 ARXIV_YEAR_END=2024 ./pnode_patent_runner/scripts/paper_benchmark_suite.example.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --- 編集用変数 ---
EPOCHS="${EPOCHS:-10}"
COPE_LINK_SCORE="${COPE_LINK_SCORE:-distance}"
MIN_PAPERS="${MIN_PAPERS:-5}"
# 特許 CSV（存在するパスに変更）
PATENT_CSV="${PATENT_CSV:-notebooks/work/dataset/topic_info3.csv}"
# 著者–論文 / 著者–トピック用（例: ARXIV_CSV=data/processed/arxiv_cs_embedded_2020-2026_full.csv）
ARXIV_CSV="${ARXIV_CSV:-}"
# 前処理の年フィルタ（arxiv / author_topic）
ARXIV_YEAR_MIN="${ARXIV_YEAR_MIN:-2020}"
ARXIV_YEAR_MAX="${ARXIV_YEAR_MAX:-2026}"
# グラフに含める年（空なら CLI に year-range を付けない = データに任せる）
ARXIV_YEAR_START="${ARXIV_YEAR_START:-}"
ARXIV_YEAR_END="${ARXIV_YEAR_END:-}"

SEEDS=(42 43 44)
# ドメイン一覧: patent arxiv author_topic（CSV が無いドメインは DOMAINS から外す）
DOMAINS=(patent)

for domain in "${DOMAINS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    extra_args=()
    data_args=()
    case "$domain" in
      patent)
        data_args=(--data "$PATENT_CSV" --year-range 2010 2020)
        ;;
      arxiv|author_topic)
        if [[ -n "$ARXIV_CSV" ]]; then
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
    esac

    echo "=== domain=$domain seed=$seed ==="
    python -m pnode_patent_runner.run_benchmark_comparison \
      --data-domain "$domain" \
      "${data_args[@]}" \
      "${extra_args[@]}" \
      --epochs "$EPOCHS" \
      --seed "$seed" \
      --methods all \
      --cope-link-score "$COPE_LINK_SCORE"
  done
done

echo "Done. JSON under pnode_patent_runner/outputs/cope_benchmark/ (benchmark_<patent|arxiv|author_topic>_seed<seed>.json)"
