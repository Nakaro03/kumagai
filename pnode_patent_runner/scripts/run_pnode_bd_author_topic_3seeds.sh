#!/usr/bin/env bash
# 著者–トピックで P-NODE (B+D) vs baselines を複数 seed 実行（有意差用）。
# 各 seed 1 JSON → aggregate_benchmark_seeds + validity_report
#
#   bash pnode_patent_runner/scripts/run_pnode_bd_author_topic_3seeds.sh
#   EPOCHS=3 SEEDS="42 43 44" ...  # 手早い煙テスト
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/pnode_BD_author_topic_seeds}"
SEEDS="${SEEDS:-42 43 44}"
EPOCHS="${EPOCHS:-20}"

mkdir -p "$OUT_DIR"
for S in $SEEDS; do
  echo "=== SEED=$S ==="
  SEED="$S" OUT_DIR="$OUT_DIR" EPOCHS="$EPOCHS" \
    bash pnode_patent_runner/scripts/run_benchmark_pnode_BD_vgae_compare.sh
done

GLOB="${OUT_DIR}/benchmark_pnode_BD_vs_baselines_seed*.json"
AGG_MD="${OUT_DIR}/aggregate_pnodeBD_vs_baselines.md"
REPORT_MD="${OUT_DIR}/validity_significance.md"

echo "=== aggregate_benchmark_seeds → ${AGG_MD}"
python -m pnode_patent_runner.aggregate_benchmark_seeds --glob "$GLOB" --markdown \
  --paired-pnode-vs neural_ode | tee "$AGG_MD"

echo ""
echo "=== run_validity_report → ${REPORT_MD}"
python -m pnode_patent_runner.run_validity_report \
  --benchmark-glob "$GLOB" \
  --aggregate-seeds-glob "$GLOB" \
  --paired-pnode-vs neural_ode \
  -o "$REPORT_MD"

echo "Done. JSON: $GLOB"
echo "集約: $AGG_MD | レポート: $REPORT_MD"
