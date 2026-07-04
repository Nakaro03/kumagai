#!/usr/bin/env bash
# 著者–トピックで Dual-Force を学習し、P-NODE (B+D) の既存 JSON と比較結果を保存する。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-20}"
OUT="${OUT:-pnode_patent_runner/outputs/dual_force_compare/author_topic_vs_pnode_seed${SEED}.json}"

python -m pnode_patent_runner.run_dual_force_vs_pnode_author_topic \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --output-json "$OUT"

echo "Saved: $OUT"
