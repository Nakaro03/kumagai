#!/usr/bin/env bash
# Gate L: TAP-NODE(+burst) を construction/agrifood(maingroup相当データ) で
# 10 seed x holdout-test-year=2021 で評価する。docs/DUAL_FORCE_REDESIGN.md 参照。
# リポジトリ kumagai ルートで実行。
set -euo pipefail

SEEDS=(0 1 2 3 4 5 6 7 42 123)
DOMAINS=(construction agrifood)
EPOCHS="${EPOCHS:-10}"
SCALAR_LR="${SCALAR_LR:-0.05}"
OUT_DIR="pnode_patent_runner/outputs/tap_node_patent"
mkdir -p "$OUT_DIR"

for dom in "${DOMAINS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$OUT_DIR/tap_node_burst_${dom}_seed${seed}.json"
    if [ -f "$out" ]; then
      echo "skip (exists): $out"
      continue
    fi
    echo "=== $dom seed=$seed ==="
    python3 -m pnode_patent_runner.run_tap_node_patent_domain \
      --domain "$dom" --year-start 2017 --year-end 2021 --holdout-test-year 2021 \
      --epochs "$EPOCHS" --seed "$seed" --scalar-lr "$SCALAR_LR" \
      --output-json "$out" > "$OUT_DIR/log_${dom}_seed${seed}.txt" 2>&1
  done
done
echo "done"
