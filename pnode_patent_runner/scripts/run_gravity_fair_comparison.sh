#!/bin/bash
# 3-seed fair comparison: gravity vs baselines (all with GRU history fusion)
set -e
cd "$(dirname "$0")/../.."

DOMAIN="author_topic"
EPOCHS=30
SEEDS=(42 43 44)
OUT_DIR="pnode_patent_runner/results/gravity_fair_comparison"
mkdir -p "$OUT_DIR"

METHODS="static,rnn,neural_ode,neural_ode_gru,pnode,gravity"

for SEED in "${SEEDS[@]}"; do
    echo "=== seed=$SEED ==="
    python -m pnode_patent_runner.run_benchmark_comparison \
        --data-domain "$DOMAIN" \
        --methods "$METHODS" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --pnode-history-len 4 \
        --pnode-ode-method rk4 \
        --pnode-ode-n-steps 4 \
        --output-json "$OUT_DIR/seed${SEED}.json" \
        2>&1 | tee "$OUT_DIR/seed${SEED}.log"
done

echo ""
echo "=== Aggregated Results ==="
python - <<'PYEOF'
import json, glob, statistics
from collections import defaultdict

out_dir = "pnode_patent_runner/results/gravity_fair_comparison"
files = sorted(glob.glob(f"{out_dir}/seed*.json"))
if not files:
    print("No result JSON files found.")
    exit()

auc_by_method = defaultdict(list)
ap_by_method  = defaultdict(list)

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for method, metrics in data.items():
        auc = metrics.get("best_val_auc") or metrics.get("val_auc")
        ap  = metrics.get("final_val_ap") or metrics.get("val_ap")
        if auc: auc_by_method[method].append(auc)
        if ap:  ap_by_method[method].append(ap)

print(f"{'Method':<22} {'AUC mean':>10} {'AUC std':>9} {'AP mean':>10} {'AP std':>9} {'N':>3}")
print("-" * 70)
order = ["Static", "RNN+VGAE", "NeuralODE", "NeuralODE+GRU", "P-NODE", "Gravity-VGAE"]
for m in order:
    aucs = auc_by_method.get(m, [])
    aps  = ap_by_method.get(m, [])
    if not aucs:
        continue
    a_mean = statistics.mean(aucs)
    a_std  = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    p_mean = statistics.mean(aps) if aps else float('nan')
    p_std  = statistics.stdev(aps) if len(aps) > 1 else 0.0
    print(f"{m:<22} {a_mean:10.4f} {a_std:9.4f} {p_mean:10.4f} {p_std:9.4f} {len(aucs):3d}")
PYEOF
