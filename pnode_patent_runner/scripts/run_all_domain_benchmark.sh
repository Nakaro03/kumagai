#!/usr/bin/env bash
# ============================================================
# 全ドメインベンチマーク: static / neural_ode / pnode / pnode_pc
#
# 比較手法: Static VGAE, NeuralODE, P-NODE, PC-PNODE（提案）
# 小規模ドメイン: min_events=20, epochs=20
# 大規模ドメイン: min_events=100, epochs=20
# 共通設定: year_range 2016-2021, holdout=2021, seed=42
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

OUTDIR="pnode_patent_runner/outputs/cope_benchmark"
METHODS="static,neural_ode,pnode,pnode_pc"
SEED=42
EPOCHS=20
TREND_W=0.1
DA_W=0.05

run_domain() {
    local DOMAIN=$1
    local YEAR_START=$2
    local YEAR_END=$3
    local HOLDOUT=$4
    local MIN_E=$5
    echo ""
    echo "========================================"
    echo "  DOMAIN: $DOMAIN  years=${YEAR_START}-${YEAR_END}  holdout=$HOLDOUT  min=$MIN_E"
    echo "========================================"
    python -m pnode_patent_runner.run_benchmark_comparison \
        --data-domain "$DOMAIN" \
        --year-range "$YEAR_START" "$YEAR_END" \
        --holdout-test-year "$HOLDOUT" \
        --min-patents "$MIN_E" \
        --methods "$METHODS" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --pnode-potential-feature mlp \
        --trend-weight "$TREND_W" \
        --pc-density-align-weight "$DA_W" \
        --potential-weight 0.1 \
        --trajectory-weight 0.1 \
        --output-json "$OUTDIR/benchmark_${DOMAIN}_seed${SEED}.json"
}

mkdir -p "$OUTDIR"

# author_topic は別ファイルに結果あり (benchmark_author_topic_pnode_pc_seed42.json)

# patent (企業–特許 IPC サブグループ)
run_domain patent        2015 2021 2021 5

# bipartite 小規模
run_domain energy        2016 2021 2021 20
run_domain agrifood      2016 2021 2021 20
run_domain construction  2016 2021 2021 20

# bipartite 大規模 (min=100 でノード数を~1万台に抑制)
run_domain pharma        2016 2021 2021 100
run_domain semiconductor 2016 2021 2021 100
run_domain computing     2016 2021 2021 100

echo ""
echo "========================================"
echo "  全ドメイン完了。集計中..."
echo "========================================"

python3 - <<'PYEOF'
import json, glob, os

OUTDIR = "pnode_patent_runner/outputs/cope_benchmark"
DOMAINS = [
    ("author_topic", "benchmark_author_topic_pnode_pc_seed42.json"),
    ("patent",       "benchmark_patent_seed42.json"),
    ("energy",       "benchmark_energy_seed42.json"),
    ("agrifood",     "benchmark_agrifood_seed42.json"),
    ("construction", "benchmark_construction_seed42.json"),
    ("pharma",       "benchmark_pharma_seed42.json"),
    ("semiconductor","benchmark_semiconductor_seed42.json"),
    ("computing",    "benchmark_computing_seed42.json"),
]
KEY_ORDER = ["static", "neural_ode", "pnode", "pnode_pc"]

print(f"\n{'Domain':15s} | {'Method':12s} | {'HO_AUC':>7} | {'HO_AP':>7} | {'ECE':>7}")
print("-" * 63)

for domain, fname in DOMAINS:
    path = os.path.join(OUTDIR, fname)
    if not os.path.isfile(path):
        print(f"{domain:15s} | (結果なし: {fname})")
        continue
    with open(path) as f:
        d = json.load(f)
    results = {r["key"]: r for r in d.get("results", [])}
    for key in KEY_ORDER:
        r = results.get(key, {})
        auc = r.get("final_val_auc", float("nan"))
        ap  = r.get("final_val_ap",  float("nan"))
        ece = r.get("final_val_ece", float("nan"))
        marker = " *" if key == "pnode_pc" else "  "
        print(f"{domain:15s} | {key:12s} | {auc:7.4f} | {ap:7.4f} | {ece:7.4f}{marker}")
    print("-" * 63)
PYEOF
