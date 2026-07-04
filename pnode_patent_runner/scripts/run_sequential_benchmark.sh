#!/usr/bin/env bash
# 全ドメイン順次ベンチマーク（GPU メモリ節約のため1ドメインずつ実行）
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMON="--year-range 2010 2020 --holdout-test-year 2020
        --methods static,neural_ode,pnode,cope
        --epochs 10 --hidden-dim 64 --latent-dim 16
        --pnode-ode-method rk4 --pnode-ode-n-steps 1
        --seed 42"

run() {
    local domain=$1 minp=$2
    echo "===== $domain (min_events=$minp) ====="
    python -u -m pnode_patent_runner.run_benchmark_comparison \
        --data-domain "$domain" --min-patents "$minp" $COMMON
    echo "===== $domain done ====="
}

run agrifood     50
run energy       50
run construction 50
run computing    200
run pharma       200
run semiconductor 200

echo "全ドメイン完了"

python3 - <<'PYEOF'
import json, glob
DOMAINS = ['agrifood','energy','construction','computing','pharma','semiconductor']
KEYS    = ['static','neural_ode','pnode','cope']
print(f"\n{'Domain':15s} | {'Method':12s} | {'AUC':>7} | {'AP':>7} | {'ECE':>7}")
print('-'*60)
for domain in DOMAINS:
    pat = f"pnode_patent_runner/outputs/cope_benchmark/benchmark_{domain}*seed42.json"
    files = sorted(glob.glob(pat))
    if not files:
        print(f"{domain:15s} | (未完了)")
        continue
    with open(files[-1]) as f:
        d = json.load(f)
    results = {r['key']: r for r in d['results']}
    for key in KEYS:
        r = results.get(key, {})
        auc = r.get('final_val_auc', float('nan'))
        ap  = r.get('final_val_ap',  float('nan'))
        ece = r.get('final_val_ece', float('nan'))
        print(f"{domain:15s} | {key:12s} | {auc:7.4f} | {ap:7.4f} | {ece:7.4f}")
    print('-'*60)
PYEOF
