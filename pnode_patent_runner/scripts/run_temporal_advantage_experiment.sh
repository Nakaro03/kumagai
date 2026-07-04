#!/usr/bin/env bash
# ================================================================
# 「時系列の長さ → 時系列モデルの優位性」検証実験
#
# 仮説: 短期(4年)では Static が最強。長期(12年)では時系列モデルが逆転する。
#
# 条件:
#   SHORT:  year-range 2018 2021  (4年)  holdout=2021
#   LONG:   year-range 2010 2021  (12年) holdout=2021
#   手法:   static, neural_ode, pnode, pnode_pc
#   epochs: 30
# ================================================================
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

OUTDIR="pnode_patent_runner/outputs/temporal_advantage"
METHODS="static,neural_ode,pnode,pnode_pc"
SEED=42
EPOCHS=30
TREND_W=0.1
DA_W=0.05
mkdir -p "$OUTDIR"

run() {
    local DOMAIN=$1; local YS=$2; local YE=$3; local MIN=$4; local TAG=$5
    echo ""
    echo "=== $DOMAIN [$TAG: ${YS}-${YE}] min=$MIN ==="
    python -m pnode_patent_runner.run_benchmark_comparison \
        --data-domain "$DOMAIN" \
        --year-range "$YS" "$YE" \
        --holdout-test-year 2021 \
        --min-patents "$MIN" \
        --methods "$METHODS" \
        --epochs "$EPOCHS" \
        --seed "$SEED" \
        --pnode-potential-feature mlp \
        --trend-weight "$TREND_W" \
        --pc-density-align-weight "$DA_W" \
        --potential-weight 0.1 \
        --trajectory-weight 0.1 \
        --output-json "$OUTDIR/bench_${DOMAIN}_${TAG}_seed${SEED}.json"
}

# ── 小規模ドメイン (min=20) ──────────────────────────────────────
for DOMAIN in energy agrifood construction; do
    run "$DOMAIN" 2018 2021 20 "short4"
    run "$DOMAIN" 2010 2021 20 "long12"
done

# ── 大規模ドメイン (min=200 でノード抑制) ───────────────────────
for DOMAIN in pharma semiconductor computing; do
    run "$DOMAIN" 2018 2021 200 "short4"
    run "$DOMAIN" 2010 2021 200 "long12"
done

# ── 集計 ─────────────────────────────────────────────────────────
echo ""
echo "========== 実験完了。集計中... =========="

python3 - << 'PYEOF'
import json, glob, os

OUTDIR = "pnode_patent_runner/outputs/temporal_advantage"
DOMAINS = ["energy", "agrifood", "construction", "pharma", "semiconductor", "computing"]
TAGS    = [("short4", "短期4年 2018-2021"), ("long12", "長期12年 2010-2021")]
KEY_ORDER = ["static", "neural_ode", "pnode", "pnode_pc"]

results = {}
for domain in DOMAINS:
    results[domain] = {}
    for tag, _ in TAGS:
        path = f"{OUTDIR}/bench_{domain}_{tag}_seed42.json"
        if not os.path.isfile(path):
            results[domain][tag] = {}
            continue
        with open(path) as f:
            d = json.load(f)
        results[domain][tag] = {r["key"]: r for r in d.get("results", [])}

# 差分表: (PC-PNODE - Static) の AUC で時系列モデル優位を確認
print("\n=== 手法別 Holdout AUC (short4 vs long12) ===\n")
for domain in DOMAINS:
    print(f"[ {domain} ]")
    print(f"  {'method':<12} {'short4':>8} {'long12':>8} {'gain':>8}")
    print(f"  {'-'*42}")
    for key in KEY_ORDER:
        r_s = results[domain].get("short4", {}).get(key, {})
        r_l = results[domain].get("long12", {}).get(key, {})
        auc_s = r_s.get("final_val_auc", float("nan"))
        auc_l = r_l.get("final_val_auc", float("nan"))
        gain  = auc_l - auc_s
        marker = " ◀" if key == "pnode_pc" else ""
        gain_str = f"{gain:+.4f}" if gain == gain else "   nan"
        print(f"  {key:<12} {auc_s:>8.4f} {auc_l:>8.4f} {gain_str:>8}{marker}")
    print()

# 要約: 長期でどれだけ Static を超えたか
print("=== 長期12年: (手法 - Static) の AUC 差 ===\n")
print(f"{'domain':<14} {'neural_ode':>11} {'pnode':>8} {'pnode_pc':>10}")
print("-" * 50)
for domain in DOMAINS:
    r = results[domain].get("long12", {})
    auc_static = r.get("static", {}).get("final_val_auc", float("nan"))
    for key in ["neural_ode", "pnode", "pnode_pc"]:
        a = r.get(key, {}).get("final_val_auc", float("nan"))
        r[key + "_diff"] = a - auc_static
    nd = r.get("neural_ode_diff", float("nan"))
    pd_ = r.get("pnode_diff", float("nan"))
    pcd = r.get("pnode_pc_diff", float("nan"))
    fmt = lambda v: f"{v:+.4f}" if v == v else "   nan"
    print(f"{domain:<14} {fmt(nd):>11} {fmt(pd_):>8} {fmt(pcd):>10}")
PYEOF
