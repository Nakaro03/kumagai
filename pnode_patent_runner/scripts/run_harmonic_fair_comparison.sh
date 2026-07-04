#!/usr/bin/env bash
# 公平な対称実験:
#   1. static / rnn / neural_ode(Linear,K=4) / neural_ode_gru(GRU,K=4) / pnode(B+D) / harmonic
# 3 seeds (42, 43, 44) × author_topic ドメイン
#
# 比較の対称性:
#   - neural_ode    : Linear 履歴融合 K=4（従来）
#   - neural_ode_gru: GRU 履歴融合 K=4（P-NODE と対称）
#   - pnode         : MLP Φ + GRU 履歴融合 K=4（B+D 設定）
#   - harmonic      : 調和ポテンシャル（解析解、ODEソルバー不要）
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA="${DATA:-data/processed/arxiv_cs_embedded_2020-2026_full.csv}"
DOMAIN="${DOMAIN:-author_topic}"
TOPIC_COL="${TOPIC_COL:-topic}"
Y0="${Y0:-2022}"
Y1="${Y1:-2025}"
MIN_P="${MIN_P:-5}"
EPOCHS="${EPOCHS:-20}"
OUT_DIR="${OUT_DIR:-pnode_patent_runner/outputs/harmonic_fair_comparison}"
METHODS="static,rnn,neural_ode,neural_ode_gru,pnode,harmonic"

mkdir -p "$OUT_DIR"

for SEED in 42 43 44; do
  OUT_JSON="$OUT_DIR/benchmark_seed${SEED}.json"
  echo "=== seed=${SEED} -> $OUT_JSON ==="

  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain "$DOMAIN" \
    --data "$DATA" \
    --topic-column "$TOPIC_COL" \
    --year-range "$Y0" "$Y1" \
    --min-patents "$MIN_P" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --methods "$METHODS" \
    --loss-aux-warmup-epochs 0 \
    --pnode-potential-feature mlp \
    --pnode-history-len 4 \
    --pnode-hist-fuse-mode gru \
    --output-json "$OUT_JSON"

  echo ""
done

echo "=== 集計 (final_val_auc) ==="
python3 - <<'PYEOF'
import json, pathlib, statistics

out_dir = pathlib.Path("pnode_patent_runner/outputs/harmonic_fair_comparison")
seeds = [42, 43, 44]
method_order = ["static", "rnn", "neural_ode", "neural_ode_gru", "pnode", "harmonic"]
label_map = {
    "static": "Static",
    "rnn": "RNN+VGAE",
    "neural_ode": "NeuralODE(Linear,K=4)",
    "neural_ode_gru": "NeuralODE(GRU,K=4)",
    "pnode": "P-NODE(B+D)",
    "harmonic": "Harmonic-VGAE",
}

scores = {mk: [] for mk in method_order}
for seed in seeds:
    p = out_dir / f"benchmark_seed{seed}.json"
    if not p.exists():
        print(f"  missing: {p}")
        continue
    d = json.loads(p.read_text())
    for r in d["results"]:
        k = r["key"]
        if k in scores:
            scores[k].append(r["final_val_auc"])

print(f"\n{'手法':<26} {'mean AUC':>9}  {'std':>6}  n")
print("-" * 50)
for mk in method_order:
    vals = scores[mk]
    if not vals:
        print(f"{label_map[mk]:<26}  ---")
        continue
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{label_map[mk]:<26}  {mean:.4f}    {std:.4f}  {len(vals)}")
PYEOF
