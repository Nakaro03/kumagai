"""
マルチドメイン × 5 seed の trend benchmark。

データ: 6 つの bipartite ドメイン (energy, agrifood, construction, pharma, semiconductor, computing)
        ArXiv CS author_topic は別実験で実施済み。

評価:
  - PC-PNODE (A+BEF) vs Static, NeuralODE, PNODE
  - Spearman(Φ, g) を主指標に
  - 年範囲: 2010–2021 (12年, long history で H_C 検証)

使用:
  PNODE_DOMAIN=energy PNODE_SEED=42 python -m pnode_patent_runner.run_multidomain_trend
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle
from pnode_patent_runner.unified_training import (
    evaluate_val_future_link_metrics_for_years,
    train_model_improved,
)
from pnode_patent_runner.trend_evaluation import evaluate_trend_metrics

# ── 設定 ──────────────────────────────────────────────────────────────────────
DOMAIN     = os.environ.get("PNODE_DOMAIN", "energy")
SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 20))
YEAR_START = int(os.environ.get("PNODE_YEAR_START", 2010))
YEAR_END   = int(os.environ.get("PNODE_YEAR_END", 2021))
HOLDOUT    = int(os.environ.get("PNODE_HOLDOUT", YEAR_END))
MIN_EVENTS = int(os.environ.get("PNODE_MIN_EVENTS", 100))

# 改善のフラグ
USE_A      = bool(int(os.environ.get("PNODE_USE_A", "1")))   # 既定で A 有効

DOMAIN_CSV = {
    "energy":        "data/processed/bipartite_energy.csv",
    "agrifood":      "data/processed/bipartite_agrifood.csv",
    "construction":  "data/processed/bipartite_construction.csv",
    "pharma":        "data/processed/bipartite_pharma.csv",
    "semiconductor": "data/processed/bipartite_semiconductor.csv",
    "computing":     "data/processed/bipartite_computing.csv",
}

OUTDIR = Path("pnode_patent_runner/outputs/multidomain_trend")
TAG    = "_a" if USE_A else ""
OUT_JSON = OUTDIR / f"trend_{DOMAIN}{TAG}_y{YEAR_START}-{YEAR_END}_seed{SEED}.json"

METHODS = {
    "static":     dict(variant="static",     trend_weight=0.0),
    "neural_ode": dict(variant="neural_ode", trend_weight=0.0),
    "pnode":      dict(variant="pnode",      trend_weight=0.0),
    "pnode_pc":   dict(variant="pnode_pc",   trend_weight=1.0),
}

TRAIN_KWARGS = dict(
    num_epochs=EPOCHS,
    lr=1e-3,
    beta=0.01,
    pos_weight=5.0,
    latent_pred_weight=1.0,
    future_link_weight=10.0,
    potential_weight=0.1,
    trajectory_weight=0.1,
    num_neg_recon=800,
    num_neg_future=400,
    density_align_weight=0.05,
    loss_aux_warmup_epochs=5,
)


def build_model(variant, bundle, device):
    use_topic_pe = USE_A and (variant == "pnode_pc")
    return BenchmarkTemporalVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=128,
        latent_dim=2,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode="distance",
        variant=variant,
        pnode_potential_feature="mlp",
        year_min=YEAR_START,
        year_max=YEAR_END,
        topic_position_embedding=use_topic_pe,
    ).to(device)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    csv_path = DOMAIN_CSV.get(DOMAIN)
    if csv_path is None or not Path(csv_path).is_file():
        raise SystemExit(f"unknown or missing CSV for domain={DOMAIN}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Domain: {DOMAIN}  CSV: {csv_path}")
    print(f"YearRange: {YEAR_START}-{YEAR_END}  Holdout: {HOLDOUT}  MinEvents: {MIN_EVENTS}")
    print(f"Architecture A: {'enabled' if USE_A else 'disabled'}")

    bundle = load_bipartite_domain_graph_bundle(
        csv_path, min_events=MIN_EVENTS,
        year_min=YEAR_START, year_max=YEAR_END,
    )
    n_topics = len(bundle.right_nodes)
    print(f"  Actors={bundle.num_corps}  Topics(IPC)={n_topics}  N={bundle.total_n}  in_dim={bundle.in_dim}")
    if bundle.topic_growth_by_year is None:
        raise SystemExit("成長率が計算されていません。loader を確認してください。")

    all_years   = sorted(bundle.graphs.keys())
    train_years = [y for y in all_years if y < HOLDOUT]
    if not train_years:
        raise SystemExit("学習年がありません。HOLDOUT を確認してください。")
    graphs_train = {y: bundle.graphs[y] for y in train_years}
    hist_edges_train = {
        y: bundle.hist_edges[y] for y in train_years if y in bundle.hist_edges
    }
    year_prev = max(train_years)
    year_next = HOLDOUT

    # 成長率 ±1年平均
    growth = bundle.topic_growth_by_year or {}
    all_gy = sorted(growth.keys())
    smoothed = {}
    for y in all_gy:
        nb = [yy for yy in all_gy if abs(yy - y) <= 1]
        smoothed[y] = torch.stack([growth[yy] for yy in nb]).mean(dim=0)

    print(f"  TrainYears={train_years}  Eval: {year_prev}→{year_next}")

    results = []
    for key, cfg in METHODS.items():
        set_seed(SEED)
        variant = cfg["variant"]
        trend_w = cfg["trend_weight"]
        print(f"\n[{key}] 学習中... (variant={variant}, trend_weight={trend_w})")

        model = build_model(variant, bundle, device)
        kw = dict(**TRAIN_KWARGS)
        if variant == "pnode_pc" and trend_w > 0:
            kw["trend_weight"]        = trend_w
            kw["topic_growth_by_year"] = smoothed
            if USE_A:
                kw["trend_loss_type"]    = "listnet"
                kw["trend_listnet_temp"] = 0.5
                kw["trend_warmup_epochs"] = max(EPOCHS // 3, 5)
        else:
            kw["trend_weight"]        = 0.0
            kw["topic_growth_by_year"] = None

        train_model_improved(
            model, graphs_train, bundle.num_corps, hist_edges_train, **kw,
        )
        model.eval()

        link_m = evaluate_val_future_link_metrics_for_years(
            model, bundle.graphs, bundle.num_corps, device, year_prev, year_next,
        )
        growth_for_eval = smoothed if variant == "pnode_pc" else None
        trend_m = evaluate_trend_metrics(
            model, bundle.graphs, bundle.num_corps, device,
            year_prev, year_next,
            topic_growth_by_year=growth_for_eval,
            ndcg_k=10,
        )
        sig = "*" if trend_m.get("spearman_p", 1.0) < 0.05 else ""
        sp_str = f"{trend_m.get('spearman_r', float('nan')):+.4f}{sig}" if trend_m.get('spearman_r', float('nan')) == trend_m.get('spearman_r', float('nan')) else " N/A "
        print(f"  {key:<12} Link={link_m.get('auc', float('nan')):.4f}  "
              f"Entry={trend_m.get('entry_auc', float('nan')):.4f}  "
              f"Spearman={sp_str}(p={trend_m.get('spearman_p', float('nan')):.3f})  "
              f"NDCG={trend_m.get('ndcg', float('nan')):.3f}  "
              f"n_topics={n_topics}")

        results.append({
            "key": key,
            "domain": DOMAIN,
            "year_start": YEAR_START,
            "year_end": YEAR_END,
            "n_topics": n_topics,
            "link_auc":  link_m.get("auc", float("nan")),
            "link_ap":   link_m.get("ap", float("nan")),
            "entry_auc": trend_m.get("entry_auc", float("nan")),
            "spearman_r": trend_m.get("spearman_r", float("nan")),
            "spearman_p": trend_m.get("spearman_p", float("nan")),
            "ndcg_at_10": trend_m.get("ndcg", float("nan")),
            "phi_span":   trend_m.get("phi_span", float("nan")),
        })

    out = {"results": results, "settings": {
        "domain": DOMAIN, "year_start": YEAR_START, "year_end": YEAR_END,
        "holdout": HOLDOUT, "min_events": MIN_EVENTS, "epochs": EPOCHS,
        "seed": SEED, "use_a": USE_A,
    }}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
