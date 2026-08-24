#!/usr/bin/env python3
"""
特許ドメイン（firm×CPC 二部グラフ、`data/processed/bipartite_{domain}.csv`）で
Dual-Force P-NODE (v2) を学習・評価する。`run_tap_node_patent_domain.py` と同じ手続き
（holdout-test-year, future-link AUC/AP/ECE）を Dual-Force に適用する。

未解決の設計論点（`docs/DUAL_FORCE_REDESIGN.md`）に対応する2つのablationフラグ:
  --d-scale-mode {raw,learnable,zscore}   Key に足し込む D_j の大きさの扱い
  --renorm-masked-attention               マスク後に成長側/衰退側で再正規化するか

例:
  python -m pnode_patent_runner.run_dual_force_patent_domain \\
    --domain construction --year-start 2017 --year-end 2021 --holdout-test-year 2021 \\
    --epochs 10 --seed 42 --d-scale-mode learnable --renorm-masked-attention \\
    --output-json pnode_patent_runner/outputs/dual_force_patent/dual_force_construction_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple

import torch

from pnode_patent_runner.cope_experiment import hist_edges_union_from_graphs
from pnode_patent_runner.dual_force_data_patent import load_dual_force_bundle_patent_domain
from pnode_patent_runner.dual_force_eval import future_link_auc_scores_dual_force
from pnode_patent_runner.dual_force_training import train_dual_force
from pnode_patent_runner.dual_force_vgae import DualForceVGAE
from pnode_patent_runner.unified_training import future_link_metrics_from_scores


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dual-Force P-NODE (v2) on patent CPC bipartite domains, holdout evaluation"
    )
    p.add_argument("--domain", type=str, required=True,
                    help="agrifood / construction / energy / semiconductor / pharma / computing")
    p.add_argument("--csv-path", type=str, default=None,
                    help="省略時は data/processed/bipartite_{domain}.csv")
    p.add_argument("--year-start", type=int, default=2017)
    p.add_argument("--year-end", type=int, default=2021)
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--link-score-mode", type=str, default="distance", choices=("distance", "cosine"))
    p.add_argument(
        "--d-scale-mode", type=str, default="raw", choices=("raw", "learnable", "zscore", "rank"),
        help="Key = W_K(P_j + scale(D_j)) の scale の扱い（②A=learnable, ②B=zscore, 既定=raw=現行版）",
    )
    p.add_argument(
        "--renorm-masked-attention", action="store_true",
        help="マスク後に成長側/衰退側それぞれの中で確率質量を1に再正規化する（①B、既定は①A=しない）",
    )
    p.add_argument("--holdout-test-year", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--save-checkpoint", type=Path, default=None)
    p.add_argument(
        "--coarsen-to-maingroup", action="store_true",
        help="CPCノードをsubgroup粒度(生の値、既定)ではなくmaingroup粒度(\"/\"以前)に粗視化する",
    )
    args = p.parse_args()

    torch.manual_seed(int(args.seed))

    csv_path = args.csv_path or f"data/processed/bipartite_{args.domain}.csv"
    bundle = load_dual_force_bundle_patent_domain(
        csv_path,
        year_range=(int(args.year_start), int(args.year_end)),
        min_events=int(args.min_events),
        coarsen_to_maingroup=bool(args.coarsen_to_maingroup),
    )
    graphs_f = bundle.graphs
    if len(graphs_f) < 2:
        raise SystemExit("年範囲内に2年以上必要です。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualForceVGAE(
        bundle.total_n,
        bundle.num_corps,
        bundle.in_dim,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        initial_author_vectors=bundle.init_vectors,
        gamma=float(args.gamma),
        link_score_mode=args.link_score_mode,
        d_scale_mode=args.d_scale_mode,
        renorm_masked_attention=bool(args.renorm_masked_attention),
    ).to(device)

    for d in graphs_f.values():
        d.to(device)

    graphs_train = graphs_f
    hist: Set[Tuple[int, int]] = bundle.hist_edges
    eval_year_prev = None
    if args.holdout_test_year is not None:
        hy = int(args.holdout_test_year)
        if hy not in graphs_f:
            raise SystemExit(f"holdout-test-year={hy} が年範囲内にありません。")
        train_years = sorted(y for y in graphs_f if y < hy)
        if len(train_years) < 2:
            raise SystemExit(f"holdout の前に2年以上必要です。現状: {train_years}")
        graphs_train = {y: graphs_f[y] for y in train_years}
        hist = hist_edges_union_from_graphs(graphs_train, bundle.num_corps)
        eval_year_prev = train_years[-1]
        print(
            f"ホールドアウト: 学習年={train_years}, テスト遷移 {eval_year_prev}→{hy}",
            file=sys.stderr,
        )

    _ = train_dual_force(
        model, graphs_train, bundle.num_corps, hist, num_epochs=int(args.epochs), lr=float(args.lr),
    )

    if args.holdout_test_year is not None:
        yt, ys = future_link_auc_scores_dual_force(
            model, graphs_f, bundle.num_corps, device,
            int(eval_year_prev), int(args.holdout_test_year),
        )
        metrics = future_link_metrics_from_scores(yt, ys)
    else:
        years_sorted = sorted(graphs_f.keys())
        yt, ys = future_link_auc_scores_dual_force(
            model, graphs_f, bundle.num_corps, device, years_sorted[-2], years_sorted[-1],
        )
        metrics = future_link_metrics_from_scores(yt, ys)

    ode_func = model.temporal_predictor.ode_func
    learned = {"gamma": float(ode_func.gamma.abs().item())}
    if args.d_scale_mode == "learnable":
        learned["d_scale"] = float(torch.exp(ode_func.log_d_scale).item())

    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "dual_force_v2",
        "data_domain": args.domain,
        "csv_path": str(csv_path),
        "year_range": [int(args.year_start), int(args.year_end)],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "holdout_test_year": (
            int(args.holdout_test_year) if args.holdout_test_year is not None else None
        ),
        "config": {
            "hidden_dim": int(args.hidden_dim),
            "latent_dim": int(args.latent_dim),
            "gamma_init": float(args.gamma),
            "link_score_mode": args.link_score_mode,
            "d_scale_mode": args.d_scale_mode,
            "renorm_masked_attention": bool(args.renorm_masked_attention),
        },
        "dual_force_final_val_auc": metrics.get("auc"),
        "dual_force_final_val_ap": metrics.get("ap"),
        "dual_force_final_val_ece": metrics.get("ece"),
        "learned_scalars": learned,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        oj = Path(args.output_json)
        oj.parent.mkdir(parents=True, exist_ok=True)
        with open(oj, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {oj}", file=sys.stderr)

    if args.save_checkpoint is not None:
        sc = Path(args.save_checkpoint)
        sc.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "num_nodes": int(bundle.total_n),
                "num_authors": int(bundle.num_corps),
                "hidden_dim": int(args.hidden_dim),
                "latent_dim": int(args.latent_dim),
                "input_dim": int(bundle.in_dim),
                "initial_author_vectors": bundle.init_vectors,
                "link_score_mode": str(args.link_score_mode),
                "d_scale_mode": args.d_scale_mode,
                "renorm_masked_attention": bool(args.renorm_masked_attention),
                "gamma_init": float(args.gamma),
                "domain": args.domain,
                "csv_path": str(csv_path),
                "year_range": [int(args.year_start), int(args.year_end)],
                "holdout_test_year": (
                    int(args.holdout_test_year) if args.holdout_test_year is not None else None
                ),
            },
            str(sc),
        )
        print(f"Checkpoint: {sc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
