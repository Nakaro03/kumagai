#!/usr/bin/env python3
"""
特許ドメイン（firm×CPC 二部グラフ、`data/processed/bipartite_{domain}.csv`）で
TAP-NODE（burst 拡張込み）を学習・評価する。`run_tap_node_author_topic.py` と同じ手続き
（holdout-test-year, future-link AUC/AP/ECE）を、著者–トピックではなく特許ドメインに適用する。

`docs/DUAL_FORCE_REDESIGN.md` の Gate L（TAP-NODE+burst が Gate S を上回るか）に使う。
burst_j(t) の定義・データローダーは `dual_force_data_patent.py`（`gem_skill.py` と同一定義、
`docs/RELATED_WORK_SURVEY.md` の Gate 0 と共有）。

例:
  python -m pnode_patent_runner.run_tap_node_patent_domain \\
    --domain construction --year-start 2017 --year-end 2021 --holdout-test-year 2021 \\
    --epochs 10 --seed 42 --scalar-lr 0.05 \\
    --output-json pnode_patent_runner/outputs/tap_node_patent/tap_node_construction_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Set, Tuple

import torch
import torch.nn.functional as F

from pnode_patent_runner.cope_experiment import hist_edges_union_from_graphs
from pnode_patent_runner.dual_force_data_patent import load_dual_force_bundle_patent_domain
from pnode_patent_runner.dual_force_eval import future_link_auc_scores_dual_force
from pnode_patent_runner.dual_force_training import train_dual_force
from pnode_patent_runner.tap_node_models import TAPVGAE
from pnode_patent_runner.unified_training import future_link_metrics_from_scores


def main() -> int:
    p = argparse.ArgumentParser(
        description="TAP-NODE(+burst) on patent CPC bipartite domains (Gate L)"
    )
    p.add_argument("--domain", type=str, required=True,
                    help="agrifood / construction / energy / semiconductor / pharma / computing")
    p.add_argument("--csv-path", type=str, default=None,
                    help="省略時は data/processed/bipartite_{domain}.csv")
    p.add_argument("--year-start", type=int, default=2017)
    p.add_argument("--year-end", type=int, default=2021)
    p.add_argument("--min-events", type=int, default=2)
    p.add_argument("--burst-percentile", type=float, default=80.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--kappa-init", type=float, default=1.0)
    p.add_argument("--b-init", type=float, default=0.5)
    p.add_argument("--log-h-init", type=float, default=0.0)
    p.add_argument("--c-init", type=float, default=0.0,
                    help="burst 交互作用スカラー c の初期値（0=既定TAP-NODEと同一挙動から学習開始）")
    p.add_argument(
        "--scalar-lr", type=float, default=None,
        help="ODE スカラー (kappa,b,log_h,log_scale,c) 専用の学習率。"
        "総ステップ数が少ないと既定 lr では動けない（既知の pitfall）。",
    )
    p.add_argument("--link-score-mode", type=str, default="distance", choices=("distance", "cosine"))
    p.add_argument(
        "--ode-method", type=str, default="dopri5", choices=("dopri5", "rk4", "euler"),
        help="dopri5=適応的多段積分(既定・連続時間)。euler --ode-n-steps 1 で離散1ステップ版になる。",
    )
    p.add_argument(
        "--ode-n-steps", type=int, default=4,
        help="rk4/euler の固定刻み数。dopri5では無視される。連続時間 vs 離散1ステップの"
        "アブレーションには --ode-method euler --ode-n-steps 1 を使う。",
    )
    p.add_argument("--holdout-test-year", type=int, default=None)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--save-checkpoint", type=Path, default=None)
    args = p.parse_args()

    torch.manual_seed(int(args.seed))

    csv_path = args.csv_path or f"data/processed/bipartite_{args.domain}.csv"
    bundle = load_dual_force_bundle_patent_domain(
        csv_path,
        year_range=(int(args.year_start), int(args.year_end)),
        min_events=int(args.min_events),
        burst_percentile=float(args.burst_percentile),
    )
    graphs_f = bundle.graphs
    if len(graphs_f) < 2:
        raise SystemExit("年範囲内に2年以上必要です。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAPVGAE(
        bundle.total_n,
        bundle.num_corps,
        bundle.in_dim,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        initial_author_vectors=bundle.init_vectors,
        link_score_mode=args.link_score_mode,
        ode_method=args.ode_method,
        ode_n_steps=int(args.ode_n_steps),
    ).to(device)
    ode_func = model.temporal_predictor.ode_func
    with torch.no_grad():
        ode_func.kappa.fill_(float(args.kappa_init))
        ode_func.b.fill_(float(args.b_init))
        ode_func.log_h.fill_(float(args.log_h_init))
        ode_func.c.fill_(float(args.c_init))

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

    optimizer = None
    if args.scalar_lr is not None:
        scalar_ids = {id(q) for q in ode_func.parameters()}
        base_params = [q for q in model.parameters() if id(q) not in scalar_ids]
        optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": float(args.lr)},
                {"params": list(ode_func.parameters()), "lr": float(args.scalar_lr)},
            ]
        )

    _ = train_dual_force(
        model, graphs_train, bundle.num_corps, hist, num_epochs=int(args.epochs), lr=float(args.lr),
        optimizer=optimizer,
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

    learned = {
        "kappa": float(ode_func.kappa.item()),
        "b": float(ode_func.b.item()),
        "h": float(torch.exp(ode_func.log_h).item()),
        "alpha": float(F.softplus(ode_func.log_scale).item()),
        "c": float(ode_func.c.item()),
    }

    out = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "tap_node_burst",
        "data_domain": args.domain,
        "csv_path": str(csv_path),
        "year_range": [int(args.year_start), int(args.year_end)],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "holdout_test_year": (
            int(args.holdout_test_year) if args.holdout_test_year is not None else None
        ),
        "burst_percentile": float(args.burst_percentile),
        "tap_node_config": {
            "hidden_dim": int(args.hidden_dim),
            "latent_dim": int(args.latent_dim),
            "kappa_init": float(args.kappa_init),
            "b_init": float(args.b_init),
            "log_h_init": float(args.log_h_init),
            "c_init": float(args.c_init),
            "link_score_mode": args.link_score_mode,
            "scalar_lr": args.scalar_lr,
            "ode_method": args.ode_method,
            "ode_n_steps": int(args.ode_n_steps),
        },
        "tap_node_final_val_auc": metrics.get("auc"),
        "tap_node_final_val_ap": metrics.get("ap"),
        "tap_node_final_val_ece": metrics.get("ece"),
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
                "domain": args.domain,
                "csv_path": str(csv_path),
                "year_range": [int(args.year_start), int(args.year_end)],
                "holdout_test_year": (
                    int(args.holdout_test_year) if args.holdout_test_year is not None else None
                ),
                "burst_percentile": float(args.burst_percentile),
            },
            str(sc),
        )
        print(f"Checkpoint: {sc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
