#!/usr/bin/env python3
"""
著者–トピックデータで TAP-NODE (Trend-Anchored Potential Neural ODE) を学習・評価し、
既存の P-NODE / Dual-Force の JSON と並べて比較する。

手続き（データ・損失・評価）は Dual-Force ランナーと同一
（`run_dual_force_vs_pnode_author_topic` / `run_benchmark_comparison` の future-link 最終2年）。

例:
  python -m pnode_patent_runner.run_tap_node_author_topic \\
    --epochs 20 --seed 42 \\
    --output-json pnode_patent_runner/outputs/tap_node/tap_node_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from pnode_patent_runner.dual_force_data import load_dual_force_bundle
from pnode_patent_runner.dual_force_eval import evaluate_dual_force_future_link_metrics
from pnode_patent_runner.dual_force_training import train_dual_force
from pnode_patent_runner.tap_node_models import TAPVGAE


def _load_benchmark_rows(path: Path, seed: int) -> Optional[Dict[str, Any]]:
    """run_benchmark_comparison の JSON から手法別 AUC/AP/ECE を抜く。"""
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if int(raw.get("seed", -1)) != int(seed):
        return None
    out = {}
    for row in raw.get("results") or []:
        out[row.get("key")] = {
            "final_val_auc": row.get("final_val_auc"),
            "final_val_ap": row.get("final_val_ap"),
            "final_val_ece": row.get("final_val_ece"),
        }
    out["source_json"] = str(path)
    return out


def _load_dual_force_row(path: Path, seed: int) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if int(raw.get("seed", -1)) != int(seed):
        return None
    return {
        "final_val_auc": raw.get("dual_force_final_val_auc"),
        "final_val_ap": raw.get("dual_force_final_val_ap"),
        "final_val_ece": raw.get("dual_force_final_val_ece"),
        "source_json": str(path),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="TAP-NODE vs P-NODE / Dual-Force (著者–トピック, 同一データ手続き)"
    )
    p.add_argument(
        "--data",
        type=str,
        default="data/processed/arxiv_cs_embedded_2020-2026_full.csv",
    )
    p.add_argument("--year-start", type=int, default=2022)
    p.add_argument("--year-end", type=int, default=2025)
    p.add_argument("--min-patents", type=int, default=5)
    p.add_argument("--topic-column", type=str, default="topic")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--kappa-init", type=float, default=1.0)
    p.add_argument("--b-init", type=float, default=0.5)
    p.add_argument("--log-h-init", type=float, default=0.0)
    p.add_argument(
        "--scalar-lr",
        type=float,
        default=None,
        help="ODE スカラー (κ, b, log_h, log_scale) 専用の学習率。"
        "Adam のステップ幅は lr 程度なので、総ステップ数が少ないと既定 lr では動けない。",
    )
    p.add_argument(
        "--link-score-mode",
        type=str,
        default="distance",
        choices=("distance", "cosine"),
    )
    p.add_argument(
        "--refs-dir",
        type=Path,
        default=None,
        help="multiseed 比較 JSON のディレクトリ（pnode_baselines_seed{S}.json / dual_force_vs_pnode_seed{S}.json）",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--save-checkpoint", type=Path, default=None)
    p.add_argument(
        "--holdout-test-year",
        type=int,
        default=None,
        help="この年のエッジを学習に含めず、(直前年→この年) を最終テストとする。"
        "hist_edges も学習年のみで再構成（run_benchmark_comparison と同一プロトコル）。",
    )
    args = p.parse_args()

    torch.manual_seed(int(args.seed))

    bundle = load_dual_force_bundle(
        args.data,
        topic_column=args.topic_column,
        min_papers=args.min_patents,
    )
    graphs = bundle.graphs
    graphs_f = {y: g for y, g in graphs.items() if int(args.year_start) <= y <= int(args.year_end)}
    if len(graphs_f) < 2:
        raise SystemExit("年範囲内に2年以上必要です。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TAPVGAE(
        bundle.total_n,
        bundle.num_corps,
        bundle.in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_author_vectors=bundle.init_vectors,
        link_score_mode=args.link_score_mode,
    ).to(device)
    ode_func = model.temporal_predictor.ode_func
    with torch.no_grad():
        ode_func.kappa.fill_(float(args.kappa_init))
        ode_func.b.fill_(float(args.b_init))
        ode_func.log_h.fill_(float(args.log_h_init))

    for d in graphs_f.values():
        d.to(device)

    graphs_train = graphs_f
    hist: Set[Tuple[int, int]] = bundle.hist_edges
    eval_year_prev = None
    if args.holdout_test_year is not None:
        from pnode_patent_runner.cope_experiment import hist_edges_union_from_graphs

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
    _ = train_dual_force(  # noqa: F841
        model, graphs_train, bundle.num_corps, hist, num_epochs=int(args.epochs), lr=float(args.lr),
        optimizer=optimizer,
    )

    if args.holdout_test_year is not None:
        from pnode_patent_runner.dual_force_eval import (
            future_link_auc_scores_dual_force,
        )
        from pnode_patent_runner.unified_training import future_link_metrics_from_scores

        yt, ys = future_link_auc_scores_dual_force(
            model, graphs_f, bundle.num_corps, device,
            int(eval_year_prev), int(args.holdout_test_year),
        )
        metrics = future_link_metrics_from_scores(yt, ys)
    else:
        metrics = evaluate_dual_force_future_link_metrics(
            model, graphs_f, bundle.num_corps, device
        )

    learned = {
        "kappa": float(ode_func.kappa.item()),
        "b": float(ode_func.b.item()),
        "h": float(torch.exp(ode_func.log_h).item()),
        "alpha": float(F.softplus(ode_func.log_scale).item()),
    }

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
                "learned_scalars": learned,
            },
            str(sc),
        )
        print(f"Checkpoint: {sc}", file=sys.stderr)

    pnode_ref = None
    dual_force_ref = None
    if args.refs_dir is not None:
        pnode_ref = _load_benchmark_rows(
            args.refs_dir / f"pnode_baselines_seed{args.seed}.json", int(args.seed)
        )
        dual_force_ref = _load_dual_force_row(
            args.refs_dir / f"dual_force_vs_pnode_seed{args.seed}.json", int(args.seed)
        )

    out: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "tap_node",
        "data_domain": "author_topic",
        "data": str(args.data),
        "year_range": [int(args.year_start), int(args.year_end)],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "holdout_test_year": (
            int(args.holdout_test_year) if args.holdout_test_year is not None else None
        ),
        "tap_node_config": {
            "hidden_dim": int(args.hidden_dim),
            "latent_dim": int(args.latent_dim),
            "kappa_init": float(args.kappa_init),
            "b_init": float(args.b_init),
            "log_h_init": float(args.log_h_init),
            "link_score_mode": args.link_score_mode,
        },
        "tap_node_final_val_auc": metrics.get("auc"),
        "tap_node_final_val_ap": metrics.get("ap"),
        "tap_node_final_val_ece": metrics.get("ece"),
        "learned_scalars": learned,
        "baselines_reference": pnode_ref,
        "dual_force_reference": dual_force_ref,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        oj = Path(args.output_json)
        oj.parent.mkdir(parents=True, exist_ok=True)
        with open(oj, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {oj}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
