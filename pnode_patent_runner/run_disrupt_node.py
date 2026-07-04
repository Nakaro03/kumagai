"""
DisruptNODE ベンチマーク実行スクリプト。

DisruptNODE を P-NODE / CoPE-VGAE / Gravity-VGAE / Static と比較する。
データ: arxiv 著者–トピック二部グラフ（dual_force_data.load_dual_force_bundle が
        topic_trend_plus / topic_trend_minus 付きで返すものを使用）

使用例:
  python -m pnode_patent_runner.run_disrupt_node \\
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \\
    --year-range 2020 2025 --epochs 15 --seed 42

  # 速度優先（latent_dim=8）:
  python -m pnode_patent_runner.run_disrupt_node \\
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \\
    --year-range 2020 2024 --epochs 10 --latent-dim 8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _default_csv() -> Path:
    for cand in (
        _REPO_ROOT / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        _REPO_ROOT / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError("arxiv CSV が見つかりません。--data で指定してください。")


def _build_disrupt(bundle, kw: Dict[str, Any], device: torch.device):
    from pnode_patent_runner.disrupt_node import DisruptVGAE
    return DisruptVGAE(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        hidden_dim=kw["hidden_dim"],
        latent_dim=kw["latent_dim"],
        initial_corp_vectors=bundle.init_vectors,
        w_pot_init=kw["w_pot_init"],
        link_score_mode=kw["link_score_mode"],
        sigma_init=kw["sigma_init"],
        ode_method=kw["ode_method"],
        ode_n_steps=kw["ode_n_steps"],
    ).to(device)


def _build_baseline(key: str, bundle, kw: Dict[str, Any], device: torch.device):
    from pnode_patent_runner.cope_experiment import ModelBuildKw, build_baseline_model, CopeGraphBundle

    years = sorted(bundle.graphs.keys())
    mk = ModelBuildKw(
        hidden_dim=kw["hidden_dim"],
        latent_dim=kw["latent_dim"],
        link_score_mode=kw["link_score_mode"],
        w_pot_init=kw["w_pot_init"],
        pnode_ode_method=kw["ode_method"],
        pnode_ode_n_steps=kw["ode_n_steps"],
        pnode_potential_feature="mlp",
        year_min=years[0],
        year_max=years[-1],
    )
    return build_baseline_model(key, device, bundle, mk)


def _train_disrupt(
    model,
    bundle,
    device,
    epochs: int,
    lr: float,
    loss_kw: Dict,
):
    from pnode_patent_runner.disrupt_node import train_one_epoch_disrupt
    from pnode_patent_runner.unified_training import evaluate_val_future_link_metrics

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0.0
    history: List[Dict] = []

    for ep in range(epochs):
        t0 = time.time()
        br = train_one_epoch_disrupt(
            model, bundle.graphs, bundle.num_corps,
            bundle.hist_edges, optimizer, device, loss_kw,
        )
        val = evaluate_val_future_link_metrics(
            model, bundle.graphs, bundle.num_corps, device
        )
        elapsed = time.time() - t0
        best_auc = max(best_auc, val["auc"])
        history.append({"epoch": ep + 1, "train_total": br["total"], **val})
        print(
            f"  ep{ep+1:>3d} | loss={br['total']:.4f} "
            f"recon={br['recon']:.4f} fut={br['future_link']:.4f} "
            f"| val AUC={val['auc']:.4f} AP={val['ap']:.4f} ECE={val['ece']:.4f} "
            f"| {elapsed:.1f}s"
        )
    return history, best_auc


def _train_baseline(
    model,
    bundle,
    device,
    epochs: int,
    lr: float,
    loss_kw: Dict,
):
    from pnode_patent_runner.unified_training import (
        train_one_epoch,
        evaluate_val_future_link_metrics,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0.0
    history: List[Dict] = []

    for ep in range(epochs):
        t0 = time.time()
        br = train_one_epoch(
            model, bundle.graphs, bundle.num_corps,
            bundle.hist_edges, optimizer, device, loss_kw,
        )
        val = evaluate_val_future_link_metrics(
            model, bundle.graphs, bundle.num_corps, device
        )
        elapsed = time.time() - t0
        best_auc = max(best_auc, val["auc"])
        history.append({"epoch": ep + 1, "train_total": br["total"], **val})
        print(
            f"  ep{ep+1:>3d} | loss={br['total']:.4f} "
            f"| val AUC={val['auc']:.4f} AP={val['ap']:.4f} "
            f"| {elapsed:.1f}s"
        )
    return history, best_auc


def _print_table(results: Dict[str, Dict]):
    header = f"{'Method':<20} {'AUC':>8} {'AP':>8} {'ECE':>8}"
    print("\n" + "=" * 50)
    print("  FINAL RESULTS (last epoch val metrics)")
    print("=" * 50)
    print(header)
    print("-" * 50)
    for name, r in results.items():
        print(
            f"{name:<20} {r['auc']:>8.4f} {r['ap']:>8.4f} {r['ece']:>8.4f}"
        )
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="DisruptNODE benchmark")
    parser.add_argument("--data", default=None, help="arxiv 著者–トピック CSV パス")
    parser.add_argument("--year-range", nargs=2, type=int, default=[2020, 2025])
    parser.add_argument("--min-papers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--methods",
        default="disrupt,pnode,cope,gravity,static",
        help="カンマ区切り: disrupt,pnode,cope,gravity,static,rnn,neural_ode",
    )
    parser.add_argument("--sigma-init", type=float, default=0.5,
                        help="TrendModulatedPotentialNet の Gaussian 幅初期値")
    parser.add_argument("--ode-method", default="rk4",
                        help="ODE 積分法: rk4 / euler / dopri5")
    parser.add_argument("--ode-n-steps", type=int, default=4)
    parser.add_argument("--out-json", default=None, help="結果 JSON 保存先")
    parser.add_argument("--topic-column", default="topic")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    csv_path = args.data or str(_default_csv())
    print(f"Data: {csv_path}")

    # データロード (dual_force_bundle でトレンド信号付き)
    from pnode_patent_runner.dual_force_data import load_dual_force_bundle
    from pnode_patent_runner.cope_experiment import subset_graphs

    print("Loading data with trend signals...")
    bundle = load_dual_force_bundle(
        csv_path,
        topic_column=args.topic_column,
        min_papers=args.min_papers,
    )

    # 年範囲フィルタ
    yr0, yr1 = args.year_range
    years_use = [y for y in sorted(bundle.graphs.keys()) if yr0 <= y <= yr1]
    if len(years_use) < 2:
        print(f"Error: 指定年範囲 {yr0}–{yr1} に利用可能なグラフが2年未満です。")
        sys.exit(1)
    bundle.graphs = {y: bundle.graphs[y] for y in years_use}
    print(
        f"Years: {years_use}  |  "
        f"num_corps(authors)={bundle.num_corps}  "
        f"num_topics={bundle.total_n - bundle.num_corps}  "
        f"total_nodes={bundle.total_n}"
    )

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    model_kw = dict(
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        link_score_mode="distance",
        w_pot_init=0.05,
        sigma_init=args.sigma_init,
        ode_method=args.ode_method,
        ode_n_steps=args.ode_n_steps,
    )

    loss_kw = dict(
        beta=0.01,
        pos_weight=5.0,
        latent_pred_weight=1.0,
        future_link_weight=10.0,
        potential_weight=0.01,
        trajectory_weight=0.05,
        num_neg_recon=800,
        num_neg_future=400,
    )

    results: Dict[str, Dict] = {}
    all_histories: Dict[str, List] = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"  Training: {method.upper()}  (seed={args.seed}, epochs={args.epochs})")
        print(f"{'='*50}")
        torch.manual_seed(args.seed)

        t_start = time.time()
        if method == "disrupt":
            model = _build_disrupt(bundle, model_kw, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Params: {n_params:,}")
            history, best_auc = _train_disrupt(
                model, bundle, device, args.epochs, args.lr, loss_kw
            )
        else:
            model = _build_baseline(method, bundle, model_kw, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Params: {n_params:,}")
            history, best_auc = _train_baseline(
                model, bundle, device, args.epochs, args.lr, loss_kw
            )

        total_time = time.time() - t_start
        last = history[-1]
        results[method] = {
            "auc": last["auc"],
            "ap": last["ap"],
            "ece": last["ece"],
            "best_auc": best_auc,
            "total_time_sec": total_time,
        }
        all_histories[method] = history
        print(
            f"  Finished in {total_time:.1f}s | best_auc={best_auc:.4f}"
        )

    _print_table(results)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"results": results, "histories": all_histories}, f, indent=2)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
