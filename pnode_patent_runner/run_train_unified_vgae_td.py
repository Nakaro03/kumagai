#!/usr/bin/env python3
"""
UnifiedVGAETD（時間依存 Φ(z, year)）を企業–特許データで学習し checkpoint を保存する。

例（リポジトリルートから）:
  python -m pnode_patent_runner.run_train_unified_vgae_td \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --year-range 2010 2020 --epochs 20 \\
    --save pnode_patent_runner/outputs/cope_landscape/unified_vgae_td.pt

可視化:
  python -m pnode_patent_runner.run_interactive_landscape_td_vector_field \\
    --load-checkpoint ... --data ... --year-range ...
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import load_cope_graph_bundle
from pnode_patent_runner.unified_training import (
    README_DEFAULT_BETA,
    README_DEFAULT_FUTURE_LINK_WEIGHT,
    README_DEFAULT_LATENT_PRED_WEIGHT,
    README_DEFAULT_NUM_NEG_FUTURE,
    README_DEFAULT_NUM_NEG_RECON,
    README_DEFAULT_POS_WEIGHT,
    README_DEFAULT_POTENTIAL_WEIGHT,
    README_DEFAULT_TRAJECTORY_WEIGHT,
)
from pnode_patent_runner.unified_training_td import evaluate_val_auc_td, train_model_td
from pnode_patent_runner.unified_vgae_td import METHOD_SHORT_NAME_TD, UnifiedVGAETD


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_csv = repo / "notebooks/work/dataset/topic_info3.csv"
    default_save = repo / "pnode_patent_runner/outputs/cope_landscape/unified_vgae_td.pt"

    p = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME_TD} — Φ(z, year) で学習し .pt を保存",
    )
    p.add_argument("--data", type=str, default=str(default_csv))
    p.add_argument(
        "--year-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
    )
    p.add_argument("--years", type=str, default="")
    p.add_argument("--all-years", action="store_true")
    p.add_argument("--min-patents", type=int, default=2)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--save", type=str, default=str(default_save))
    p.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="distance",
    )
    p.add_argument("--cosine-logit-scale", type=float, default=5.0)
    p.add_argument("--w-pot-init", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=README_DEFAULT_BETA)
    p.add_argument("--pos-weight", type=float, default=README_DEFAULT_POS_WEIGHT)
    p.add_argument("--potential-weight", type=float, default=README_DEFAULT_POTENTIAL_WEIGHT)
    p.add_argument("--trajectory-weight", type=float, default=README_DEFAULT_TRAJECTORY_WEIGHT)
    p.add_argument("--latent-pred-weight", type=float, default=README_DEFAULT_LATENT_PRED_WEIGHT)
    p.add_argument("--future-link-weight", type=float, default=README_DEFAULT_FUTURE_LINK_WEIGHT)
    p.add_argument("--num-neg-recon", type=int, default=README_DEFAULT_NUM_NEG_RECON)
    p.add_argument("--num-neg-future", type=int, default=README_DEFAULT_NUM_NEG_FUTURE)
    args = p.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("可視化のため latent_dim=2 を推奨します。")

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    yr = tuple(args.year_range) if args.year_range is not None else None
    try:
        bundle = load_cope_graph_bundle(
            data_path,
            min_patents=args.min_patents,
            year_range=yr,
            years_csv=args.years,
            all_years=args.all_years,
        )
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e))

    years_sorted = sorted(bundle.graphs.keys())
    if len(years_sorted) < 2:
        raise SystemExit("年が2年以上必要です。")
    y_min, y_max = years_sorted[0], years_sorted[-1]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"years=[{y_min}, {y_max}] count={len(years_sorted)}, num_corps={bundle.num_corps}, total_n={bundle.total_n}")

    model = UnifiedVGAETD(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        year_min=y_min,
        year_max=y_max,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_corp_vectors=bundle.init_vectors,
        w_pot_init=args.w_pot_init,
        link_score_mode=args.cope_link_score,
        cosine_logit_scale=args.cosine_logit_scale,
    ).to(device)

    _, _, best_auc, hist = train_model_td(
        model,
        bundle.graphs,
        bundle.num_corps,
        bundle.hist_edges,
        num_epochs=args.epochs,
        potential_weight=args.potential_weight,
        trajectory_weight=args.trajectory_weight,
        lr=args.lr,
        latent_pred_weight=args.latent_pred_weight,
        future_link_weight=args.future_link_weight,
        num_neg_recon=args.num_neg_recon,
        num_neg_future=args.num_neg_future,
        beta=args.beta,
        pos_weight=args.pos_weight,
    )
    final_auc = evaluate_val_auc_td(model, bundle.graphs, bundle.num_corps, device)
    print(f"best_val_auc (max over epochs): {best_auc:.4f}")
    print(f"final_val_auc: {final_auc:.4f}")
    if hist["val_auc"]:
        print(f"val_auc per epoch: {[round(x, 4) for x in hist['val_auc']]}")

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "year_min": y_min,
        "year_max": y_max,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "cope_link_score": args.cope_link_score,
        "cosine_logit_scale": args.cosine_logit_scale,
    }
    torch.save(payload, str(out))
    print(f"Wrote: {out}")
    print(
        "可視化例:\n"
        f"  python -m pnode_patent_runner.run_interactive_landscape_td_vector_field \\\n"
        f"    --data {data_path} --year-range {y_min} {y_max} \\\n"
        f"    --load-checkpoint {out} --cope-link-score {args.cope_link_score}"
    )


if __name__ == "__main__":
    main()
