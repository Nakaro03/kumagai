#!/usr/bin/env python3
"""
CoPE（UnifiedVGAE）の補助損失の有効性を、README と同じデータパイプラインで学習し
future-link AUC（`evaluate_val_auc`）を比較する。

データ読込は `cope_experiment.load_cope_graph_bundle`（README_COPE.md の手順と同一）。

例（リポジトリルート）:
  python -m pnode_patent_runner.run_cope_effectiveness \\
    --data notebooks/work/dataset/topic_info3.csv \\
    --epochs 5 \\
    --seed 42

- **cope**: recon + KL + latent 予測 + future リンク + potential + trajectory（既定重み）
- **ablation**: 上記のうち latent / future / potential / trajectory の重みを 0（VGAE 再構成＋KL 中心）
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import load_cope_graph_bundle
from pnode_patent_runner.unified_training import evaluate_val_auc, train_model_improved
from pnode_patent_runner.unified_vgae import METHOD_SHORT_NAME, UnifiedVGAE


def _make_model(
    device: torch.device,
    total_n: int,
    num_corps: int,
    in_dim: int,
    args: argparse.Namespace,
    init_vectors: torch.Tensor,
) -> UnifiedVGAE:
    return UnifiedVGAE(
        num_nodes=total_n,
        num_corps=num_corps,
        input_dim=in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_corp_vectors=init_vectors,
        w_pot_init=args.w_pot_init,
        link_score_mode=args.cope_link_score,
        cosine_logit_scale=args.cosine_logit_scale,
        density_calibrated_potential=bool(args.cope_density_calibrated),
        density_log_weight=float(args.cope_density_log_weight),
        density_ema_momentum=float(args.cope_density_ema_momentum),
    ).to(device)


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_csv = repo / "notebooks/work/dataset/topic_info3.csv"

    p = argparse.ArgumentParser(
        description=f"{METHOD_SHORT_NAME} — CoPE 補助損失の有効性（future-link AUC 比較）",
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--mode",
        type=str,
        choices=("both", "cope", "ablation"),
        default="both",
        help="both: 両方学習して比較 / cope / ablation のみ",
    )
    p.add_argument(
        "--cope-link-score",
        type=str,
        choices=("distance", "cosine"),
        default="distance",
        help="README のインタラクティブ例に合わせ既定は distance",
    )
    p.add_argument("--cosine-logit-scale", type=float, default=5.0)
    p.add_argument("--w-pot-init", type=float, default=0.05)
    p.add_argument("--potential-weight", type=float, default=0.01)
    p.add_argument("--trajectory-weight", type=float, default=0.05)
    p.add_argument("--latent-pred-weight", type=float, default=1.0)
    p.add_argument("--future-link-weight", type=float, default=10.0)
    p.add_argument(
        "--cope-density-calibrated",
        action="store_true",
        help="案1: 密度校準ポテンシャル（EMA 対角ガウス log p）",
    )
    p.add_argument("--cope-density-log-weight", type=float, default=1.0)
    p.add_argument("--cope-density-ema-momentum", type=float, default=0.05)
    args = p.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(
        f"{METHOD_SHORT_NAME} effectiveness — mode={args.mode}, epochs={args.epochs}, "
        f"seed={args.seed}, years={sorted(bundle.graphs.keys())}"
    )

    def run_one(
        label: str,
        *,
        latent_pred_weight: float,
        future_link_weight: float,
        potential_weight: float,
        trajectory_weight: float,
    ) -> Tuple[float, List[float]]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = _make_model(
            device,
            bundle.total_n,
            bundle.num_corps,
            bundle.in_dim,
            args,
            bundle.init_vectors,
        )
        _, _, best_auc, hist = train_model_improved(
            model,
            bundle.graphs,
            bundle.num_corps,
            bundle.hist_edges,
            num_epochs=args.epochs,
            potential_weight=potential_weight,
            trajectory_weight=trajectory_weight,
            lr=args.lr,
            latent_pred_weight=latent_pred_weight,
            future_link_weight=future_link_weight,
        )
        final_auc = evaluate_val_auc(model, bundle.graphs, bundle.num_corps, device)
        print(f"\n[{label}] best_val_auc(途中最大): {best_auc:.4f}")
        print(f"[{label}] final_val_auc(最終エポック後): {final_auc:.4f}")
        if hist["val_auc"]:
            print(f"[{label}] val_auc per epoch: {[round(x, 4) for x in hist['val_auc']]}")
        return final_auc, hist["val_auc"]

    results: Dict[str, float] = {}

    if args.mode in ("both", "cope"):
        results["cope"], _ = run_one(
            "cope",
            latent_pred_weight=args.latent_pred_weight,
            future_link_weight=args.future_link_weight,
            potential_weight=args.potential_weight,
            trajectory_weight=args.trajectory_weight,
        )

    if args.mode in ("both", "ablation"):
        results["ablation"], _ = run_one(
            "ablation (no aux)",
            latent_pred_weight=0.0,
            future_link_weight=0.0,
            potential_weight=0.0,
            trajectory_weight=0.0,
        )

    if args.mode == "both" and len(results) == 2:
        print("\n--- まとめ（final_val_auc） ---")
        print(f"  cope:     {results['cope']:.4f}")
        print(f"  ablation: {results['ablation']:.4f}")
        delta = results["cope"] - results["ablation"]
        print(f"  Δ (cope - ablation): {delta:+.4f}")


if __name__ == "__main__":
    main()
