#!/usr/bin/env python3
"""
著者–トピックデータで Dual-Force VGAE を学習・評価し、既存の P-NODE (B+D) ベンチ JSON と並べて比較する。

手続きは `run_benchmark_comparison` の future-link（最終2年）に揃える。
P-NODE の数値は再学習せず `--pnode-json`（既定: pnode_BD_vgae_compare の seed 一致）を読む。

例:
  python -m pnode_patent_runner.run_dual_force_vs_pnode_author_topic \\
    --epochs 10 --seed 42 \\
    --output-json pnode_patent_runner/outputs/dual_force_compare/author_topic_vs_pnode_seed42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch

from pnode_patent_runner.dual_force_data import load_dual_force_bundle
from pnode_patent_runner.dual_force_eval import evaluate_dual_force_future_link_metrics
from pnode_patent_runner.dual_force_training import train_dual_force
from pnode_patent_runner.dual_force_vgae import DualForceVGAE


def _load_pnode_row(path: Path, seed: int) -> Optional[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if int(raw.get("seed", -1)) != int(seed):
        return None
    for row in raw.get("results") or []:
        if row.get("key") == "pnode":
            return {
                "final_val_auc": row.get("final_val_auc"),
                "final_val_ap": row.get("final_val_ap"),
                "final_val_ece": row.get("final_val_ece"),
                "source_json": str(path),
            }
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Dual-Force vs P-NODE (著者–トピック, 同一データ手続き)"
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
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument(
        "--link-score-mode",
        type=str,
        default="distance",
        choices=("distance", "cosine"),
        help="P-NODE ベンチの `--cope-link-score` に合わせるなら distance",
    )
    p.add_argument(
        "--pnode-json",
        type=Path,
        default=None,
        help="P-NODE 行の比輯用。既定: seed 一致の pnode_BD_vgae_compare",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="比輯結果（dual_force 指標 + pnode 参照）",
    )
    p.add_argument(
        "--save-checkpoint",
        type=Path,
        default=None,
        help="学習後に DualForceVGAE 状態（state_dict 等）を .pt へ保存",
    )
    args = p.parse_args()

    torch.manual_seed(int(args.seed))

    pnode_default = (
        Path(__file__).resolve().parent
        / "outputs"
        / "pnode_BD_vgae_compare"
        / f"benchmark_pnode_BD_vs_baselines_seed{args.seed}.json"
    )
    pnode_path = args.pnode_json or pnode_default
    pnode_ref = None
    if pnode_path.is_file():
        pnode_ref = _load_pnode_row(pnode_path, int(args.seed))
    else:
        print(
            f"注意: P-NODE 参照 JSON が見つかりません: {pnode_path}",
            file=sys.stderr,
        )

    bundle = load_dual_force_bundle(
        args.data,
        topic_column=args.topic_column,
        min_papers=args.min_patents,
    )
    graphs = bundle.graphs
    years = sorted(graphs.keys())
    graphs_f = {y: g for y, g in graphs.items() if int(args.year_start) <= y <= int(args.year_end)}
    if len(graphs_f) < 2:
        raise SystemExit("年範囲内に2年以上必要です。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualForceVGAE(
        bundle.total_n,
        bundle.num_corps,
        bundle.in_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_author_vectors=bundle.init_vectors,
        gamma=args.gamma,
        link_score_mode=args.link_score_mode,
    ).to(device)

    for d in graphs_f.values():
        d.to(device)

    hist: Set[Tuple[int, int]] = bundle.hist_edges
    _ = train_dual_force(  # noqa: F841
        model, graphs_f, bundle.num_corps, hist, num_epochs=int(args.epochs), lr=float(args.lr)
    )

    metrics = evaluate_dual_force_future_link_metrics(
        model, graphs_f, bundle.num_corps, device
    )

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
                "gamma": float(args.gamma),
                "link_score_mode": str(args.link_score_mode),
            },
            str(sc),
        )
        print(f"Checkpoint: {sc}", file=sys.stderr)

    out: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_domain": "author_topic",
        "data": str(args.data),
        "year_range": [int(args.year_start), int(args.year_end)],
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "dual_force_config": {
            "hidden_dim": int(args.hidden_dim),
            "latent_dim": int(args.latent_dim),
            "gamma": float(args.gamma),
            "link_score_mode": args.link_score_mode,
        },
        "dual_force_final_val_auc": metrics.get("auc"),
        "dual_force_final_val_ap": metrics.get("ap"),
        "dual_force_final_val_ece": metrics.get("ece"),
        "pnode_b_plus_d_reference": pnode_ref,
        "note": (
            "P-NODE (B+D) 参照行は学習 epoch 等が本実行と異なる場合があります。"
            "厳密比較は同一 `epochs` でベンチを再実行し JSON を揃えてください。"
        ),
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    oj = args.output_json
    if oj is not None:
        oj = Path(oj)
        oj.parent.mkdir(parents=True, exist_ok=True)
        with open(oj, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {oj}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    main()
