#!/usr/bin/env python3
"""
最小実験: 主体×CPC 二部グラフ → UnifiedVGAETD（時間依存ポテンシャル Φ(z,y)）を
軽く学習し、潜在地形 ＋ 流れ場 −∇Φ ＋ 密度ホットスポットのインタラクティブ HTML を出力する。

「今後盛り上がる技術を潜在空間上で可視化する」ための土台。主体（企業 or 発明者）と
CPC を同一潜在空間に埋め込み、CPC が作るポテンシャル地形（谷＝集積する技術領域）の
上を主体が −∇Φ に沿って流れる描像を 1 枚の地図として描く。

例:
  # 企業×CPC（建設ドメイン, 2012-2020）
  python -m pnode_patent_runner.run_bipartite_landscape \\
      --csv data/processed/bipartite_construction_firm.csv \\
      --year-range 2012 2020 --min-events 20 --epochs 30 \\
      --actor-kind firm \\
      --output pnode_patent_runner/outputs/bipartite_landscape/map_construction_firm.html

  # 発明者×CPC（建設ドメイン）
  python -m pnode_patent_runner.run_bipartite_landscape \\
      --csv data/processed/bipartite_construction.csv \\
      --year-range 2012 2020 --min-events 30 --epochs 30 \\
      --actor-kind inventor \\
      --output pnode_patent_runner/outputs/bipartite_landscape/map_construction_inventor.html
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle
from pnode_patent_runner.unified_vgae_td import METHOD_SHORT_NAME_TD, UnifiedVGAETD
from pnode_patent_runner.unified_training_td import train_model_td
from pnode_patent_runner.interactive_landscape import build_interactive_payload_author_topic
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)
from pnode_patent_runner.interactive_landscape_vector_field_td import (
    compute_vector_field_for_plotly_td,
)


def parse_args():
    repo = _REPO_ROOT
    default_tpl = repo / "pnode_patent_runner/interactive_vector_field_alt_dark.html"
    p = argparse.ArgumentParser(description="二部グラフ Φ(z,y) 地形 + 流れ場 + ホットスポット HTML")
    p.add_argument("--csv", required=True, help="ts,u,i 形式の二部イベント CSV")
    p.add_argument("--year-range", nargs=2, type=int, metavar=("START", "END"), required=True)
    p.add_argument("--min-events", type=int, default=20, help="主体の最小イベント数（小さいほどノード増）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--actor-kind", type=str, default="firm",
                   choices=("firm", "inventor"), help="左パーティションの呼称（UI ラベル用）")
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="", help="学習済み重みの保存先（省略時は output と同階層）")
    p.add_argument("--field-resolution", type=int, default=48)
    p.add_argument("--field-margin", type=float, default=0.5)
    p.add_argument("--quiver-stride", type=int, default=3)
    p.add_argument("--quiver-length", type=float, default=1.0)
    p.add_argument("--n-mag-bins", type=int, default=6)
    p.add_argument("--phi-heatmap-style", type=str, default="valley_red_peak_blue")
    p.add_argument("--arrow-style", type=str, default="gray_grid",
                   choices=("magnitude", "gray_grid"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--html-template", type=str, default=str(default_tpl))
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()
    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    y0, y1 = sorted(args.year_range)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"CSV が見つかりません: {csv_path}")

    # ── 1. 二部グラフ bundle ──────────────────────────────────────────────
    print(f"Loading bipartite bundle: {csv_path.name}  years={y0}..{y1}  min_events={args.min_events}")
    bundle = load_bipartite_domain_graph_bundle(
        csv_path,
        min_events=args.min_events,
        year_min=y0,
        year_max=y1,
        year_range=(y0, y1),
    )
    years_list = sorted(bundle.graphs.keys())
    print(f"  actors(主体)={bundle.num_corps}  CPC(技術)={len(bundle.right_nodes)}  "
          f"total_nodes={bundle.total_n}  years={years_list}")

    # ── 2. モデル ─────────────────────────────────────────────────────────
    model = UnifiedVGAETD(
        num_nodes=bundle.total_n,
        num_corps=bundle.num_corps,
        input_dim=bundle.in_dim,
        year_min=y0,
        year_max=y1,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        initial_corp_vectors=bundle.init_vectors,
    ).to(device)

    # ── 3. 学習 ───────────────────────────────────────────────────────────
    print(f"Training UnifiedVGAETD for {args.epochs} epochs ...")
    _, _, best_auc, _ = train_model_td(
        model,
        bundle.graphs,
        bundle.num_corps,
        bundle.hist_edges,
        num_epochs=args.epochs,
        loss_aux_warmup_epochs=max(args.epochs // 4, 3),
    )
    print(f"  done. best_val_auc (max over epochs): {best_auc:.4f}")

    # ── 4. checkpoint 保存 ────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint) if args.checkpoint.strip() else out.with_suffix(".pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "year_min": y0,
            "year_max": y1,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
        },
        str(ckpt_path),
    )
    print(f"Saved checkpoint: {ckpt_path}")

    # ── 5. payload 用の CPC 説明 df（hover は CPC コードのみ）──────────────
    topics = list(bundle.right_nodes)
    topic_df = pd.DataFrame({"topic": [str(t) for t in topics]})

    # ── 6. 各年の地形 + 流れ場 + ホットスポット ───────────────────────────
    model.eval()
    by_year = {}
    for y in years_list:
        data_y = bundle.graphs[y].to(device)
        with torch.no_grad():
            z, _, _ = model.encode(data_y.x, data_y.edge_index)
            z_np = z.cpu().numpy()
        base = build_interactive_payload_author_topic(
            topic_df,
            bundle.graphs[y],
            bundle.num_corps,
            bundle.corps,
            topics,
            z_np,
            y,
            topic_column="topic",
        )
        with torch.enable_grad():
            vf = compute_vector_field_for_plotly_td(
                model,
                device,
                z_np,
                calendar_year=int(y),
                margin=args.field_margin,
                resolution=args.field_resolution,
                quiver_stride=args.quiver_stride,
                quiver_length_mult=args.quiver_length,
                n_mag_bins=args.n_mag_bins,
                phi_heatmap_style=args.phi_heatmap_style,
                arrow_style=args.arrow_style,
            )
        by_year[str(y)] = merge_payload_with_vector_field(base, vf)
        print(f"  year {y}: nodes={z_np.shape[0]} rendered")

    # ── 7. HTML 出力 ──────────────────────────────────────────────────────
    actor_label = "企業" if args.actor_kind == "firm" else "発明者"
    page_title = f"{METHOD_SHORT_NAME_TD}: {actor_label}×CPC Φ(z,y) map + field"
    heading = f"{actor_label}×CPC 二部グラフ — Φ(z, 年) ＋ −∇Φ ＋ 密度ホットスポット"
    tpl_opt = Path(args.html_template.strip()) if (args.html_template or "").strip() else None
    write_interactive_vector_field_html(
        by_year,
        out,
        str(max(years_list)),
        page_title=page_title,
        heading=heading,
        template_path=tpl_opt,
        ui=alt_dark_ui_labels("patent"),
    )
    print(f"Wrote: {out}")
    print(f"  埋め込み年: {years_list}（初期表示: {max(years_list)}）")


if __name__ == "__main__":
    main()
