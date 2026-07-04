#!/usr/bin/env python3
"""
PC-PNODE (PopulationCoupledPotentialNet) 学習済みチェックポイントから
インタラクティブ HTML（潜在マップ + Φ / −∇Φ）を生成する。

PC-PNODE の特徴:
  Φ_t(z) = ψ_θ(z) − w_ρ log ρ̂_t(z) − w_Δ Δ_t(z)
  - ψ_θ: 純粋ニューラルポテンシャル（谷=注目領域）
  - −w_ρ log ρ̂: 高密度領域を谷に → 著者が集まるほど引力が強まる
  - −w_Δ Δ_t: 成長中トピック周辺を谷に（L_trend で学習）

例:
  python -m pnode_patent_runner.run_benchmark_comparison \\
    --data-domain author_topic \\
    --year-range 2022 2025 --min-patents 5 \\
    --epochs 30 --seed 42 --methods pnode_pc \\
    --pnode-potential-feature mlp --trend-weight 0.1 \\
    --save-checkpoint-dir pnode_patent_runner/outputs/pnode_pc_landscape/ckpt

  python -m pnode_patent_runner.run_interactive_landscape_pnode_pc_vector_field \\
    --data-domain author_topic \\
    --year-range 2022 2025 --min-patents 5 \\
    --load-checkpoint pnode_patent_runner/outputs/pnode_pc_landscape/ckpt/pnode_pc_seed42.pt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.cope_experiment import (
    load_author_paper_graph_bundle,
    load_author_topic_graph_bundle,
    load_cope_graph_bundle,
)
from pnode_patent_runner.interactive_landscape import (
    build_interactive_payload,
    build_interactive_payload_author_paper,
    build_interactive_payload_author_topic,
)
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    compute_delta_phi_grid,
    compute_node_movement_arrows,
    compute_node_phi_density_grid,
    compute_vector_field_for_plotly,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)

METHOD_LABEL = "PC-PNODE"


def _default_csv(repo: Path, domain: str) -> Path:
    if domain == "patent":
        return repo / "notebooks/work/dataset/topic_info3.csv"
    for candidate in (
        repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        repo / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if candidate.is_file():
            return candidate
    return repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv"


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_out = repo / "pnode_patent_runner/outputs/pnode_pc_landscape/map_pnode_pc_author_topic.html"
    default_html_template = repo / "pnode_patent_runner/interactive_vector_field_alt_dark.html"

    parser = argparse.ArgumentParser(
        description=f"{METHOD_LABEL} — 潜在マップ + Φ / −∇Φ のインタラクティブ HTML",
    )
    parser.add_argument(
        "--data-domain",
        type=str,
        choices=("patent", "arxiv", "author_topic"),
        default="author_topic",
    )
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--output", type=str, default=str(default_out))
    parser.add_argument("--topic-column", type=str, default="topic")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--years", type=str, default="")
    parser.add_argument("--all-years", action="store_true")
    parser.add_argument("--year-range", nargs=2, type=int, metavar=("START", "END"), default=None)
    parser.add_argument("--min-patents", type=int, default=5)
    parser.add_argument("--arxiv-year-min", type=int, default=2020)
    parser.add_argument("--arxiv-year-max", type=int, default=2026)
    parser.add_argument("--arxiv-no-year-filter", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--load-checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-max-chars", type=int, default=450)
    parser.add_argument("--field-resolution", type=int, default=42)
    parser.add_argument("--field-margin", type=float, default=0.5)
    parser.add_argument("--quiver-stride", type=int, default=3)
    parser.add_argument("--quiver-length", type=float, default=1.75)
    parser.add_argument("--n-mag-bins", type=int, default=5)
    parser.add_argument(
        "--phi-heatmap-style", type=str,
        choices=("default", "valley_red_peak_blue"),
        default="valley_red_peak_blue",
    )
    parser.add_argument(
        "--arrow-style", type=str, choices=("magnitude", "gray_grid"), default="gray_grid",
    )
    parser.add_argument("--link-score", type=str, choices=("distance", "cosine"), default="distance")
    parser.add_argument("--cosine-logit-scale", type=float, default=5.0)
    parser.add_argument("--pnode-potential-feature", type=str, choices=("rff", "mlp"), default="mlp")
    parser.add_argument("--pnode-rff-frozen-basis", action="store_true")
    parser.add_argument("--pnode-history-len", type=int, default=1, metavar="K")
    parser.add_argument("--pnode-hist-fuse-mode", type=str, choices=("linear", "gru"), default="gru")
    parser.add_argument("--pnode-ode-method", type=str, default="dopri5",
                        choices=("dopri5", "rk4", "euler"))
    parser.add_argument("--pnode-ode-n-steps", type=int, default=4)
    parser.add_argument("--pc-w-rho-init", type=float, default=0.1)
    parser.add_argument("--pc-w-delta-init", type=float, default=0.1)
    parser.add_argument("--pc-log-bandwidth-init", type=float, default=0.0)
    parser.add_argument("--html-template", type=str, default=str(default_html_template))
    args = parser.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path((args.data or "").strip() or _default_csv(repo, args.data_domain))
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    ckpt_path = Path(args.load_checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw_ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
        hidden_dim = int(raw_ckpt.get("hidden_dim", args.hidden_dim))
        latent_dim = int(raw_ckpt.get("latent_dim", args.latent_dim))
        link_score = str(raw_ckpt.get("cope_link_score", args.link_score))
        cos_scale = float(raw_ckpt.get("cosine_logit_scale", args.cosine_logit_scale))
    else:
        state_dict = raw_ckpt
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim
        link_score = args.link_score
        cos_scale = args.cosine_logit_scale

    ymin: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_min
    ymax: Optional[int] = None if args.arxiv_no_year_filter else args.arxiv_year_max
    yr = tuple(args.year_range) if args.year_range is not None else None

    if args.data_domain == "patent":
        bundle = load_cope_graph_bundle(
            data_path, min_patents=args.min_patents, year_range=yr, years_csv=args.years,
            all_years=args.all_years,
        )
    elif args.data_domain == "arxiv":
        bundle = load_author_paper_graph_bundle(
            data_path, min_papers=args.min_patents, arxiv_year_min=ymin, arxiv_year_max=ymax,
            year_range=yr, years_csv=args.years, all_years=args.all_years,
        )
    else:
        bundle = load_author_topic_graph_bundle(
            data_path, min_papers=args.min_patents, topic_column=args.topic_column,
            arxiv_year_min=ymin, arxiv_year_max=ymax, year_range=yr,
            years_csv=args.years, all_years=args.all_years,
        )

    graphs = bundle.graphs
    num_left = bundle.num_corps
    years_list: List[int] = sorted(graphs.keys())

    default_year = str(args.year) if args.year is not None else str(max(years_list))
    if args.year is not None and args.year not in graphs:
        raise SystemExit(f"--year {args.year} が束にありません。利用可能: {years_list}")

    print(
        f"data_domain={args.data_domain}, data={data_path}\n"
        f"グラフ年: {years_list}, 左={num_left}, N={bundle.total_n}, in_dim={bundle.in_dim}\n"
        f"PC-PNODE config: Φ={args.pnode_potential_feature}, K={args.pnode_history_len}"
    )

    # checkpoint に trend_adapter があれば A+BEF アーキテクチャを有効化
    _ckpt_state = raw_ckpt.get("state_dict", raw_ckpt) if isinstance(raw_ckpt, dict) else {}
    has_adapter = any(str(k).startswith("trend_adapter.") for k in _ckpt_state.keys())
    model = BenchmarkTemporalVGAE(
        num_nodes=bundle.total_n,
        num_corps=num_left,
        input_dim=bundle.in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        initial_corp_vectors=bundle.init_vectors,
        link_score_mode=link_score,
        cosine_logit_scale=cos_scale,
        variant="pnode_pc",
        pnode_history_len=int(args.pnode_history_len),
        pnode_potential_feature=str(args.pnode_potential_feature),
        pnode_rff_frozen_basis=bool(args.pnode_rff_frozen_basis),
        pnode_hist_fuse_mode=str(args.pnode_hist_fuse_mode),
        pnode_ode_method=str(args.pnode_ode_method),
        pnode_ode_n_steps=int(args.pnode_ode_n_steps),
        pc_w_rho_init=float(args.pc_w_rho_init),
        pc_w_delta_init=float(args.pc_w_delta_init),
        pc_log_bandwidth_init=float(args.pc_log_bandwidth_init),
        year_min=years_list[0],
        year_max=years_list[-1],
        topic_position_embedding=has_adapter,
    ).to(device)
    if has_adapter:
        print("[info] checkpoint に trend_adapter 検出 → Architecture A 有効化")

    skip_log, _ = load_state_dict_skip_shape_mismatch(model, state_dict)
    if skip_log:
        print("チェックポイント読込（一部スキップ）:")
        for line in skip_log[:20]:
            print("  ", line)
        if len(skip_log) > 20:
            print(f"  ... 他 {len(skip_log) - 20} 行")
    model.eval()

    # 学習済み w_rho, w_delta を表示
    pot = model.temporal_predictor.potential_net
    if hasattr(pot, "w_rho"):
        print(f"  w_rho={float(pot.w_rho):.4f}  w_delta={float(pot.w_delta):.4f}")
        bw = float(torch.exp(pot.kde_field.log_bandwidth)) if hasattr(pot, "kde_field") else None
        if bw is not None:
            print(f"  KDE bandwidth={bw:.4f}")

    # topic_growth_by_year から成長率を取得
    growth_by_year = bundle.topic_growth_by_year or {}

    hist_K = model.temporal_history_len
    year_z: dict = {}
    by_year: dict = {}

    for y in years_list:
        data_y = graphs[y].to(device)

        # PC-PNODE: population を set してから encode
        with torch.no_grad():
            z, _, _ = model.encode(data_y.x, data_y.edge_index)
            z_np = z.cpu().numpy()

        # 現年の population を potential_net に設定（密度場の更新）
        if hasattr(pot, "set_population"):
            pot.set_population(z.detach())

        year_z[y] = z.detach()

        if args.data_domain == "patent":
            base = build_interactive_payload(
                bundle.dataframe, graphs[y], num_left, bundle.corps, bundle.right_nodes,
                z_np, y, summary_max=args.summary_max_chars,
            )
        elif args.data_domain == "arxiv":
            base = build_interactive_payload_author_paper(
                bundle.dataframe, graphs[y], num_left, bundle.corps, bundle.right_nodes,
                z_np, y, summary_max=args.summary_max_chars,
            )
        else:
            base = build_interactive_payload_author_topic(
                bundle.dataframe, graphs[y], num_left, bundle.corps, bundle.right_nodes,
                z_np, y, topic_column=args.topic_column, summary_max=args.summary_max_chars,
            )

        # トピック（corporations = right nodes）に成長率を付加
        growth_t = growth_by_year.get(y)
        if growth_t is not None:
            g_np = growth_t.numpy()
            for i, corp in enumerate(base["corporations"]):
                if i < len(g_np):
                    corp["growthRate"] = float(g_np[i])
                else:
                    corp["growthRate"] = None
        else:
            for corp in base["corporations"]:
                corp["growthRate"] = None

        with torch.enable_grad():
            vf = compute_vector_field_for_plotly(
                model,
                device,
                z_np,
                margin=args.field_margin,
                resolution=args.field_resolution,
                quiver_stride=args.quiver_stride,
                quiver_length_mult=args.quiver_length,
                n_mag_bins=args.n_mag_bins,
                phi_heatmap_style=args.phi_heatmap_style,
                arrow_style=args.arrow_style,
            )

        # Φ 値をノードに付加
        all_coords = (
            [[p["x"], p["y"]] for p in base["patents"]]
            + [[c["x"], c["y"]] for c in base["corporations"]]
        )
        if all_coords:
            z_t = torch.tensor(all_coords, dtype=torch.float32, device=device)
            with torch.no_grad():
                phi_vals = pot(z_t).squeeze(-1).cpu().numpy()
            n_pat = len(base["patents"])
            for i, entry in enumerate(base["patents"]):
                entry["phi"] = float(phi_vals[i])
            for i, entry in enumerate(base["corporations"]):
                entry["phi"] = float(phi_vals[n_pat + i])

        by_year[str(y)] = merge_payload_with_vector_field(base, vf)

    sorted_years = sorted(by_year.keys(), key=int)

    # ΔΦ（年間変化）
    for idx, y_str in enumerate(sorted_years):
        payload = by_year[y_str]
        prev_y_str = sorted_years[idx - 1] if idx > 0 else None

        if prev_y_str is None:
            for entry in payload["patents"] + payload["corporations"]:
                entry["deltaPhi"] = None
            continue

        prev = by_year[prev_y_str]
        prev_pat_map = {p["id"]: p for p in prev["patents"]}
        prev_corp_map = {c["name"]: c for c in prev["corporations"]}

        for entry in payload["patents"]:
            pid = entry["id"]
            if pid in prev_pat_map and "phi" in entry and "phi" in prev_pat_map[pid]:
                entry["deltaPhi"] = entry["phi"] - prev_pat_map[pid]["phi"]
            else:
                entry["deltaPhi"] = None

        for entry in payload["corporations"]:
            name = entry["name"]
            if name in prev_corp_map and "phi" in entry and "phi" in prev_corp_map[name]:
                entry["deltaPhi"] = entry["phi"] - prev_corp_map[name]["phi"]
            else:
                entry["deltaPhi"] = None

        dphi_nodes_xy = []
        dphi_nodes_val = []
        for entry in payload["patents"] + payload["corporations"]:
            if entry.get("deltaPhi") is not None:
                dphi_nodes_xy.append([entry["x"], entry["y"]])
                dphi_nodes_val.append(entry["deltaPhi"])

        vf = payload["vectorField"]
        if dphi_nodes_xy and "heatmap" in vf:
            nxy = np.array(dphi_nodes_xy, dtype=np.float32)
            dphi_arr = np.array(dphi_nodes_val, dtype=np.float32)
            xc = np.array(vf["heatmap"]["x"], dtype=np.float32)
            yc = np.array(vf["heatmap"]["y"], dtype=np.float32)
            xs = float(xc[-1] - xc[0]) if len(xc) > 1 else 1.0
            ys = float(yc[-1] - yc[0]) if len(yc) > 1 else 1.0
            bw = 0.08 * (xs + ys) / 2.0
            dphi_grid = compute_delta_phi_grid(nxy, dphi_arr, xc, yc, bandwidth=bw)
            dp_max = float(np.abs(dphi_grid).max()) or 1.0
            vf["deltaPhiHeatmap"] = {
                "x": vf["heatmap"]["x"],
                "y": vf["heatmap"]["y"],
                "z": (dphi_grid / dp_max).tolist(),
            }
            vf["meta"]["deltaPhiMax"] = dp_max

        move_pat = compute_node_movement_arrows(payload["patents"], prev_pat_map, key_field="id")
        move_corp = compute_node_movement_arrows(payload["corporations"], prev_corp_map, key_field="name")
        all_xl = move_pat["xl"] + move_corp["xl"]
        all_yl = move_pat["yl"] + move_corp["yl"]
        if all_xl:
            vf["movementArrows"] = {"xl": all_xl, "yl": all_yl}

    print(f"ΔΦ computed for {len(sorted_years) - 1} year transitions")

    # ノード密度×Φ ヒートマップ
    for y_str in sorted_years:
        payload = by_year[y_str]
        vf = payload["vectorField"]
        node_xy_list, node_phi_list = [], []
        for entry in payload["patents"] + payload["corporations"]:
            if entry.get("phi") is not None:
                node_xy_list.append([entry["x"], entry["y"]])
                node_phi_list.append(entry["phi"])
        if node_xy_list and "heatmap" in vf:
            nxy = np.array(node_xy_list, dtype=np.float32)
            nphi = np.array(node_phi_list, dtype=np.float32)
            xc = np.array(vf["heatmap"]["x"], dtype=np.float32)
            yc = np.array(vf["heatmap"]["y"], dtype=np.float32)
            xs = float(xc[-1] - xc[0]) if len(xc) > 1 else 1.0
            ys = float(yc[-1] - yc[0]) if len(yc) > 1 else 1.0
            bw = 0.06 * (xs + ys) / 2.0
            phi_grid, _ = compute_node_phi_density_grid(nxy, nphi, xc, yc, bandwidth=bw, density_floor_pct=15.0)
            phi_z = []
            for j in range(phi_grid.shape[1]):
                row = []
                for i in range(phi_grid.shape[0]):
                    v = phi_grid[i, j]
                    row.append(None if np.isnan(v) else float(v))
                phi_z.append(row)
            vf["nodePhiDensity"] = {"x": vf["heatmap"]["x"], "y": vf["heatmap"]["y"], "z": phi_z}
    print("Node Φ density grid computed")

    # 予測ΔΦ（ODE ロールアウト: z_t → z_{t+1}）
    for y_idx, y in enumerate(years_list):
        y_str = str(y)
        payload = by_year[y_str]

        z_history: List[torch.Tensor] = []
        for ki in range(hist_K):
            h_idx = max(0, y_idx - hist_K + 1 + ki)
            z_history.append(year_z[years_list[h_idx]])

        with torch.no_grad():
            z_pred = model.predict_future(z_history)
            phi_pred_all = pot(z_pred).squeeze(-1).cpu().numpy()
            phi_curr_all = pot(year_z[y]).squeeze(-1).cpu().numpy()

        pred_dphi_all = phi_pred_all - phi_curr_all
        z_pred_np = z_pred.cpu().numpy()
        z_curr_np = year_z[y].cpu().numpy()

        coord_to_idx: dict = {}
        for ni in range(z_curr_np.shape[0]):
            coord_to_idx[(float(z_curr_np[ni, 0]), float(z_curr_np[ni, 1]))] = ni

        pred_label = str(y + 1)
        for entry in payload["patents"] + payload["corporations"]:
            ni = coord_to_idx.get((entry["x"], entry["y"]))
            if ni is not None:
                entry["predDeltaPhi"] = float(pred_dphi_all[ni])
                entry["predX"] = float(z_pred_np[ni, 0])
                entry["predY"] = float(z_pred_np[ni, 1])
                entry["predYear"] = pred_label

        vf = payload["vectorField"]
        dphi_pred_nodes = [
            (e["x"], e["y"], e["predDeltaPhi"])
            for e in payload["patents"] + payload["corporations"]
            if e.get("predDeltaPhi") is not None
        ]
        if dphi_pred_nodes and "heatmap" in vf:
            nxy = np.array([(n[0], n[1]) for n in dphi_pred_nodes], dtype=np.float32)
            dpv = np.array([n[2] for n in dphi_pred_nodes], dtype=np.float32)
            xc = np.array(vf["heatmap"]["x"], dtype=np.float32)
            yc = np.array(vf["heatmap"]["y"], dtype=np.float32)
            xs = float(xc[-1] - xc[0]) if len(xc) > 1 else 1.0
            ys = float(yc[-1] - yc[0]) if len(yc) > 1 else 1.0
            bw = 0.08 * (xs + ys) / 2.0
            pg = compute_delta_phi_grid(nxy, dpv, xc, yc, bandwidth=bw)
            pm = float(np.abs(pg).max()) or 1.0
            vf["predDeltaPhiHeatmap"] = {
                "x": vf["heatmap"]["x"],
                "y": vf["heatmap"]["y"],
                "z": (pg / pm).tolist(),
            }
            vf["meta"]["predDeltaPhiMax"] = pm

        pxl, pyl = [], []
        for entry in payload["patents"] + payload["corporations"]:
            px, py = entry.get("predX"), entry.get("predY")
            if px is not None and py is not None:
                dx, dy = px - entry["x"], py - entry["y"]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    pxl.extend([entry["x"], px, None])
                    pyl.extend([entry["y"], py, None])
        if pxl:
            vf["predMovementArrows"] = {"xl": pxl, "yl": pyl}

    print(f"Predicted ΔΦ computed for {len(years_list)} years")

    # HTML 出力
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    domain_label = {"patent": "patent", "arxiv": "author_paper", "author_topic": "author_topic"}[args.data_domain]
    tpl_opt = Path(args.html_template.strip()) if (args.html_template or "").strip() else None

    write_interactive_vector_field_html(
        by_year,
        out,
        default_year,
        page_title=f"{METHOD_LABEL}: 技術トレンド予測マップ",
        heading=f"{METHOD_LABEL} — Φ 景観（谷=注目集中, 山=離散）+ −∇Φ ベクトル場",
        template_path=tpl_opt,
        ui=alt_dark_ui_labels(domain_label),
    )

    print(f"\nWrote: {out}")
    print(f"  表示年: {years_list}（初期: {default_year}）")
    if hasattr(pot, "w_rho"):
        print(f"  学習済み w_rho={float(pot.w_rho):.4f}  w_delta={float(pot.w_delta):.4f}")
    for y in years_list:
        p = by_year[str(y)]
        g_vals = [c.get("growthRate") for c in p["corporations"] if c.get("growthRate") is not None]
        g_info = f"  成長率 [{min(g_vals):.2f}, {max(g_vals):.2f}]" if g_vals else ""
        print(f"    {y}: 著者={len(p['patents'])} トピック={len(p['corporations'])}{g_info}")


if __name__ == "__main__":
    main()
