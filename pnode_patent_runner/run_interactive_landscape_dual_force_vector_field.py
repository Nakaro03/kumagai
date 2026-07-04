#!/usr/bin/env python3
"""
Dual-Force 学習済み checkpoint から、alt_dark テンプレでインタラクティブ HTML
（潜在マップ + |v| ヒート・等高線 + フロー v 矢印）を出力する。

スカラーポテンシャル Φ / −∇Φ ではなく、``DualForcePotentialODEFunc`` の速度場 v(z) を可視化する。

チェックポイント例:
  python -m pnode_patent_runner.run_dual_force_vs_pnode_author_topic \\
    --epochs 10 --seed 42 --save-checkpoint pnode_patent_runner/outputs/dual_force/ckpt/dual_force_seed42.pt

  python -m pnode_patent_runner.run_interactive_landscape_dual_force_vector_field \\
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \\
    --year-range 2022 2025 --min-patents 5 \\
    --load-checkpoint pnode_patent_runner/outputs/dual_force/ckpt/dual_force_seed42.pt
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

from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.dual_force_data import load_dual_force_bundle
from pnode_patent_runner.dual_force_models import DualForcePotentialODEFunc
from pnode_patent_runner.dual_force_vgae import DualForceVGAE
from pnode_patent_runner.interactive_landscape import build_interactive_payload_author_topic
from pnode_patent_runner.interactive_landscape_vector_field import (
    alt_dark_ui_labels,
    compute_dual_force_node_speeds,
    compute_dual_force_vector_field_for_plotly,
    compute_node_phi_density_grid,
    merge_payload_with_vector_field,
    write_interactive_vector_field_html,
)

METHOD_LABEL = "Dual-Force"


def _default_csv(repo: Path) -> Path:
    for candidate in (
        repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv",
        repo / "data/processed/arxiv_cs_embedded_2020-2026.csv",
    ):
        if candidate.is_file():
            return candidate
    return repo / "data/processed/arxiv_cs_embedded_2020-2026_full.csv"


def _dual_force_ui_overrides() -> dict:
    u = alt_dark_ui_labels("author_topic")
    u["nodeScalarLabel"] = "|v|（ODE 速度の大きさ）"
    u["toggleHeat"] = "|v| ヒートマップ"
    u["toggleContour"] = "|v| 等高線"
    u["toggleArrows"] = "フロー v 短矢印"
    u["toggleNodePhiD"] = "ノード密度×|v|（赤=低速集中・青=高速集中）"
    u["hintHtml"] = (
        "著者–トピック＋Dual-Force。背景のスカラーは|v|、矢印はODEのベクトル場v。"
        "スカラーポテンシャルΦは定義しません。チェックポイントは <code>--save-checkpoint</code> 出力を使ってください。"
    )
    return u


def _restore_ode_p_proj_from_checkpoint(
    ode: DualForcePotentialODEFunc,
    state_dict: dict,
    target_device: torch.device,
) -> bool:
    """
    DualForcePotentialODEFunc は ``P_proj`` を第1 forward で lazy 生成する。
    学習後の state_dict には ``P_proj.weight`` が入るが、新規 ``DualForceVGAE`` には
    当該サブモジュールが存在しないため、手で ``nn.Linear`` を割り当てて読み込む。
    """
    if not isinstance(ode, DualForcePotentialODEFunc):
        return False
    key = "temporal_predictor.ode_func.P_proj.weight"
    w = state_dict.get(key)
    if w is None or not torch.is_tensor(w):
        return False
    w = w.to(device=target_device, dtype=torch.float32)
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    p = torch.nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        p.weight.copy_(w)
    p.to(target_device, dtype=torch.float32)
    ode.P_proj = p
    return True


def _assign_node_phi(
    base: dict,
    bundle: object,
    speeds_cpu: np.ndarray,
) -> None:
    """グラフ節点インデックスに従い JSON の patents / corporations に phi（|v|）を入れる。"""
    num_authors = int(bundle.num_corps)
    authors: List = list(bundle.corps)
    topics: List = list(bundle.right_nodes)
    topic_to_gi = {str(topics[i]): num_authors + i for i in range(len(topics))}
    author_to_gi = {authors[i]: i for i in range(len(authors))}

    for p in base["patents"]:
        gi = topic_to_gi.get(str(p["id"]))
        if gi is not None and gi < len(speeds_cpu):
            p["phi"] = float(speeds_cpu[gi])
    for c in base["corporations"]:
        gi = author_to_gi.get(c["name"])
        if gi is not None and gi < len(speeds_cpu):
            c["phi"] = float(speeds_cpu[gi])


def main() -> None:
    warnings.filterwarnings("ignore")

    repo = _REPO_ROOT
    default_out = repo / "pnode_patent_runner/outputs/dual_force_landscape/map_dual_force_alt_dark.html"
    default_tpl = repo / "pnode_patent_runner/interactive_vector_field_alt_dark.html"

    ap = argparse.ArgumentParser(
        description=f"{METHOD_LABEL} — 潜在マップ + |v| / フロー v のインタラクティブ HTML"
    )
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--output", type=str, default=str(default_out))
    ap.add_argument("--topic-column", type=str, default="topic")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument(
        "--year-range", nargs=2, type=int, metavar=("START", "END"), default=None
    )
    ap.add_argument("--min-patents", type=int, default=5)
    ap.add_argument(
        "--arxiv-year-min", type=int, default=2020
    )
    ap.add_argument(
        "--arxiv-year-max", type=int, default=2026
    )
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=2)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument(
        "--link-score-mode", type=str, default="distance", choices=("distance", "cosine")
    )
    ap.add_argument("--load-checkpoint", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-max-chars", type=int, default=450)
    ap.add_argument("--field-resolution", type=int, default=42)
    ap.add_argument("--field-margin", type=float, default=0.5)
    ap.add_argument("--quiver-stride", type=int, default=3)
    ap.add_argument("--quiver-length", type=float, default=1.75)
    ap.add_argument("--n-mag-bins", type=int, default=5)
    ap.add_argument(
        "--phi-heatmap-style", type=str,
        choices=("default", "valley_red_peak_blue"),
        default="valley_red_peak_blue",
    )
    ap.add_argument(
        "--arrow-style", type=str,
        choices=("magnitude", "gray_grid"),
        default="gray_grid",
    )
    ap.add_argument("--html-template", type=str, default=str(default_tpl))
    args = ap.parse_args()

    if args.latent_dim != 2:
        raise SystemExit("ベクトル場 HTML は latent_dim=2 のみ対応です。")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path((args.data or "").strip() or _default_csv(repo))
    if not data_path.is_file():
        raise SystemExit(f"データが見つかりません: {data_path}")

    ckpt_path = Path(args.load_checkpoint)
    if not ckpt_path.is_file():
        raise SystemExit(f"チェックポイントが見つかりません: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw = torch.load(str(ckpt_path), map_location=device)
    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        hidden_dim = int(raw.get("hidden_dim", args.hidden_dim))
        latent_dim = int(raw.get("latent_dim", args.latent_dim))
        gamma = float(raw.get("gamma", args.gamma))
        in_dim = int(raw.get("input_dim", 0)) or None
        lsm = str(raw.get("link_score_mode", args.link_score_mode))
    else:
        state_dict = raw
        hidden_dim = args.hidden_dim
        latent_dim = args.latent_dim
        gamma = args.gamma
        in_dim = None
        lsm = args.link_score_mode

    bundle = load_dual_force_bundle(
        str(data_path),
        topic_column=args.topic_column,
        min_papers=args.min_patents,
    )
    graphs = bundle.graphs
    years_list: List[int] = sorted(graphs.keys())
    yr = tuple(args.year_range) if args.year_range is not None else None
    if yr is not None:
        y0, y1 = int(yr[0]), int(yr[1])
        years_list = [y for y in years_list if y0 <= y <= y1]
    if not years_list:
        raise SystemExit("年リストが空です。--year-range を確認してください。")

    if in_dim is None or in_dim != bundle.in_dim:
        in_dim = bundle.in_dim

    if args.year is not None:
        if args.year not in graphs:
            raise SystemExit(f"--year {args.year} が束にありません: {list(graphs.keys())}")
        default_year = str(args.year)
    else:
        default_year = str(max(years_list))

    print(
        f"data={data_path}, years={years_list}, authors={bundle.num_corps}, "
        f"N={bundle.total_n}, in_dim={in_dim}"
    )

    model = DualForceVGAE(
        bundle.total_n,
        bundle.num_corps,
        in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        initial_author_vectors=bundle.init_vectors,
        gamma=gamma,
        link_score_mode=lsm,
    ).to(device)
    # ckpt には学習中に埋まった実行用バッファ（P_j 等）や lazy な P_proj が入る。shape が空モデルと合わないためスキップし、P_proj だけ手復元。
    skip_log, _ = load_state_dict_skip_shape_mismatch(model, state_dict)
    _restore_ode_p_proj_from_checkpoint(
        model.temporal_predictor.ode_func, state_dict, target_device=device,
    )
    if skip_log:
        print("チェックポイント読込（一部スキップ）:")
        for line in skip_log[:24]:
            print("  ", line)
        if len(skip_log) > 24:
            print(f"  ... 他 {len(skip_log) - 24} 行")
    model.eval()

    df = bundle.dataframe
    rights: List = bundle.right_nodes
    by_year: dict = {}

    for y in years_list:
        data_y = graphs[y].to(device)
        with torch.no_grad():
            z, _, _ = model.encode(data_y.x, data_y.edge_index)
            z_np = z.cpu().numpy()
        sp = compute_dual_force_node_speeds(model, data_y, device, z)
        speeds_cpu = sp.cpu().numpy()

        base = build_interactive_payload_author_topic(
            df, graphs[y], bundle.num_corps, bundle.corps, rights,
            z_np, y, topic_column=args.topic_column, summary_max=args.summary_max_chars,
        )
        _assign_node_phi(base, bundle, speeds_cpu)

        vf = compute_dual_force_vector_field_for_plotly(
            model, data_y, device, z_np,
            margin=args.field_margin,
            resolution=args.field_resolution,
            quiver_stride=args.quiver_stride,
            quiver_length_mult=args.quiver_length,
            n_mag_bins=args.n_mag_bins,
            phi_heatmap_style=args.phi_heatmap_style,
            arrow_style=args.arrow_style,
        )
        by_year[str(y)] = merge_payload_with_vector_field(base, vf)

    # ノード密度 × |v|（phi 欄の値を利用）
    for y_str in sorted(by_year.keys(), key=int):
        payload = by_year[y_str]
        vf = payload["vectorField"]
        node_xy_list: List = []
        node_s_list: List = []
        for entry in payload["patents"] + payload["corporations"]:
            if entry.get("phi") is not None:
                node_xy_list.append([entry["x"], entry["y"]])
                node_s_list.append(entry["phi"])
        if node_xy_list and "heatmap" in vf:
            nxy = np.array(node_xy_list, dtype=np.float32)
            nphi = np.array(node_s_list, dtype=np.float32)
            xc = np.array(vf["heatmap"]["x"], dtype=np.float32)
            yc = np.array(vf["heatmap"]["y"], dtype=np.float32)
            xs = float(xc[-1] - xc[0]) if len(xc) > 1 else 1.0
            ys = float(yc[-1] - yc[0]) if len(yc) > 1 else 1.0
            bw = 0.06 * (xs + ys) / 2.0
            phi_grid, _ = compute_node_phi_density_grid(
                nxy, nphi, xc, yc, bandwidth=bw, density_floor_pct=15.0,
            )
            phi_z: List = []
            for j in range(phi_grid.shape[1]):
                row: List = []
                for i in range(phi_grid.shape[0]):
                    v = phi_grid[i, j]
                    row.append(None if np.isnan(v) else float(v))
                phi_z.append(row)
            vf["nodePhiDensity"] = {
                "x": vf["heatmap"]["x"],
                "y": vf["heatmap"]["y"],
                "z": phi_z,
            }
            if vf.get("meta") is not None:
                vf["meta"]["nodePhiDensityColorbarTitle"] = "ノード|v|（赤=低速・青=高速）"
                vf["meta"]["nodePhiDensityTraceName"] = "ノード密度×|v|"
    print("Node |v| density grid computed (optional layer).")

    out = Path(args.output)
    tpl_opt = (
        Path(args.html_template.strip())
        if (args.html_template or "").strip()
        else default_tpl
    )
    page_title = f"{METHOD_LABEL}: latent map + |v| / flow v"
    heading = f"{METHOD_LABEL} — 潜在マップ ＋ |v|（ヒート・等高線）＋ フロー v 矢印"

    write_interactive_vector_field_html(
        by_year,
        out,
        default_year,
        page_title=page_title,
        heading=heading,
        template_path=tpl_opt,
        ui=_dual_force_ui_overrides(),
    )
    print(f"Wrote: {out}")
    print(f"  年: {years_list}（初期表示: {default_year}）")


if __name__ == "__main__":
    main()
