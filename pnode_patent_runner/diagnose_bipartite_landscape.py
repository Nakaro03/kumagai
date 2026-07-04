#!/usr/bin/env python3
"""run_bipartite_landscape が学習した地形が「創発を読めるか」を定量点検する。

確認項目:
  (1) Φ(z, year) が年ごとに変化しているか（時間依存が学習されているか）
  (2) CPC の将来成長（翌年出願増）が、地形の谷の深さ／−∇Φ の向きと対応するか
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD

CSV = "data/processed/bipartite_construction_firm.csv"
CKPT = "pnode_patent_runner/outputs/bipartite_landscape/map_construction_firm.pt"
Y0, Y1, MIN_EVENTS = 2012, 2020, 20

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CKPT, map_location=dev)
bundle = load_bipartite_domain_graph_bundle(CSV, min_events=MIN_EVENTS,
                                            year_min=Y0, year_max=Y1, year_range=(Y0, Y1))
model = UnifiedVGAETD(num_nodes=bundle.total_n, num_corps=bundle.num_corps,
                      input_dim=bundle.in_dim, year_min=Y0, year_max=Y1,
                      hidden_dim=ck["hidden_dim"], latent_dim=ck["latent_dim"],
                      initial_corp_vectors=bundle.init_vectors).to(dev)
model.load_state_dict(ck["state_dict"])
model.eval()

A = bundle.num_corps                      # actor count; CPC nodes are indices [A:]
years = sorted(bundle.graphs.keys())
growth = bundle.topic_growth_by_year or {}

PN = model.temporal_predictor.potential_net   # TimeDependentPotentialNet

def phi(z, year):
    zt = torch.as_tensor(z, dtype=torch.float32, device=dev)
    yi = PN.year_tensor(int(year), zt.shape[0], dev)
    with torch.no_grad():
        out = PN(zt, yi)
    return out.squeeze(-1).cpu().numpy()

# encode each year
Z = {}
for y in years:
    g = bundle.graphs[y].to(dev)
    with torch.no_grad():
        z, _, _ = model.encode(g.x, g.edge_index)
    Z[y] = z.cpu().numpy()

# (1) time-dependence: same z grid, Φ across years
print("=== (1) Φ(z,year) の時間変化（固定グリッド上の平均Φ）===")
xs = np.linspace(-4, 4, 30)
gx, gy = np.meshgrid(xs, xs)
grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
prev = None
for y in years:
    p = phi(grid, y)
    d = "" if prev is None else f"  Δgrid_L2_vs_prev={np.linalg.norm(p-prev):.3f}"
    print(f"  {y}: meanΦ={p.mean():.3f} std={p.std():.3f}{d}")
    prev = p

# (2) growth vs valley-depth correlation, per year (CPC nodes only)
print("\n=== (2) CPC将来成長 g_j(year) と 谷の深さ(−Φ) の Spearman 相関 ===")
print("    （メモリの知見通り弱相関なら『予測』ではなく『記述/可視化』が妥当）")
for y in years:
    if y not in growth:
        continue
    zc = Z[y][A:]                          # CPC latent
    pc = phi(zc, y)
    g = growth[y].cpu().numpy()            # length == num CPC
    n = min(len(g), len(pc))
    # active CPC only (nonzero degree this year): use growth!=0 OR appears
    mask = np.isfinite(pc[:n]) & np.isfinite(g[:n])
    rho, p_ = spearmanr(-pc[:n][mask], g[:n][mask])  # -Φ = valley depth
    print(f"  {y}: n={mask.sum():5d}  Spearman(-Φ, growth)={rho:+.3f} (p={p_:.1e})")

print("\n注: 強相関でなくても可視化の価値は別軸（構造と力学の解釈可能な提示）。")
