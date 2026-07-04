#!/usr/bin/env python3
"""(a)成長オーバーレイ と (b)成長考慮Φ学習 を比較する。

問い: Φ の谷を成長に揃えるよう明示的に学習したとき、その整合は
      hold-out 年（未学習）にも汎化するか? しなければ「記述」確定。

設計:
  - グラフは TRAIN_YEARS のみで学習、HOLDOUT_YEARS は完全に未使用。
  - モデルA(structure): 標準 TD 損失のみ。
  - モデルB(growth-aware): 標準 TD 損失 + corr-align 損失(−Φ(z,Y) を growth[Y] に揃える)。
  - 指標: 各年で Spearman(−Φ, growth)（active CPC のみ）。train vs holdout を比較。
  - 出力: 比較表(stdout) と オーバーレイ PNG(A/B × holdout年)。
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pnode_patent_runner.cope_experiment import load_bipartite_domain_graph_bundle
from pnode_patent_runner.unified_vgae_td import UnifiedVGAETD
from pnode_patent_runner.unified_training_td import compute_loss_standardized_td

CSV = "data/processed/bipartite_construction_firm.csv"
Y0, Y1, MIN_EVENTS = 2012, 2020, 20
TRAIN_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
HOLDOUT_YEARS = [2019, 2020]
EPOCHS = 60
ALIGN_W = 1.0
OUTDIR = Path("pnode_patent_runner/outputs/bipartite_landscape")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_bundle():
    return load_bipartite_domain_graph_bundle(
        CSV, min_events=MIN_EVENTS, year_min=Y0, year_max=Y1, year_range=(Y0, Y1))


def new_model(bundle):
    torch.manual_seed(42)
    return UnifiedVGAETD(
        num_nodes=bundle.total_n, num_corps=bundle.num_corps, input_dim=bundle.in_dim,
        year_min=Y0, year_max=Y1, hidden_dim=128, latent_dim=2,
        initial_corp_vectors=bundle.init_vectors).to(dev)


def phi_of(model, z, year):
    PN = model.temporal_predictor.potential_net
    zt = torch.as_tensor(z, dtype=torch.float32, device=dev)
    yi = PN.year_tensor(int(year), zt.shape[0], dev)
    return PN(zt, yi).squeeze(-1)


def active_cpc_mask(bundle, year):
    """その年に出現した CPC（右ノード）の index（CPC配列内, 0..K-1）。"""
    A = bundle.num_corps
    ei = bundle.graphs[year].edge_index.cpu().numpy()
    cpc_idx = set()
    for a, b in zip(ei[0], ei[1]):
        if a >= A: cpc_idx.add(int(a) - A)
        if b >= A: cpc_idx.add(int(b) - A)
    m = np.zeros(len(bundle.right_nodes), dtype=bool)
    for i in cpc_idx:
        if i < len(m): m[i] = True
    return m


def encode_year(model, bundle, year):
    g = bundle.graphs[year].to(dev)
    z, _, _ = model.encode(g.x, g.edge_index)
    return z


def eval_spearman(model, bundle):
    growth = bundle.topic_growth_by_year or {}
    A = bundle.num_corps
    model.eval()
    rows = {}
    with torch.no_grad():
        for y in sorted(bundle.graphs.keys()):
            if y not in growth:
                continue
            z = encode_year(model, bundle, y)
            zc = z[A:].cpu().numpy()
            pc = phi_of(model, zc, y).cpu().numpy()
            g = growth[y].cpu().numpy()
            n = min(len(g), len(pc))
            m = active_cpc_mask(bundle, y)[:n]
            if m.sum() < 20:
                continue
            rho, _ = spearmanr(-pc[:n][m], g[:n][m])
            rows[y] = (rho, int(m.sum()))
    return rows


def train(model, bundle, growth_aware: bool):
    growth = bundle.topic_growth_by_year or {}
    A = bundle.num_corps
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_kw = dict()  # README defaults
    pairs = list(zip(TRAIN_YEARS[:-1], TRAIN_YEARS[1:]))
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0; al = 0.0
        for y0, y1 in pairs:
            data_t = bundle.graphs[y0].to(dev)
            data_t1 = bundle.graphs[y1].to(dev)
            z_t, mu_t, logvar_t = model.encode(data_t.x, data_t.edge_index)
            opt.zero_grad()
            loss, _ = compute_loss_standardized_td(
                model, data_t, data_t1, bundle.num_corps, [z_t], bundle.hist_edges,
                y0, y1, precomputed_z_mu_logvar=(z_t, mu_t, logvar_t), **loss_kw)
            if growth_aware and y0 in growth:
                zc = z_t[A:]
                pc = phi_of(model, zc, y0)             # Φ on CPC, differentiable
                g = growth[y0].to(dev)[:zc.shape[0]]
                m = torch.as_tensor(active_cpc_mask(bundle, y0)[:zc.shape[0]], device=dev)
                if m.sum() > 20:
                    x = (-pc[m]); t = g[m]
                    x = (x - x.mean()) / (x.std() + 1e-6)
                    t = (t - t.mean()) / (t.std() + 1e-6)
                    corr = (x * t).mean()              # Pearson(-Φ, growth)
                    align = -corr * ALIGN_W
                    loss = loss + align
                    al += float(align)
            loss.backward(); opt.step()
            tot += float(loss)
        if ep % 15 == 0 or ep == EPOCHS - 1:
            print(f"    ep{ep:02d} total={tot/len(pairs):.3f} align={al/len(pairs):+.3f}")
    return model


def overlay_png(model, bundle, year, path, title):
    A = bundle.num_corps
    growth = (bundle.topic_growth_by_year or {})[year].cpu().numpy()
    model.eval()
    with torch.no_grad():
        z = encode_year(model, bundle, year)
        zc = z[A:].cpu().numpy()
    n = min(len(growth), len(zc))
    m = active_cpc_mask(bundle, year)[:n]
    zc = zc[:n][m]; gg = growth[:n][m]
    # Φ contour
    pad = 0.5
    xr = (zc[:, 0].min() - pad, zc[:, 0].max() + pad)
    yr = (zc[:, 1].min() - pad, zc[:, 1].max() + pad)
    gx, gy = np.meshgrid(np.linspace(*xr, 80), np.linspace(*yr, 80))
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    with torch.no_grad():
        pg = phi_of(model, grid, year).cpu().numpy().reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.contourf(gx, gy, -pg, levels=25, cmap="Greys", alpha=0.6)  # 谷=明
    c = np.clip(gg, np.percentile(gg, 5), np.percentile(gg, 95))
    sc = ax.scatter(zc[:, 0], zc[:, 1], c=c, cmap="hot", s=14, edgecolor="k", linewidth=0.2)
    plt.colorbar(sc, label=f"CPC growth into {year}")
    ax.set_title(title)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"    wrote {path}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    print(f"bundle: firms={bundle.num_corps} CPC={len(bundle.right_nodes)} "
          f"TRAIN={TRAIN_YEARS} HOLDOUT={HOLDOUT_YEARS}")

    print("\n[A] structure-only model (train on 2012-2018 graphs)")
    mA = train(new_model(bundle), bundle, growth_aware=False)
    rA = eval_spearman(mA, bundle)

    print("\n[B] growth-aware model (train on 2012-2018 graphs + align loss)")
    mB = train(new_model(bundle), bundle, growth_aware=True)
    rB = eval_spearman(mB, bundle)

    print("\n=== Spearman(-Φ, growth) per year  [* = TRAIN, # = HOLDOUT] ===")
    print(f"{'year':>6} {'A:structure':>14} {'B:growth-aware':>16}  split")
    for y in sorted(set(rA) | set(rB)):
        tag = "*train" if y in TRAIN_YEARS else ("#HOLD" if y in HOLDOUT_YEARS else "")
        a = rA.get(y, (float('nan'),))[0]; b = rB.get(y, (float('nan'),))[0]
        print(f"{y:>6} {a:>+14.3f} {b:>+16.3f}  {tag}")

    def avg(r, yrs):
        v = [r[y][0] for y in yrs if y in r]; return np.mean(v) if v else float('nan')
    print("\n--- summary (mean Spearman) ---")
    print(f"  TRAIN  : A={avg(rA,TRAIN_YEARS):+.3f}  B={avg(rB,TRAIN_YEARS):+.3f}")
    print(f"  HOLDOUT: A={avg(rA,HOLDOUT_YEARS):+.3f}  B={avg(rB,HOLDOUT_YEARS):+.3f}")
    print("  → B>>A on TRAIN but B≈A on HOLDOUT ⇒ alignment は記憶であり汎化せず（記述用途確定）")

    hy = HOLDOUT_YEARS[0]
    overlay_png(mA, bundle, hy, OUTDIR / f"overlay_A_structure_{hy}.png",
                f"[A structure-only] Φ valley (light) + CPC growth — holdout {hy}")
    overlay_png(mB, bundle, hy, OUTDIR / f"overlay_B_growth_aware_{hy}.png",
                f"[B growth-aware] Φ valley (light) + CPC growth — holdout {hy}")
    torch.save({"state_dict": mB.state_dict(), "year_min": Y0, "year_max": Y1,
                "hidden_dim": 128, "latent_dim": 2},
               str(OUTDIR / "map_construction_firm_growthaware.pt"))
    print("saved growth-aware checkpoint.")


if __name__ == "__main__":
    main()
