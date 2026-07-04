"""test_flow_vs_pointcloud.py — close the "data advantage" objection.

test_flow_vs_growthgrad.py showed -grad Phi aligns with topic motion far better
than the gradient of a growth field built from the 50 CENTROIDS. A reviewer can
object: "Phi wins only because it uses the full ~4000-pt/year member clouds (via
the OT/Sinkhorn term), while the growth baseline saw just 50 centroids — a DATA
advantage, not a modeling advantage."

This script gives the non-parametric baseline the SAME data Phi uses. A region
'grows' iff its point density rises from t to t+1, so the growth direction is the
gradient of the density-change field, estimated from the FULL point clouds:

  growth flow  f^pc_i = meanshift(z_i; cloud_{t+1}, h) - meanshift(z_i; cloud_t, h)
  meanshift(z; X, h) = (sum_k w_k x_k / sum_k w_k) - z,  w_k = exp(-||z-x_k||^2/2h^2)

(meanshift = grad log density up to 1/h^2; the difference = grad of log p_{t+1}/p_t,
i.e. points toward where density increased = where activity grew.) Bandwidth swept.

Compare cos(motion, -grad Phi) vs cos(motion, f^pc). If Phi still wins, the
advantage is MODELING (Phi compresses the trajectory into a coherent potential
flow), not mere data access => objection closed.

Run:  python pnode_patent_runner/test_flow_vs_pointcloud.py --all --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace

import numpy as np
import torch

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from test_gradient_flow import DOMAINS, CKPT_TMPL, cos_rows


def meanshift(z, X, h):
    """meanshift vector at each z_i toward the density of cloud X (bandwidth h)."""
    d2 = (z ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * z @ X.T
    np.maximum(d2, 0, out=d2)
    w = np.exp(-d2 / (2 * h * h))                       # [n,N]
    denom = w.sum(1, keepdims=True) + 1e-12
    return (w @ X) / denom - z                          # [n,d]


def run_domain(domain, seed, device, bw_mults, rng, max_pts=3000):
    from src.model import ForwardSDE
    name, ddir = DOMAINS[domain]
    cfg_path = CKPT_TMPL.format(name=name, seed=seed, f="config.pt")
    ckpt_path = CKPT_TMPL.format(name=name, seed=seed, f="train.best.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
        print(f"[{domain} seed{seed}] missing — skip"); return None

    cfg = Namespace(**torch.load(cfg_path, weights_only=False))
    data = torch.load(f"{ddir}/alltime/fate_train.pt", weights_only=False)
    centroids, xp, y = data["centroids"], data["xp"], data["y"]

    model = ForwardSDE(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False,
                                     map_location=device)["model_state_dict"])
    model.eval()

    cos_phi, cos_pc = [], {m: [] for m in bw_mults}
    for t in range(len(centroids) - 1):
        c_t, c_n = centroids[t], centroids[t + 1]
        act = (c_t.abs().sum(-1) > 1e-6) & (c_n.abs().sum(-1) > 1e-6)
        if act.sum() < 3:
            continue
        idx = torch.where(act)[0]
        z = c_t[idx].numpy()
        disp = (c_n[idx] - c_t[idx]).numpy()

        c = c_t[idx].to(device).clone().requires_grad_(True)
        tcol = torch.full((c.shape[0], 1), float(y[t]), device=device)
        phi = model._func._pot(torch.cat([c, tcol], 1)).squeeze(-1)
        phi_flow = (-torch.autograd.grad(phi.sum(), c)[0]).detach().cpu().numpy()
        cos_phi.append(cos_rows(disp, phi_flow))

        Xt, Xn = xp[t].numpy(), xp[t + 1].numpy()
        if Xt.shape[0] > max_pts:
            Xt = Xt[rng.choice(Xt.shape[0], max_pts, replace=False)]
        if Xn.shape[0] > max_pts:
            Xn = Xn[rng.choice(Xn.shape[0], max_pts, replace=False)]
        d2 = (z[:, None, :] - z[None, :, :]) ** 2
        d2 = d2.sum(-1)
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        for m in bw_mults:
            flow = meanshift(z, Xn, med * m) - meanshift(z, Xt, med * m)
            cos_pc[m].append(cos_rows(disp, flow))

    if not cos_phi:
        return None
    res = {"domain": domain, "phi": float(np.concatenate(cos_phi).mean())}
    for m in bw_mults:
        res[f"pc{m}"] = float(np.concatenate(cos_pc[m]).mean())
    res["pc_best"] = max(res[f"pc{m}"] for m in bw_mults)
    bwstr = "  ".join(f"pc(h={m})={res[f'pc{m}']:+.3f}" for m in bw_mults)
    print(f"[{domain:22s}] Phi={res['phi']:+.3f}   {bwstr}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="patent_energy_top50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bw_mults = [0.5, 1.0, 2.0]

    doms = list(DOMAINS) if args.all else [args.domain]
    print("cos(motion, -grad Phi) vs cos(motion, full-point-cloud density-change flow):\n")
    results = [r for d in doms if (r := run_domain(d, args.seed, device, bw_mults, rng))]

    if results:
        mp = np.mean([r["phi"] for r in results])
        mpc = np.mean([r["pc_best"] for r in results])
        print("\n" + "=" * 70)
        print(f"  mean cos(motion, -grad Phi)                       = {mp:+.3f}")
        print(f"  mean cos(motion, point-cloud growth flow) [best h] = {mpc:+.3f}")
        print(f"  Phi advantage over FULL-DATA growth baseline      = {mp - mpc:+.3f}")
        if mp - mpc >= 0.10:
            print("  VERDICT: Phi beats the growth flow even when it uses the SAME 4000-pt")
            print("           clouds => advantage is MODELING, not data. Objection closed.")
        elif mp - mpc >= 0.03:
            print("  VERDICT: Phi modestly ahead with equal data. Defensible; report honestly.")
        else:
            print("  VERDICT: Phi ~ full-data growth flow. The earlier win WAS a data")
            print("           advantage. Phi does not beat a fair density-change baseline.")
        print("=" * 70)


if __name__ == "__main__":
    main()
