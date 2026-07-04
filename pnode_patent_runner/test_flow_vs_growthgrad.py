"""test_flow_vs_growthgrad.py — the DECISIVE test for X3-clean's Phi.

test_gradient_flow.py showed topics drift downhill in Phi (cos ~+0.28 vs null
~0.04), so Phi carries flow info a flat growth-coloring lacks. BUT Phi is trained
to Phi ~ -growth, so -grad Phi may simply ~ +grad(growth): "topics move toward
higher growth." If so, the gradient of a plain interpolated GROWTH field aligns
with motion just as well, and Phi adds nothing over growth-color (+interpolation)
— the user's point.

This script puts both flows on the SAME footing:
  observed displacement   d_i = centroid[t+1]_i - centroid[t]_i
  Phi flow                f^Phi_i    = -grad_c Phi(z_i, t)
  growth-gradient flow    f^g_i      = grad of Nadaraya-Watson growth field
                          f^g_i propto sum_j w_ij (g_j - G_i)(z_j - z_i),
                          w_ij = exp(-||z_i-z_j||^2 / 2h^2), bandwidth h swept.
Compare mean cosine(d, f^Phi) vs mean cosine(d, f^g).

  cos(Phi) >> cos(growth-grad)  => Phi learned dynamics beyond the growth surface
                                   => Phi earns its keep over growth-color.
  cos(Phi) ~ cos(growth-grad)   => Phi ~ smoothed growth; growth-color suffices.

Run:  python pnode_patent_runner/test_flow_vs_growthgrad.py --all
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

from test_gradient_flow import DOMAINS, CKPT_TMPL, cos_rows  # reuse


def growth_grad_flow(z, g, h):
    """Nadaraya-Watson growth-field gradient direction at each point z_i.
    z:[n,d] positions, g:[n] growth, h:bandwidth. Returns [n,d] flow (toward higher g)."""
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)        # [n,n]
    W = np.exp(-d2 / (2 * h * h))                              # [n,n]
    G = (W @ g) / (W.sum(1) + 1e-12)                           # [n] smoothed growth
    coef = W * (g[None, :] - G[:, None])                       # [n,n]
    diff = z[None, :, :] - z[:, None, :]                       # [n,n,d] (z_j - z_i)
    return np.einsum("ij,ijd->id", coef, diff)                 # [n,d]


def run_domain(domain, seed, device, bw_mults):
    from src.model import ForwardSDE
    name, ddir = DOMAINS[domain]
    cfg_path = CKPT_TMPL.format(name=name, seed=seed, f="config.pt")
    ckpt_path = CKPT_TMPL.format(name=name, seed=seed, f="train.best.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
        print(f"[{domain} seed{seed}] missing — skip"); return None

    cfg = Namespace(**torch.load(cfg_path, weights_only=False))
    data = torch.load(f"{ddir}/alltime/fate_train.pt", weights_only=False)
    centroids, growth_norm, y = data["centroids"], data["growth_norm"], data["y"]

    model = ForwardSDE(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False,
                                     map_location=device)["model_state_dict"])
    model.eval()

    cos_phi, cos_g = [], {m: [] for m in bw_mults}
    for t in range(len(centroids) - 1):
        c_t, c_n = centroids[t], centroids[t + 1]
        act = (c_t.abs().sum(-1) > 1e-6) & (c_n.abs().sum(-1) > 1e-6)
        if act.sum() < 3:
            continue
        idx = torch.where(act)[0]
        z = c_t[idx].numpy()
        g = growth_norm[t][idx].numpy()
        disp = (c_n[idx] - c_t[idx]).numpy()

        c = c_t[idx].to(device).clone().requires_grad_(True)
        tcol = torch.full((c.shape[0], 1), float(y[t]), device=device)
        phi = model._func._pot(torch.cat([c, tcol], 1)).squeeze(-1)
        phi_flow = (-torch.autograd.grad(phi.sum(), c)[0]).detach().cpu().numpy()
        cos_phi.append(cos_rows(disp, phi_flow))

        # bandwidth as multiple of median pairwise distance
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
        med = np.sqrt(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        for m in bw_mults:
            cos_g[m].append(cos_rows(disp, growth_grad_flow(z, g, med * m)))

    if not cos_phi:
        return None
    cp = np.concatenate(cos_phi)
    res = {"domain": domain, "n": len(cp), "phi": float(cp.mean())}
    for m in bw_mults:
        res[f"g{m}"] = float(np.concatenate(cos_g[m]).mean())
    best_g = max(res[f"g{m}"] for m in bw_mults)
    res["g_best"] = best_g
    bwstr = "  ".join(f"g(h={m})={res[f'g{m}']:+.3f}" for m in bw_mults)
    print(f"[{domain:22s}] n={res['n']:4d}  Phi={res['phi']:+.3f}   {bwstr}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="patent_energy_top50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bw_mults = [0.5, 1.0, 2.0]

    doms = list(DOMAINS) if args.all else [args.domain]
    print("Alignment cosine of observed motion with each flow "
          "(Phi drift vs growth-field gradient, bandwidth h x median dist):\n")
    results = [r for d in doms if (r := run_domain(d, args.seed, device, bw_mults))]

    if results:
        mp = np.mean([r["phi"] for r in results])
        mg = np.mean([r["g_best"] for r in results])
        print("\n" + "=" * 70)
        print(f"  mean cos(motion, -grad Phi)        = {mp:+.3f}")
        print(f"  mean cos(motion, grad growth) [best h per domain] = {mg:+.3f}")
        print(f"  Phi advantage over growth-gradient = {mp - mg:+.3f}")
        if mp - mg >= 0.10:
            print("  VERDICT: Phi BEATS the growth-gradient baseline => it learned dynamics")
            print("           beyond the growth surface. Phi earns its keep over growth-color.")
        elif mp - mg >= 0.03:
            print("  VERDICT: Phi modestly ahead. Defensible but reviewers will push hard;")
            print("           lead with interpretability, report this honestly.")
        else:
            print("  VERDICT: Phi ~ growth-gradient. Phi is effectively a smoothed growth")
            print("           field => growth-color (+interpolation) suffices. User is right.")
        print("=" * 70)


if __name__ == "__main__":
    main()
