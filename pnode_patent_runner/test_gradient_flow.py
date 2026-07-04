"""test_gradient_flow.py — does X3-clean's Phi add anything over growth-color?

A potential Phi differs from a mere growth-rate coloring ONLY if its gradient
encodes DYNAMICS: PI-SDE's drift is dz = -grad Phi dt + sigma dW, so topics
should DRIFT DOWNHILL in Phi. If the observed year-to-year motion of topic
centroids aligns with -grad Phi, then Phi captures "where topics are heading"
(something growth-color cannot show). If it does not, Phi is just a recoloring
of growth and the descriptive contribution collapses.

Test (in the FULL embedding space, NOT the 4%-variance PCA-2D):
  for each topic active at both t and t+1:
    observed displacement  d = centroid[t+1] - centroid[t]   (R^49)
    score / flow           f = -grad_c Phi(centroid[t], t)   (R^49)
    cosine(d, f)
  Aggregate mean cosine, fraction>0, vs a SHUFFLED null (topic-flow
  correspondence broken within each t -> expected cosine ~ 0).

  mean cosine >> 0 (clearly above shuffled null) => Phi encodes real dynamics
                                                    => beats growth-color.
  mean cosine ~ 0                                 => Phi is just recoloring
                                                    => descriptive claim is thin.

Run (main env; needs /tmp/PI-SDE on path):
  python pnode_patent_runner/test_gradient_flow.py --domain patent_energy_top50
  python pnode_patent_runner/test_gradient_flow.py --all
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

DOMAINS = {
    "paper":                ("PNode_Paper_X1", "data/PNode_Paper_X1"),
    "patent_energy_top50":  ("PNode_Patent_Energy_X1_top50", "data/PNode_Patent_Energy_X1_top50"),
    "arxiv_construction":   ("PNode_ArXiv_Construction_X1_v2", "data/PNode_ArXiv_Construction_X1_v2"),
    "jp_construction":      ("PNode_JP_Construction_X1", "data/PNode_JP_Construction_X1"),
}
CKPT_TMPL = ("RESULTS_X3_ABLATION/{name}/mask/x3abl_mask_g0.5/seed_{seed}/alltime/{f}")


def cos_rows(A, B):
    """row-wise cosine similarity between two [n,d] arrays."""
    na = np.linalg.norm(A, axis=1) + 1e-12
    nb = np.linalg.norm(B, axis=1) + 1e-12
    return (A * B).sum(1) / (na * nb)


def run_domain(domain, seed, device):
    from src.model import ForwardSDE

    name, ddir = DOMAINS[domain]
    cfg_path = CKPT_TMPL.format(name=name, seed=seed, f="config.pt")
    ckpt_path = CKPT_TMPL.format(name=name, seed=seed, f="train.best.pt")
    if not (os.path.exists(cfg_path) and os.path.exists(ckpt_path)):
        print(f"[{domain} seed{seed}] missing config/ckpt — skip"); return None

    cfg = Namespace(**torch.load(cfg_path, weights_only=False))
    data = torch.load(f"{ddir}/alltime/fate_train.pt", weights_only=False)
    centroids = data["centroids"]
    growth = data["growth"]
    y = data["y"]

    model = ForwardSDE(cfg).to(device)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_cos, all_cos_null, all_cos_growthnull = [], [], []
    T = len(centroids)
    for t in range(T - 1):
        c_t, c_n = centroids[t], centroids[t + 1]
        act = (c_t.abs().sum(-1) > 1e-6) & (c_n.abs().sum(-1) > 1e-6)
        if act.sum() < 3:
            continue
        idx = torch.where(act)[0]
        c = c_t[idx].to(device).clone().requires_grad_(True)
        tcol = torch.full((c.shape[0], 1), float(y[t]), device=device)
        phi = model._func._pot(torch.cat([c, tcol], 1)).squeeze(-1)
        grad = torch.autograd.grad(phi.sum(), c)[0]
        flow = (-grad).detach().cpu().numpy()          # -grad Phi  (drift direction)
        disp = (c_n[idx] - c_t[idx]).numpy()           # observed motion

        all_cos.append(cos_rows(disp, flow))
        # null: break topic<->flow correspondence (random permutation of flow rows)
        perm = np.random.permutation(len(idx))
        all_cos_null.append(cos_rows(disp, flow[perm]))

    if not all_cos:
        print(f"[{domain} seed{seed}] no usable consecutive pairs"); return None
    cos = np.concatenate(all_cos)
    null = np.concatenate(all_cos_null)
    res = dict(domain=domain, seed=seed, n=len(cos),
               mean_cos=float(cos.mean()), std_cos=float(cos.std()),
               frac_pos=float((cos > 0).mean()),
               null_mean=float(null.mean()), null_frac_pos=float((null > 0).mean()))
    print(f"[{domain:22s} seed{seed}] n={res['n']:4d}  "
          f"mean_cos={res['mean_cos']:+.3f} (frac>0 {res['frac_pos']:.2f})   "
          f"null={res['null_mean']:+.3f} (frac>0 {res['null_frac_pos']:.2f})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="patent_energy_top50")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all", action="store_true", help="run all 4 domains")
    args = ap.parse_args()
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doms = list(DOMAINS) if args.all else [args.domain]
    results = [r for d in doms if (r := run_domain(d, args.seed, device))]

    if results:
        m = np.mean([r["mean_cos"] for r in results])
        nm = np.mean([r["null_mean"] for r in results])
        print("\n" + "=" * 66)
        print(f"  mean alignment cosine across {len(results)} domain(s) = {m:+.3f}  "
              f"(shuffled null {nm:+.3f})")
        if m >= 0.20 and m - nm >= 0.15:
            print("  VERDICT: Phi encodes real DYNAMICS — topics drift downhill in Phi.")
            print("           This is something growth-color CANNOT show => Phi earns its keep.")
        elif m >= 0.10:
            print("  VERDICT: WEAK alignment. Some dynamical signal but modest; defend Phi")
            print("           cautiously, expect reviewers to push the growth-color baseline.")
        else:
            print("  VERDICT: NO alignment beyond null. Phi does NOT capture motion => it is")
            print("           effectively a recoloring of growth. Growth-color baseline wins.")
        print("=" * 66)


if __name__ == "__main__":
    main()
