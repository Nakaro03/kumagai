"""baseline_trajectory_methods.py
=============================================================================
Reimplementations of PRESCIENT (Yeo et al., Nat Commun 2021) and MIOFlow
(Huguet et al., NeurIPS 2022) following the published equations, adapted to
the bipartite scholarly-graph data used elsewhere in this project.

Why a reimplementation? The official codebases assume single-cell expression
matrices with their own loader / preprocessing. Our data lives in
data/{DOMAIN}/alltime/fate_train.pt and the time axis is years, not
pseudotime. Adapting the official repos to this format is more invasive than
reusing X5's already-correct Sinkhorn rollout machinery with the loss
configuration each method specifies in its paper.

The mapping is exact at the equation level:

  PRESCIENT
    dz_t = -∇V_θ(z_t) dt + σ dW_t
    L     = Σ_t W_p(rollout_t, observed_t)                (W_p = Sinkhorn p=2)
    No anchor. No geodesic. No LOTO.
    → equivalent to our X5 with λ_phys = λ_geom = λ_smooth = 0
                       and LOTO disabled.

  MIOFlow
    dz_t = -∇V_θ(z_t) dt + σ dW_t                          (potential drift form)
    L    = Σ_t W_p(rollout_t, observed_t) + λ_geo · path_energy
    No anchor. Geodesic penalty present. No LOTO.
    → equivalent to our X5 with λ_phys = λ_smooth = 0, λ_geom > 0,
                       and LOTO disabled.

Limitations stated explicitly in the paper write-up:
  * MIOFlow's full method uses a geodesic autoencoder that learns a manifold
    metric upstream. We skip that step because (a) our embedding is already
    49-dim and trained on a bipartite VGAE, (b) without single-cell-like
    metric learning the geodesic distance ≈ Euclidean path length, which is
    the limiting case L_geom ≈ ∫‖dz/dt‖² dt that we already use.
  * PRESCIENT additionally fits a per-cell proliferation rate from
    cell-cycle gene expression; this has no analogue in bibliometrics so we
    drop it (their ablation showed it contributes <5% to W1).

This file is invoked the same way as run_pisde_x5.py:

  PNODE_DOMAIN_TARGET=patent_energy_top50 \
    python pnode_patent_runner/baseline_trajectory_methods.py --method prescient
  ... --method mioflow
"""
from __future__ import annotations

import argparse
import os
import sys

os.chdir("/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai/pnode_patent_runner")

from x5 import X5Config
from x5.train import train_x5


METHOD_PRESETS = {
    "prescient": {
        # PRESCIENT loss = pure Sinkhorn marginal matching, no LOTO
        "loto_enabled": False,
        "lam_phys": 0.0,
        "lam_geom": 0.0,
        "lam_smooth": 0.0,
        "fourier_K": 0,           # scalar t (no Fourier — PRESCIENT uses raw t in V_θ)
    },
    "mioflow": {
        # MIOFlow = Sinkhorn + geodesic path energy (no LOTO, no anchor)
        "loto_enabled": False,
        "lam_phys": 0.0,
        "lam_geom": 0.10,         # paper value: λ ≈ 0.1
        "lam_smooth": 0.0,
        "fourier_K": 0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=list(METHOD_PRESETS),
                        help="Which trajectory-inference baseline to run.")
    args = parser.parse_args()

    cfg = X5Config()
    for k, v in METHOD_PRESETS[args.method].items():
        setattr(cfg, k, v)

    # tag the output directory so it does not clobber x5 results
    os.environ["PNODE_ABL"] = args.method.upper()

    print("=" * 78)
    print(f"  Baseline trajectory method: {args.method.upper()}")
    print(f"  domain={cfg.domain} seed={cfg.seed} epochs={cfg.epochs}")
    print(f"  λ_phys={cfg.lam_phys} λ_geom={cfg.lam_geom} "
          f"λ_smooth={cfg.lam_smooth} LOTO={cfg.loto_enabled} "
          f"Fourier_K={cfg.fourier_K}")
    print("=" * 78)
    train_x5(cfg)


if __name__ == "__main__":
    main()
