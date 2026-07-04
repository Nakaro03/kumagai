"""run_pisde_x5_loo.py — True leave-one-timepoint-out evaluation of X5.

This is the protocol that mirrors X3-clean's prior leave-one-out experiment
(where Spearman ρ = −0.35 confirmed X3-clean has no predictive ability). If
X5 with full settings also collapses here, then X5 = X3-clean re-implemented
and the "predictive recovery" narrative is dead.

Environment variables:
  PNODE_DOMAIN_TARGET   patent_energy_top50 | arxiv_construction | jp_construction
  PNODE_SEED            random seed
  PNODE_HOLDOUT_T       comma-separated held-out timepoints (e.g. "5" or "5,9")
  PNODE_EPOCHS          training epochs
  PNODE_ABLATION        optional name to tag output: full|no_anchor|no_fourier|prescient

Usage:
  PNODE_HOLDOUT_T=5 PNODE_DOMAIN_TARGET=patent_energy_top50 \
    python pnode_patent_runner/run_pisde_x5_loo.py
"""
from __future__ import annotations

import os
import sys

os.chdir("/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai/pnode_patent_runner")

from x5 import X5Config
from x5.train_loo import train_x5_loo


ABLATION_PRESETS = {
    "full":        {},                                       # X5 full
    "no_anchor":   {"lam_phys": 0.0},                        # ≈ A2 with held-out
    "no_fourier":  {"fourier_K": 0},                         # ≈ A5 with held-out
    "prescient":   {"lam_phys": 0.0, "lam_geom": 0.0,
                    "lam_smooth": 0.0, "fourier_K": 0},      # PRESCIENT held-out
    "anchor_only": {"lam_geom": 0.0, "lam_smooth": 0.0,
                    "fourier_K": 0},                         # X5 minus geom/smooth/fourier
}


def main() -> None:
    cfg = X5Config()

    abl = os.environ.get("PNODE_ABLATION", "full")
    if abl not in ABLATION_PRESETS:
        raise SystemExit(f"unknown PNODE_ABLATION={abl}; options={list(ABLATION_PRESETS)}")
    for k, v in ABLATION_PRESETS[abl].items():
        setattr(cfg, k, v)
    os.environ["PNODE_ABL"] = abl

    holdout_str = os.environ.get("PNODE_HOLDOUT_T", "5")
    holdout_ts = [int(t) for t in holdout_str.split(",") if t.strip()]

    print("=" * 78)
    print("  X5 LOO — true held-out leave-one-timepoint-out")
    print(f"  domain={cfg.domain} seed={cfg.seed} holdout={holdout_ts} ablation={abl}")
    print(f"  λ_phys={cfg.lam_phys} λ_geom={cfg.lam_geom} "
          f"λ_smooth={cfg.lam_smooth} Fourier_K={cfg.fourier_K}")
    print("=" * 78)
    train_x5_loo(cfg, holdout_ts)


if __name__ == "__main__":
    main()
