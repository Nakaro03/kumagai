"""run_pisde_x5.py — main entry for X5 (Φ-as-Driver Predictive PI-SDE).

Environment variables (with sensible defaults defined in x5/config.py):
  PNODE_DOMAIN_TARGET   patent_energy_top50 | arxiv_construction | jp_construction | paper
  PNODE_SEED            random seed                                 (default 42)
  PNODE_EPOCHS          training epochs                              (default 200)
  PNODE_LR              learning rate                                (default 5e-3)
  PNODE_WARMUP          α (mask weight) warmup epochs               (default 40)
  PNODE_LOTO            "1" to enable LOTO (default), "0" to disable
  PNODE_LAM_PHYS        anchor weight                                (default 0.5)
  PNODE_LAM_GEOM        path energy weight                           (default 0.01)
  PNODE_LAM_SMOOTH      time-smoothness weight                       (default 0.01)

Usage:
  cd /home/nakamuraroi/kumagai
  PNODE_DOMAIN_TARGET=patent_energy_top50 PNODE_EPOCHS=30 \
      python pnode_patent_runner/run_pisde_x5.py
"""
from __future__ import annotations

import os
import sys

os.chdir("/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai")
sys.path.insert(0, "/home/nakamuraroi/kumagai/pnode_patent_runner")

from x5 import X5Config
from x5.train import train_x5


def main() -> None:
    cfg = X5Config()
    print("=" * 78)
    print("  X5 — Φ-as-Driver Predictive PI-SDE")
    print(f"  domain={cfg.domain}  seed={cfg.seed}  epochs={cfg.epochs}")
    print("=" * 78)
    train_x5(cfg)


if __name__ == "__main__":
    main()
