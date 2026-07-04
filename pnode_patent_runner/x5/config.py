"""X5 hyperparameter container.

All defaults are chosen to match the X5_DESIGN.md spec; environment variables
can override individual values when running run_pisde_x5.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


DOMAIN_TABLE = {
    "paper":               ("PNode_Paper_X1",                 list(range(1, 4))),   # T=4 (2022-2025)
    "patent_energy_top50": ("PNode_Patent_Energy_X1_top50",   list(range(1, 12))),  # T=12
    "arxiv_construction":  ("PNode_ArXiv_Construction_X1_v2", list(range(1, 11))),  # T=11
    "jp_construction":     ("PNode_JP_Construction_X1",       list(range(1, 11))),  # T=11
}


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default


@dataclass
class X5Config:
    # ── data
    domain: str = field(default_factory=lambda: os.environ.get("PNODE_DOMAIN_TARGET", "patent_energy_top50"))
    data_root: str = "data"

    # ── model
    x_dim: int = 49                 # filled at load time
    fourier_K: int = field(default_factory=lambda: _env_int("PNODE_FOURIER_K", 8))
    hidden_dim: int = 400           # MLP width
    n_layers: int = 2               # depth
    activation: str = "softplus"
    sigma_const: float = 0.1

    # ── training
    seed: int = field(default_factory=lambda: _env_int("PNODE_SEED", 42))
    epochs: int = field(default_factory=lambda: _env_int("PNODE_EPOCHS", 200))
    lr: float = field(default_factory=lambda: _env_float("PNODE_LR", 5e-3))
    batch_frac: float = 0.1
    grad_clip: float = 0.1
    warmup_epochs: int = field(default_factory=lambda: _env_int("PNODE_WARMUP", 40))
    save_every: int = 500

    # ── LOTO (Leave-One-Timepoint-Out)
    loto_enabled: bool = field(default_factory=lambda: os.environ.get("PNODE_LOTO", "1") == "1")

    # ── loss weights
    lam_predict: float = 1.0
    lam_phys: float    = field(default_factory=lambda: _env_float("PNODE_LAM_PHYS", 0.5))
    lam_geom: float    = field(default_factory=lambda: _env_float("PNODE_LAM_GEOM", 0.01))
    lam_smooth: float  = field(default_factory=lambda: _env_float("PNODE_LAM_SMOOTH", 0.01))

    # ── Sinkhorn
    sinkhorn_blur: float = 0.05
    sinkhorn_scaling: float = 0.7

    # ── eval
    eval_n: int = 4000              # samples drawn for held-out W1/MMD

    # ── filled at load time
    n_topics: int = 0
    train_t: List[int] = field(default_factory=list)
    test_t: List[int] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    def resolve_paths(self) -> "X5Config":
        if self.domain not in DOMAIN_TABLE:
            raise ValueError(f"unknown domain {self.domain}; options={list(DOMAIN_TABLE)}")
        data_name, _ = DOMAIN_TABLE[self.domain]
        self.data_name = data_name
        self.data_dir = os.path.join(self.data_root, data_name)
        self.data_path = os.path.join(self.data_dir, "alltime", "fate_train.pt")
        return self
