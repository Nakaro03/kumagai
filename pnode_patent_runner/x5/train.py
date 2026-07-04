"""X5 training loop — LOTO + 4-term composite + held-out eval.

Outputs:
  RESULTS_X5/{DATA_NAME}/seed_{SEED}/
    config.json
    train.log
    train.best.pt          (lowest L_predict)
    train.last.pt
    evaluation.json        (per-test-t metrics, primary + secondary)
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import optim

sys.path.insert(0, "/tmp/PI-SDE")
from geomloss import SamplesLoss

from .config import X5Config, DOMAIN_TABLE
from .data import load_bipartite
from .model import X5SDE
from .loss import composite_loss
from .eval import evaluate_timepoint


def _p_samp(p: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.from_numpy(np.random.choice(p.shape[0], size=n,
                                            replace=p.shape[0] < n))
    return p[idx].clone()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def _output_dir(cfg: X5Config) -> str:
    data_name = DOMAIN_TABLE[cfg.domain][0]
    ablation = os.environ.get("PNODE_ABL", "")
    out = os.path.join("RESULTS_X5", data_name)
    if ablation:
        out = os.path.join(out, ablation)
    out = os.path.join(out, f"seed_{cfg.seed}")
    out = os.path.join(out, "loto" if cfg.loto_enabled else "alltime")
    os.makedirs(out, exist_ok=True)
    return out


def train_x5(cfg: X5Config) -> Dict:
    _set_seed(cfg.seed)
    data = load_bipartite(cfg)
    xp = data["xp"]
    y = data["y"]
    centroids = data["centroids"]
    growth = data["growth"]
    growth_norm = data["growth_norm"]

    _, train_t_full = DOMAIN_TABLE[cfg.domain]
    cfg.train_t = list(train_t_full)
    out_dir = _output_dir(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    print(f"  domain={cfg.domain}  x_dim={cfg.x_dim}  n_topics={cfg.n_topics}")
    print(f"  train_t={cfg.train_t}  LOTO={cfg.loto_enabled}")
    print(f"  λ_phys={cfg.lam_phys} λ_geom={cfg.lam_geom} λ_smooth={cfg.lam_smooth}")
    print(f"  out_dir={out_dir}")

    model = X5SDE(
        x_dim=cfg.x_dim,
        fourier_K=cfg.fourier_K,
        t_max=max(y) + 1.0,
        hidden=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        sigma_const=cfg.sigma_const,
        activation=cfg.activation,
    ).to(device)

    ot_loss = SamplesLoss("sinkhorn", p=2, blur=cfg.sinkhorn_blur,
                          scaling=cfg.sinkhorn_scaling, debias=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # save config snapshot
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.__dict__.items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, list, bool))},
                  f, indent=2)

    log_path = os.path.join(out_dir, "train.log")
    log = open(log_path, "w")

    best_predict = float("inf")
    n_batch = max(int(xp[0].shape[0] * cfg.batch_frac), 64)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        # ── LOTO scheduling
        alpha = min(1.0, (epoch + 1) / max(cfg.warmup_epochs, 1))
        mask_t = (random.choice(cfg.train_t)
                  if (cfg.loto_enabled and len(cfg.train_t) > 1) else None)

        # ── rollout from t=0
        x0 = _p_samp(xp[0], n_batch)
        r0 = torch.zeros(n_batch, 1)
        x_r_0 = torch.cat([x0, r0], dim=1).to(device)
        ts = [y[0]] + [y[t] for t in cfg.train_t]
        rollout = model(ts, x_r_0)  # list-like of (B, x_dim+1)

        # observed samples per timepoint
        observed = {j: _p_samp(xp[j], n_batch).to(device) for j in cfg.train_t}
        centroids_t = {j: centroids[j] for j in cfg.train_t}
        growth_norm_t = {j: growth_norm[j] for j in cfg.train_t}

        # smoothness grid: 200 random latent points × 50 dense timepoints
        z_grid = torch.cat([_p_samp(xp[t], 200 // max(len(cfg.train_t), 1) + 1)
                            for t in cfg.train_t], dim=0)[:200].to(device)
        t_dense = torch.linspace(y[0], max(y), 50, device=device)

        L, comp = composite_loss(
            model=model, ot_loss=ot_loss, rollout_z=list(rollout),
            observed=observed, centroids=centroids_t, growth_norm=growth_norm_t,
            train_t=cfg.train_t, mask_t=mask_t, alpha=alpha,
            lam_phys=cfg.lam_phys, lam_geom=cfg.lam_geom, lam_smooth=cfg.lam_smooth,
            z_sample=z_grid, t_dense=t_dense, device=device,
        )

        L.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        desc = (f"[ep {epoch+1:03d}] L={L.item():.4f} "
                f"pred={comp['predict']:.4f} phys={comp['phys']:.4f} "
                f"geom={comp['geom']:.4f} smooth={comp['smooth']:.4f} "
                f"α={comp['alpha']:.2f} mask={comp['mask_t']}")
        print(desc)
        log.write(desc + "\n"); log.flush()

        if comp["predict"] < best_predict:
            best_predict = comp["predict"]
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch + 1},
                       os.path.join(out_dir, "train.best.pt"))

    torch.save({"model_state_dict": model.state_dict(),
                "epoch": cfg.epochs},
               os.path.join(out_dir, "train.last.pt"))
    log.close()

    # ─── evaluate on every train_t held-out (synthetic LOTO sweep)
    metrics = evaluate_loto_sweep(model, data, cfg, device)
    eval_path = os.path.join(out_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved evaluation -> {eval_path}")
    return metrics


@torch.no_grad()
def _phi_at(model, centroids: torch.Tensor, t_val: float, device) -> np.ndarray:
    """Return Φ(c_k, t) as numpy array. Inference-time only."""
    c = centroids.to(device)
    t_col = torch.full((c.shape[0], 1), float(t_val), device=device)
    xt = torch.cat([c, t_col], dim=1)
    with torch.enable_grad():  # _pot calls requires_grad_, but no backward needed
        phi = model._func._pot(xt).squeeze(-1)
    return phi.detach().cpu().numpy()


def evaluate_loto_sweep(model, data, cfg: X5Config, device) -> Dict[str, Dict]:
    """Roll out from t=0, then at each test_t score the metrics in eval.py.

    For now we evaluate every j in train_t (descriptive comparison). The
    formal LOTO experiment will re-train with one t excluded per fold; that
    fold-driven script lives in run_pisde_x5.py.
    """
    model.eval()
    xp = data["xp"]
    y = data["y"]
    centroids = data["centroids"]
    growth = data["growth"]
    growth_norm = data["growth_norm"]

    n_eval = cfg.eval_n
    x0 = _p_samp(xp[0], n_eval)
    r0 = torch.zeros(n_eval, 1)
    x_r_0 = torch.cat([x0, r0], dim=1).to(device)
    ts = [y[0]] + [y[t] for t in cfg.train_t]

    # rollout requires grad-enabled context due to drift autograd
    with torch.enable_grad():
        rollout = model(ts, x_r_0)

    out: Dict[str, Dict] = {}
    for i, j in enumerate(cfg.train_t):
        roll = rollout[i + 1][:, :-1].detach()
        obs = xp[j].to(device)
        phi_arr = _phi_at(model, centroids[j], float(y[j]), device)
        m = evaluate_timepoint(
            rollout_samples=roll,
            observed_samples=obs,
            phi_at_centroids=phi_arr,
            growth_norm=np.asarray(growth_norm[j]),
            growth_raw=np.asarray(growth[j]),
            k=10,
            sinkhorn_blur=cfg.sinkhorn_blur,
        )
        out[f"t{j}"] = m
        print(f"  eval t={j}: W1={m['w1_marginal']:.3f} "
              f"Hits@10={m['hits_at_10']:.2f} NDCG@10={m['ndcg_at_10']:.3f} "
              f"MRR={m['mrr']:.3f} AP={m['ap']:.3f} ρ={m['spearman']:+.2f}")
    out["__mean__"] = {
        k: float(np.nanmean([out[f"t{j}"][k] for j in cfg.train_t]))
        for k in next(iter(out.values())).keys()
    }
    return out
