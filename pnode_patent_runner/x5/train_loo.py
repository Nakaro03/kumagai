"""x5/train_loo.py — TRUE leave-one-timepoint-out training for X5.

Difference from train.py (descriptive evaluation):
  * train_t is REDUCED to exclude the held-out timepoint(s)
  * SDE rollout, Sinkhorn marginal matching, AND Φ-anchor loss all exclude
    the held-out timepoint(s)
  * At eval time we roll out through ALL original timepoints (including
    held-out) and report metrics ONLY on held-out ones
  * No LOTO α-warmup masking (since we already truly exclude during training)

This is the protocol that mirrors X3-clean's prior "leave-one-out" experiment
where g_pred Spearman ρ = −0.35 was observed. If X5 also collapses to negative
correlation here, then X5 has no predictive ability over X3-clean. If X5
survives, the Φ-anchor + Fourier + smoothness combination does provide
extrapolation power.

Output: RESULTS_X5_LOO/{DATA_NAME}/holdout_t{T}/seed_{S}/
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import optim

sys.path.insert(0, "/tmp/PI-SDE")
from geomloss import SamplesLoss

from .config import X5Config, DOMAIN_TABLE
from .data import load_bipartite
from .model import X5SDE
from .loss import loss_phys, loss_geom, loss_smooth, sinkhorn_call
from .eval import evaluate_timepoint


def _p_samp(p: torch.Tensor, n: int) -> torch.Tensor:
    idx = torch.from_numpy(np.random.choice(p.shape[0], size=n,
                                            replace=p.shape[0] < n))
    return p[idx].clone()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def _output_dir(cfg: X5Config, holdout_ts: List[int]) -> str:
    data_name = DOMAIN_TABLE[cfg.domain][0]
    tag = "h" + "_".join(str(t) for t in holdout_ts)
    ablation = os.environ.get("PNODE_ABL", "full")
    out = os.path.join("RESULTS_X5_LOO", data_name, tag, ablation, f"seed_{cfg.seed}")
    os.makedirs(out, exist_ok=True)
    return out


def loss_predict_no_mask(*, ot_loss, rollout_z, observed, train_t):
    """L_predict but without LOTO masking — all included timepoints at weight 1."""
    total = rollout_z[0].new_zeros(())
    for i, j in enumerate(train_t):
        x = rollout_z[i + 1][:, :-1]
        y = observed[j]
        total = total + sinkhorn_call(ot_loss, x, y)
    return total / max(len(train_t), 1)


def train_x5_loo(cfg: X5Config, holdout_ts: List[int]) -> Dict:
    """Run X5 with t ∈ holdout_ts completely excluded from training.

    Evaluation is performed on holdout_ts using SDE rollout from t=0.
    """
    _set_seed(cfg.seed)
    data = load_bipartite(cfg)
    xp = data["xp"]; y = data["y"]
    centroids = data["centroids"]; growth = data["growth"]
    growth_norm = data["growth_norm"]

    _, train_t_full = DOMAIN_TABLE[cfg.domain]
    train_t_eff = [t for t in train_t_full if t not in holdout_ts]
    cfg.train_t = train_t_eff
    out_dir = _output_dir(cfg, holdout_ts)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device={device}")
    print(f"  domain={cfg.domain}  x_dim={cfg.x_dim}  n_topics={cfg.n_topics}")
    print(f"  train_t={train_t_eff}  holdout={holdout_ts}")
    print(f"  λ_phys={cfg.lam_phys} λ_geom={cfg.lam_geom} λ_smooth={cfg.lam_smooth}")
    print(f"  Fourier_K={cfg.fourier_K}")
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

    # save config
    cfg_dump = {k: v for k, v in cfg.__dict__.items()
                if not k.startswith("_") and isinstance(v, (int, float, str, list, bool))}
    cfg_dump["holdout_ts"] = list(holdout_ts)
    cfg_dump["train_t_effective"] = list(train_t_eff)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=2)

    log_path = os.path.join(out_dir, "train.log")
    log = open(log_path, "w")
    best_predict = float("inf")
    n_batch = max(int(xp[0].shape[0] * cfg.batch_frac), 64)

    for epoch in range(cfg.epochs):
        model.train(); optimizer.zero_grad()

        # rollout through train_t_eff (truly excludes holdout)
        x0 = _p_samp(xp[0], n_batch)
        r0 = torch.zeros(n_batch, 1)
        x_r_0 = torch.cat([x0, r0], dim=1).to(device)
        ts = [y[0]] + [y[t] for t in train_t_eff]
        rollout = model(ts, x_r_0)

        observed = {j: _p_samp(xp[j], n_batch).to(device) for j in train_t_eff}
        centroids_t = {j: centroids[j] for j in train_t_eff}
        growth_norm_t = {j: growth_norm[j] for j in train_t_eff}

        z_grid = torch.cat([_p_samp(xp[t], 200 // max(len(train_t_eff), 1) + 1)
                            for t in train_t_eff], dim=0)[:200].to(device)
        t_dense = torch.linspace(y[0], max(y), 50, device=device)

        L_pred = loss_predict_no_mask(ot_loss=ot_loss, rollout_z=list(rollout),
                                      observed=observed, train_t=train_t_eff)
        L_phys_ = loss_phys(model=model, centroids=centroids_t,
                            growth_norm=growth_norm_t, train_t=train_t_eff, device=device)
        L_geom_ = loss_geom(model=model, rollout_z=list(rollout),
                            train_t=train_t_eff, device=device)
        L_smooth_ = loss_smooth(model=model, z_sample=z_grid, t_dense=t_dense, device=device)
        L = L_pred + cfg.lam_phys * L_phys_ + cfg.lam_geom * L_geom_ + cfg.lam_smooth * L_smooth_

        L.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step(); scheduler.step()

        desc = (f"[ep {epoch+1:03d}] L={L.item():.4f} "
                f"pred={L_pred.item():.4f} phys={L_phys_.item():.4f} "
                f"geom={L_geom_.item():.4f} smooth={L_smooth_.item():.4f}")
        if (epoch + 1) % 20 == 0 or epoch < 5:
            print(desc)
        log.write(desc + "\n"); log.flush()

        if L_pred.item() < best_predict:
            best_predict = L_pred.item()
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch + 1},
                       os.path.join(out_dir, "train.best.pt"))

    torch.save({"model_state_dict": model.state_dict(), "epoch": cfg.epochs},
               os.path.join(out_dir, "train.last.pt"))
    log.close()

    # ── EVALUATE on holdout_ts (TRUE held-out)
    model.eval()
    n_eval = cfg.eval_n
    x0 = _p_samp(xp[0], n_eval); r0 = torch.zeros(n_eval, 1)
    x_r_0 = torch.cat([x0, r0], dim=1).to(device)
    # Roll through ALL original timepoints so we can extract held-out positions
    ts_full = [y[0]] + [y[t] for t in train_t_full]
    with torch.enable_grad():
        rollout_full = model(ts_full, x_r_0)

    metrics: Dict[str, Dict] = {}
    for j in holdout_ts:
        idx = train_t_full.index(j) + 1  # +1 because of initial t=0
        roll = rollout_full[idx][:, :-1].detach()
        obs = xp[j].to(device)
        # Φ at centroids
        c = centroids[j].to(device)
        t_col = torch.full((c.shape[0], 1), float(y[j]), device=device)
        xt = torch.cat([c, t_col], dim=1)
        with torch.enable_grad():
            phi_arr = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        m = evaluate_timepoint(
            rollout_samples=roll,
            observed_samples=obs,
            phi_at_centroids=phi_arr,
            growth_norm=np.asarray(growth_norm[j]),
            growth_raw=np.asarray(growth[j]),
            k=10,
            sinkhorn_blur=cfg.sinkhorn_blur,
        )
        metrics[f"t{j}"] = m
        print(f"  [HELD-OUT] t={j}: W1={m['w1_marginal']:.3f} "
              f"Hits@10={m['hits_at_10']:.2f} NDCG@10={m['ndcg_at_10']:.3f} "
              f"MRR={m['mrr']:.3f} AP={m['ap']:.3f} ρ={m['spearman']:+.2f}")

    metrics["__mean__"] = {
        k: float(np.nanmean([metrics[f"t{j}"][k] for j in holdout_ts]))
        for k in metrics[f"t{holdout_ts[0]}"].keys()
    }

    with open(os.path.join(out_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  saved -> {out_dir}/evaluation.json")
    return metrics
