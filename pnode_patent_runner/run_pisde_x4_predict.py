"""
X4-predict: PI-SDE with **predictive** capability.

Fixes the 4 prediction bugs of X3-clean diagnosed in X3_DESCRIPTIVE_FRAMING.md:

  B1 (time encoding):  Φ(x, t) uses raw t → Fourier(t) embedding (smooth extrapolation)
  B2 (causal training): SDE rollout restricted to train_t (no in-sample leak)
  B3 (learning target): self-anchor Φ(c,t)≈-g̃(t) → future-anchor Φ(c(t), t+K) ≈ -g̃(t+K)
  B4 (evaluation):     in-sample → temporal hold-out (last 30% as test)

Architecture differences from X3-clean:
  - FourierAutoGenerator: replaces raw t with [sin, cos] expansion (K Fourier components)
  - PredictivePredictor: NO g̃ input (avoid leak), takes (centroid, t_target)
  - Loss = L_Sinkhorn + L_val_future_K + L_val_retro + λ_g · L_growth_future_K
  - Temporal split: train_t = first floor(T_TRAIN_FRAC * T), test_t = remainder

Multi-horizon: train and evaluate K ∈ {1, 2, 3} simultaneously.

Output: RESULTS_X4_PREDICT/{DATA_NAME}/x4_K{K_FOURIER}_p{K_PREDICT_MAX}_g{LAM_G}/seed_{SEED}/...

Usage:
  PNODE_DOMAIN_TARGET=patent_energy_top50 PNODE_SEED=42 \\
    PNODE_K_PREDICT_MAX=3 PNODE_T_TRAIN_FRAC=0.7 PNODE_EPOCHS=200 \\
    python pnode_patent_runner/run_pisde_x4_predict.py
"""
from __future__ import annotations

import math
import os
import sys
import json
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from scipy import stats

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE, AutoGenerator

# ── Hyperparameters ─────────────────────────────────────────────────
SEED          = int(os.environ.get("PNODE_SEED", 42))
EPOCHS        = int(os.environ.get("PNODE_EPOCHS", 200))
LAM_G         = float(os.environ.get("PNODE_LAM_G", 0.5))
LAM_RETRO     = float(os.environ.get("PNODE_LAM_RETRO", 0.1))     # retrospective anchor (small)
D_CTX         = int(os.environ.get("PNODE_D_CTX", 32))
K_FOURIER     = int(os.environ.get("PNODE_K_FOURIER", 8))
T_MAX_FOURIER = float(os.environ.get("PNODE_T_MAX_FOURIER", 20.0))
K_PREDICT_MAX = int(os.environ.get("PNODE_K_PREDICT_MAX", 3))     # multi-horizon: 1..K_max
T_TRAIN_FRAC  = float(os.environ.get("PNODE_T_TRAIN_FRAC", 0.7))

DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "patent_energy_top50")
DOMAIN_MAP = {
    "paper":              ("PNode_Paper_X1",                  "data/PNode_Paper_X1",                  3),
    "patent_energy_top50":("PNode_Patent_Energy_X1_top50",    "data/PNode_Patent_Energy_X1_top50",    11),
    "arxiv_construction": ("PNode_ArXiv_Construction_X1_v2",  "data/PNode_ArXiv_Construction_X1_v2",  10),
    "jp_construction":    ("PNode_JP_Construction_X1",        "data/PNode_JP_Construction_X1",        10),
}
if DOMAIN not in DOMAIN_MAP:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")
DATA_NAME, DATA_DIR, T_MAX_DATA = DOMAIN_MAP[DOMAIN]


# ── Fourier time embedding (B1 fix) ────────────────────────────────
class FourierAutoGenerator(AutoGenerator):
    """AutoGenerator with Fourier time embedding instead of raw t scalar.

    Input to MLP changes from [x ∈ R^D, t ∈ R] (D+1 dims) to
    [x ∈ R^D, sin(2π·1·t/T_max), …, sin(2π·K·t/T_max), cos(…), …] (D + 2K dims).
    """

    def __init__(self, config, K_fourier=8, T_max=20.0):
        super().__init__(config)
        self.K_fourier = K_fourier
        self.T_max = T_max
        x_dim = config.x_dim
        new_input = x_dim + 2 * K_fourier
        # Rebuild self.net for the wider input
        layers = []
        in_d = new_input
        for k_h in config.k_dims:
            layers.append(nn.Linear(in_d, k_h))
            layers.append(self.act())
            in_d = k_h
        layers.append(nn.Linear(config.k_dims[-1], 1, bias=False))
        layers[-1].weight.data.zero_()      # zero-init output (X3 trick)
        self.net = nn.Sequential(*layers)
        # Frequencies (fixed)
        freqs = 2 * math.pi * torch.arange(1, K_fourier + 1).float() / T_max
        self.register_buffer("freqs", freqs)

    def _pot(self, xt):
        xt = xt.requires_grad_()
        x = xt[:, :-1]
        t = xt[:, -1:]                                              # (batch, 1)
        embed = torch.cat([torch.sin(self.freqs * t), torch.cos(self.freqs * t)], dim=-1)
        h = torch.cat([x, embed], dim=-1)
        return self.net(h)


# ── Predictor (no g̃ input → no leak) ─────────────────────────────
class PredictivePredictor(nn.Module):
    """Predict g̃ from (centroid, target_time) only.  NO g̃ input."""

    def __init__(self, x_dim, d_ctx=32):
        super().__init__()
        # Input: [centroid (x_dim) + t_target (1)] = x_dim + 1
        self.embed = nn.Linear(x_dim + 1, d_ctx)
        self.attn = nn.MultiheadAttention(d_ctx, num_heads=2, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_ctx)
        self.out = nn.Linear(d_ctx + x_dim + 1, 1)

    def forward(self, c, t_target):
        """c: (K, x_dim)  centroids at current time.  t_target: scalar (future)."""
        K = c.shape[0]
        t_col = torch.full((K, 1), float(t_target), device=c.device)
        feat = torch.cat([c, t_col], dim=-1)                        # (K, x+1)
        emb = self.embed(feat).unsqueeze(0)                         # (1, K, d_ctx)
        att, _ = self.attn(emb, emb, emb)
        ctx = self.norm(emb + att).squeeze(0)                       # (K, d_ctx)
        inp = torch.cat([ctx, c, t_col], dim=-1)
        return self.out(inp).squeeze(-1)


# ── Temporal split ─────────────────────────────────────────────────
def make_temporal_split(T_data: int, frac: float):
    """Split timepoints [1..T_data] into train (early frac) and test (late)."""
    T_train = max(1, int(round(frac * T_data)))
    T_train = min(T_train, T_data - 1)                              # leave at least 1 test point
    train_t = list(range(1, T_train + 1))
    test_t = list(range(T_train + 1, T_data + 1))
    return train_t, test_t


# ── make_args / init_config ────────────────────────────────────────
def make_args(train_t, test_t):
    ns = argparse.Namespace(
        seed=SEED, use_cuda=True, device=0,
        out_dir=f"RESULTS_X4_PREDICT/{DATA_NAME}",
        data=DATA_NAME, data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        k_dims=[400, 400], activation="softplus",
        sigma_type="const", sigma_const=0.1,
        train_epochs=EPOCHS, train_lr=0.005,
        train_lambda=0.0,
        train_batch=0.1, train_clip=0.1, save=500,
        evaluate_n=10000, evaluate_data=None, evaluate_baseline=False,
        task="x4_predict", train=False, evaluate=None, config=None,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, ns=2000,
        start_t=0, train_t=train_t, leaveout_t="", test_t=test_t,
    )
    ns.layers = len(ns.k_dims)
    return ns


def init_config(args):
    args.kDims = '_'.join(map(str, args.k_dims))
    name = f"x4_K{K_FOURIER}_p{K_PREDICT_MAX}_g{LAM_G}_retro{LAM_RETRO}"
    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}",
                                f"split{int(T_TRAIN_FRAC*100)}")
    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt = os.path.join(args.out_dir, "train.{}.pt")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


# ── X4 loss ────────────────────────────────────────────────────────
def compute_x4_loss(model, predictor, centroids_t, growth_norm_dict,
                    t_current, K_predict_list, max_train_t, device, lam_growth, lam_retro):
    """
    Args:
      centroids_t: centroids[t_current] — current centroid configuration
      growth_norm_dict: {t: growth_norm[t]} for all needed timepoints
      t_current: current time index
      K_predict_list: list of K values, e.g., [1, 2, 3]
      max_train_t: maximum allowed t_future (STRICT no-leak: t_future ≤ max_train_t)
      lam_growth: predictor loss weight
      lam_retro: retrospective anchor weight (small)

    Returns:
      L_total, comp_dict
    """
    active = centroids_t.abs().sum(dim=-1) > 1e-6
    if active.sum() < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {"val_fut": 0, "val_retro": 0, "growth_fut": 0, "n": 0}

    c = centroids_t[active].to(device)
    K_act = c.shape[0]

    comp = {"val_fut": 0.0, "val_retro": 0.0, "growth_fut": 0.0, "n": 0}
    L_total = torch.tensor(0.0, device=device)

    # (1) Retrospective anchor: Φ(c_t, t) ≈ -g̃(t)  [keeps landscape interpretable]
    if t_current in growth_norm_dict and lam_retro > 0:
        g_now = growth_norm_dict[t_current][active].to(device)
        t_col = torch.full((K_act, 1), float(t_current), device=device)
        xt_now = torch.cat([c, t_col], dim=1).requires_grad_()
        phi_now = model._func._pot(xt_now).squeeze(-1)
        L_retro = ((phi_now + g_now) ** 2).mean()
        L_total = L_total + lam_retro * L_retro
        comp["val_retro"] = L_retro.item()

    # (2) Future-time anchor + predictor for each K
    # STRICT no-leak: t_future must be ≤ max_train_t (test timepoints forbidden in loss)
    n_K = 0
    for K_pred in K_predict_list:
        t_future = t_current + K_pred
        if t_future > max_train_t:        # ★ leak prevention
            continue
        if t_future not in growth_norm_dict:
            continue
        g_fut = growth_norm_dict[t_future][active].to(device)
        if g_fut.shape[0] != K_act:
            continue
        # Φ(c_t, t_future) ≈ -g̃(t_future)
        t_col = torch.full((K_act, 1), float(t_future), device=device)
        xt_fut = torch.cat([c, t_col], dim=1).requires_grad_()
        phi_fut = model._func._pot(xt_fut).squeeze(-1)
        L_val_fut = ((phi_fut + g_fut) ** 2).mean()
        # predictor(c_t, t_future) ≈ g̃(t_future)
        g_pred = predictor(c, t_future)
        L_growth_fut = ((g_pred - g_fut) ** 2).mean()

        L_total = L_total + L_val_fut + lam_growth * L_growth_fut
        comp["val_fut"] += L_val_fut.item()
        comp["growth_fut"] += L_growth_fut.item()
        n_K += 1
    if n_K > 0:
        comp["val_fut"] /= n_K
        comp["growth_fut"] /= n_K
    comp["n"] = n_K
    return L_total, comp


# ── Training ────────────────────────────────────────────────────────
def run_x4(args, config, x, y, centroids, growth_norm):
    import src.train as pisde_train
    from src.train import OTLoss

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_t = config.train_t
    test_t = config.test_t
    print(f"  train_t = {train_t}")
    print(f"  test_t  = {test_t}")
    print(f"  K_predict ∈ 1..{K_PREDICT_MAX}, K_fourier = {K_FOURIER}, T_max_fourier = {T_MAX_FOURIER}")
    print(f"  λ_growth = {LAM_G}, λ_retro = {LAM_RETRO}")

    # Build model: ForwardSDE wraps FourierAutoGenerator
    model = ForwardSDE(config).to(device)
    model._func = FourierAutoGenerator(config, K_fourier=K_FOURIER, T_max=T_MAX_FOURIER).to(device)
    predictor = PredictivePredictor(x_dim=config.x_dim, d_ctx=D_CTX).to(device)

    loss_fn = OTLoss(config, device)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "weight_decay": 0.0},
        {"params": predictor.parameters(), "weight_decay": 1e-4},
    ], lr=config.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    torch.save(config.__dict__, config.config_pt)
    K_predict_list = list(range(1, K_PREDICT_MAX + 1))

    # Pre-make growth_norm dict for quick lookup
    growth_norm_dict = {t: growth_norm[t] for t in range(len(growth_norm))}

    import tqdm
    pbar = tqdm.tqdm(range(config.train_epochs))
    best_loss = np.inf
    log = open(config.train_log, "w")

    for epoch in pbar:
        model.zero_grad(); predictor.zero_grad()
        losses_xy, losses_x4 = [], []
        comp_sum = {"val_fut": 0, "val_retro": 0, "growth_fut": 0, "n": 0}

        # (1) Causal SDE rollout from t=0 through train_t (B2 fix)
        dat_prev = x[config.start_t]
        x_i, _ = pisde_train.p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
        r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
        x_r_i = torch.cat([x_i, r_i], dim=1).to(device)
        ts = [0] + train_t
        y_ts = [np.float64(y[ts_i]) for ts_i in ts]
        x_r_s = model(y_ts, x_r_i)                                   # rollout

        # (2) L_Sinkhorn at each train_t
        for j in train_t:
            dat_cur = x[j]
            y_j, _ = pisde_train.p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
            position = train_t.index(j)
            a_i = torch.ones(x_i.shape[0]); a_i = a_i / a_i.sum()
            b_j = torch.ones(y_j.shape[0]); b_j = b_j / b_j.sum()
            loss_xy = loss_fn(a_i, x_r_s[position + 1][:, 0:-1], b_j, y_j)
            losses_xy.append(loss_xy.item())

            # (3) X4 loss: future anchor + predictor (B1, B3 fix)
            # max_train_t prevents leak: future targets must stay within training range
            L_x4, c4 = compute_x4_loss(
                model, predictor, centroids[j], growth_norm_dict,
                t_current=j, K_predict_list=K_predict_list,
                max_train_t=max(train_t),    # ★ no-leak constraint
                device=device, lam_growth=LAM_G, lam_retro=LAM_RETRO,
            )
            losses_x4.append(L_x4.item())
            for k in ("val_fut", "val_retro", "growth_fut", "n"):
                comp_sum[k] += c4[k]
            (loss_xy + L_x4).backward(retain_graph=True)

        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters()),
                config.train_clip,
            )
        optimizer.step()
        scheduler.step()
        model.zero_grad(); predictor.zero_grad()

        l_xy = float(np.mean(losses_xy)) if losses_xy else 0.0
        l_x4 = float(np.mean(losses_x4)) if losses_x4 else 0.0
        n_tot = max(comp_sum["n"], 1)
        desc = (f"[ep {epoch+1}] Sink={l_xy:.3f} val_fut={comp_sum['val_fut']/n_tot:.3f} "
                f"val_retro={comp_sum['val_retro']/len(train_t):.3f} "
                f"growth_fut={comp_sum['growth_fut']/n_tot:.3f}")
        pbar.set_description(desc)
        log.write(desc + "\n"); log.flush()

        if l_xy < best_loss:
            best_loss = l_xy
            torch.save({
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "epoch": epoch + 1,
            }, config.train_pt.format("best"))

        if (epoch + 1) % config.save == 0:
            ep_str = str(epoch + 1).rjust(6, "0")
            torch.save({
                "model_state_dict": model.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "epoch": epoch + 1,
            }, config.train_pt.format(f"epoch_{ep_str}"))

    log.close()
    return config


# ── Evaluation: predict on held-out test_t from last training centroid ──
def evaluate_x4(args, config, x, y, centroids, growth_raw, growth_norm,
                model, predictor, device):
    """Evaluate predictions on held-out test_t (B4 fix).

    At test, use the LAST training centroid c_j(T_train) as input (realistic setting:
    we know the topic structure as of training cut-off but want to predict future growth).

    Reports:
      - Per-test-t spearman, NDCG, MSE
      - Per-K-step ahead metrics (where K = t_test - t_train)
    """
    train_t = config.train_t
    test_t = config.test_t
    t_last_train = max(train_t)
    c_last = centroids[t_last_train]
    active = c_last.abs().sum(dim=-1) > 1e-6
    c_a = c_last[active].to(device)

    results = []
    print(f"\n  {'t':<4} {'K_ahead':<8} {'g-Sp':<14} {'g-NDCG':<10} {'g-MSE':<10} "
          f"{'phi-Sp':<14} {'phi-NDCG':<10}")
    print("  " + "-" * 80)

    for t_eval in test_t:
        K_ahead = t_eval - t_last_train
        g_a = growth_raw[t_eval][active].numpy()
        g_n = growth_norm[t_eval][active].to(device).cpu().numpy()

        # Predict via predictor head
        g_pred = predictor(c_a, t_eval).detach().cpu().numpy()
        # Predict via Φ (anchor): Φ(c_last, t_eval) ≈ -g̃(t_eval) → -Φ ≈ g̃
        t_col = torch.full((c_a.shape[0], 1), float(y[t_eval]), device=device)
        xt = torch.cat([c_a, t_col], dim=1).requires_grad_()
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()

        # Ranking metrics (high g_pred / low phi → high growth)
        r_g, p_g = stats.spearmanr(g_pred, g_a)
        r_phi, p_phi = stats.spearmanr(phi, g_a)
        K = min(10, len(g_a))
        rel = np.maximum(g_a, 0.0)
        ideal = np.argsort(-rel)
        idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
        order_g = np.argsort(-g_pred)
        order_phi = np.argsort(phi)
        ndcg_g = sum(rel[order_g[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
        ndcg_phi = sum(rel[order_phi[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
        prec_g = (g_a[order_g[:K]] > 0).mean()
        prec_phi = (g_a[order_phi[:K]] > 0).mean()

        sig_g = "*" if p_g < 0.05 else " "
        sig_phi = "*" if p_phi < 0.05 else " "
        print(f"  {t_eval:<4} {K_ahead:<8} {r_g:+.3f}{sig_g:<9} "
              f"{ndcg_g:<10.3f} {((g_pred - g_n) ** 2).mean():<10.4f} "
              f"{r_phi:+.3f}{sig_phi:<9} {ndcg_phi:<10.3f}")

        results.append({
            "t_eval": int(t_eval), "K_ahead": int(K_ahead),
            "growth_spearman_r": float(r_g), "growth_spearman_p": float(p_g),
            "growth_ndcg": float(ndcg_g), "growth_prec_at_10": float(prec_g),
            "growth_mse": float(((g_pred - g_n) ** 2).mean()),
            "growth_mae": float(np.abs(g_pred - g_n).mean()),
            "phi_spearman_r": float(r_phi), "phi_spearman_p": float(p_phi),
            "phi_ndcg": float(ndcg_phi), "phi_prec_at_10": float(prec_phi),
            "n_active": int(active.sum()),
        })
    return results


# ── main ───────────────────────────────────────────────────────────
def main():
    data = torch.load(f"{DATA_DIR}/alltime/fate_train.pt", weights_only=False)
    x = data["xp"]; y = data["y"]
    centroids = data["centroids"]
    growth_norm = data["growth_norm"]
    growth_raw = data["growth"]
    n_topics = data["n_topics"]

    T_data = len(y) - 1
    train_t, test_t = make_temporal_split(T_data, T_TRAIN_FRAC)

    args = make_args(train_t, test_t)
    args.n_topics = n_topics
    config = init_config(args)
    config.x_dim = x[0].shape[-1]
    config.n_topics = n_topics

    print("=" * 78)
    print("  X4-PREDICT  (Fourier time + causal training + future anchor + temporal hold-out)")
    print(f"  domain={DOMAIN}  seed={SEED}  epochs={EPOCHS}")
    print(f"  data has T={T_data} timepoints; train={train_t}, test={test_t}")
    print(f"  out_dir={config.out_dir}")
    print("=" * 78)

    config.save = min(EPOCHS, 500)
    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")
    if os.path.exists(existing_ckpt):
        print(f"  ✅ existing ckpt found, skip training")
    else:
        run_x4(args, config, x, y, centroids, growth_norm)
        if not os.path.exists(existing_ckpt):
            existing_ckpt = config.train_pt.format("best")

    # ── Eval ──
    print(f"\n=== Evaluation (X4-predict) ===\n  ckpt: {existing_ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    model._func = FourierAutoGenerator(config, K_fourier=K_FOURIER, T_max=T_MAX_FOURIER).to(device)
    predictor = PredictivePredictor(x_dim=config.x_dim, d_ctx=D_CTX).to(device)
    ckpt = torch.load(existing_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.eval(); predictor.eval()

    results = evaluate_x4(args, config, x, y, centroids, growth_raw, growth_norm,
                          model, predictor, device)

    out_eval = Path(config.out_dir) / "evaluation_x4.json"
    json.dump({"results": results, "settings": {
        "seed": SEED, "epochs": EPOCHS, "lam_growth": LAM_G, "lam_retro": LAM_RETRO,
        "k_fourier": K_FOURIER, "t_max_fourier": T_MAX_FOURIER,
        "k_predict_max": K_PREDICT_MAX, "t_train_frac": T_TRAIN_FRAC,
        "train_t": train_t, "test_t": test_t,
        "model": "X4-predict (Fourier time + causal + future anchor)",
    }}, out_eval.open("w"), indent=2)
    print(f"\nSaved -> {out_eval}")


if __name__ == "__main__":
    main()
