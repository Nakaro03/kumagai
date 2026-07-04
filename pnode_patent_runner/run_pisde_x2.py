"""
X2 PI-SDE: Multi-Task + Cross-Topic Attention + Monte Carlo Uncertainty

X1 (Topic-Anchored Potential) backbone を維持し、以下を追加:
  1. TopicCrossAttention: K 個の centroid 間で context を伝搬 (Transformer 1 層)
  2. GrowthHead: 絶対成長率 g を回帰予測 (multi-task)
  3. Monte Carlo SDE: 推論時に Φ ± uncertainty を計算

評価指標 (追加):
  - rank: Spearman, NDCG@10, P@10 (X1 と同じ)
  - regression: MSE, MAE on g_norm  (新規)
  - uncertainty: Φ_std per centroid  (新規)

Usage:
  PNODE_SEED=42 PNODE_EPOCHS=300 PNODE_DOMAIN_TARGET=paper \\
    PNODE_LAMBDA_X1=1.0 PNODE_LAM_GROWTH=0.5 \\
    python run_pisde_x2.py
"""
from __future__ import annotations

import os, sys, json, argparse, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from scipy import stats

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

# ── 設定 ──────────────────────────────────────────────────────
SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 300))
LEAVEOUT_T = os.environ.get("PNODE_LEAVEOUT_T", "")
LAMBDA_X1  = float(os.environ.get("PNODE_LAMBDA_X1", 1.0))
LAM_V      = float(os.environ.get("PNODE_LAM_V", 1.0))
LAM_G      = float(os.environ.get("PNODE_LAM_G", 0.1))
LAM_B      = float(os.environ.get("PNODE_LAM_B", 0.01))
LAM_GROWTH = float(os.environ.get("PNODE_LAM_GROWTH", 0.5))  # NEW
ALPHA_VAL  = float(os.environ.get("PNODE_ALPHA_VAL", 1.0))
BASIN_SIGMA = float(os.environ.get("PNODE_BASIN_SIGMA", 0.1))
D_CTX       = int(os.environ.get("PNODE_D_CTX", 32))         # attention dim
N_HEADS     = int(os.environ.get("PNODE_N_HEADS", 2))        # attention heads
N_MC        = int(os.environ.get("PNODE_N_MC", 16))          # MC samples at eval

DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
if DOMAIN == "paper":
    DATA_NAME = "PNode_Paper_X1"; DATA_DIR = "data/PNode_Paper_X1"
    TRAIN_T = [1, 2, 3]
elif DOMAIN == "patent_energy_top50":
    DATA_NAME = "PNode_Patent_Energy_X1_top50"; DATA_DIR = "data/PNode_Patent_Energy_X1_top50"
    TRAIN_T = list(range(1, 12))
elif DOMAIN == "arxiv_construction":
    DATA_NAME = "PNode_ArXiv_Construction_X1_v2"; DATA_DIR = "data/PNode_ArXiv_Construction_X1_v2"
    TRAIN_T = list(range(1, 11))
elif DOMAIN == "jp_construction":
    DATA_NAME = "PNode_JP_Construction_X1"; DATA_DIR = "data/PNode_JP_Construction_X1"
    TRAIN_T = list(range(1, 11))
else:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")


def make_args():
    if LEAVEOUT_T:
        leaveout_ts = [int(t.strip()) for t in LEAVEOUT_T.split(",") if t.strip()]
        eff_train_t = [tt for tt in TRAIN_T if tt not in leaveout_ts]
        leaveout_tag = "leaveout" + "_".join(str(t) for t in leaveout_ts)
        test_ts = leaveout_ts
    else:
        eff_train_t = TRAIN_T
        leaveout_tag = "alltime"
        test_ts = []

    ns = argparse.Namespace(
        seed=SEED, use_cuda=True, device=0,
        out_dir=f"RESULTS_X2/{DATA_NAME}",
        data=DATA_NAME, data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        k_dims=[400, 400], activation="softplus",
        sigma_type="const", sigma_const=0.1,
        train_epochs=EPOCHS, train_lr=0.005,
        train_lambda=0.5, train_batch=0.1, train_clip=0.1, save=500,
        evaluate_n=10000, evaluate_data=None, evaluate_baseline=False,
        task="leaveout" if LEAVEOUT_T else "fate",
        train=False, evaluate=None, config=None,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, ns=2000,
        start_t=0, train_t=eff_train_t,
        leaveout_t=leaveout_tag if LEAVEOUT_T else "",
        test_t=test_ts,
    )
    ns.layers = len(ns.k_dims)
    return ns


def init_config(args):
    args.layers = len(args.k_dims)
    args.kDims = '_'.join(map(str, args.k_dims))
    name = ("{activation}-{kDims}-{train_lambda}-{sigma_type}-{sigma_const}-"
            "{train_clip}-{train_lr}").format(**args.__dict__)
    name += f"-x2_v{LAM_V}_g{LAM_G}_b{LAM_B}_gh{LAM_GROWTH}"
    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}")
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")
    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt = os.path.join(args.out_dir, "train.{}.pt")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


# ─────────────────────────────────────────────────────────────────
# NEW: TopicCrossAttention
# ─────────────────────────────────────────────────────────────────
class TopicCrossAttention(nn.Module):
    """K 個の topic centroid 間で context を伝搬する 1 層 Transformer."""

    def __init__(self, x_dim, d_model=32, n_heads=2):
        super().__init__()
        # 入力: centroid (x_dim) + growth_norm (1) + time (1) = x_dim+2
        self.embed = nn.Linear(x_dim + 2, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, centroids, growth_norm, t_val):
        """centroids: (K, x_dim), growth_norm: (K,), t_val: float -> (K, d_model)"""
        K, D = centroids.shape
        t_col = torch.full((K, 1), float(t_val), device=centroids.device)
        g_col = growth_norm.unsqueeze(-1)
        x = torch.cat([centroids, g_col, t_col], dim=-1)            # (K, x_dim+2)
        x = self.embed(x).unsqueeze(0)                              # (1, K, d_model)
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ffn(x))
        return x.squeeze(0)                                          # (K, d_model)


# ─────────────────────────────────────────────────────────────────
# NEW: GrowthHead (absolute growth regression)
# ─────────────────────────────────────────────────────────────────
class GrowthHead(nn.Module):
    """(centroid, t, attention context) → predicted growth_norm value."""

    def __init__(self, x_dim, d_ctx, hidden=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(x_dim + 1 + d_ctx, hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, centroids, t_val, ctx):
        K = centroids.shape[0]
        t_col = torch.full((K, 1), float(t_val), device=centroids.device)
        x = torch.cat([centroids, t_col, ctx], dim=-1)
        return self.mlp(x).squeeze(-1)                               # (K,)


# ─────────────────────────────────────────────────────────────────
# X2 損失: X1 + L_growth (multi-task)
# ─────────────────────────────────────────────────────────────────
def compute_x2_loss(model, attention, growth_head,
                    centroids_t, growth_norm_t, t_val, device,
                    lam_v=1.0, lam_g=0.1, lam_b=0.01, lam_growth=0.5,
                    alpha=1.0, basin_sigma=0.1, n_eps=8):
    T, D = centroids_t.shape
    active = centroids_t.abs().sum(dim=-1) > 1e-6
    if active.sum() < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {"val": 0, "grad": 0, "basin": 0, "growth": 0}

    c = centroids_t[active].to(device).requires_grad_(True)
    g_n = growth_norm_t[active].to(device)
    t_col = torch.ones(c.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([c, t_col], dim=1)

    # ── L_X1 components (unchanged) ──
    phi_c = model._func._pot(xt).squeeze(-1)
    target = -alpha * g_n
    L_val = ((phi_c - target) ** 2).mean()
    grad_xt = torch.autograd.grad(phi_c.sum(), xt, create_graph=True)[0]
    grad_x = grad_xt[:, :-1]
    L_grad = (grad_x ** 2).sum(dim=-1).mean()
    eps = torch.randn(n_eps, c.shape[0], c.shape[1], device=device) * basin_sigma
    c_pert = c.unsqueeze(0) + eps
    c_pert_flat = c_pert.view(-1, c.shape[1])
    t_pert = torch.ones(c_pert_flat.shape[0], 1, device=device) * float(t_val)
    xt_pert = torch.cat([c_pert_flat, t_pert], dim=1)
    phi_pert = model._func._pot(xt_pert).squeeze(-1).view(n_eps, c.shape[0])
    diff = phi_pert - phi_c.unsqueeze(0)
    L_basin = torch.nn.functional.relu(-diff).mean()
    L_x1 = lam_v * L_val + lam_g * L_grad + lam_b * L_basin

    # ── NEW: cross-topic attention + growth regression ──
    ctx = attention(c, g_n, t_val)                                   # (K, d_ctx)
    g_pred = growth_head(c, t_val, ctx)                              # (K,)
    L_growth = ((g_pred - g_n) ** 2).mean()

    L_total = L_x1 + lam_growth * L_growth
    return L_total, {
        "val": L_val.item(), "grad": L_grad.item(), "basin": L_basin.item(),
        "growth": L_growth.item(),
    }


# ─────────────────────────────────────────────────────────────────
# 学習ループ
# ─────────────────────────────────────────────────────────────────
def run_x2(args, config, x, y, centroids, growth_norm):
    from src.model import ForwardSDE
    import src.train as pisde_train
    from src.train import OTLoss

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_t = config.train_t
    print(f"  train_t = {train_t}")

    model = ForwardSDE(config).to(device)
    attention = TopicCrossAttention(x_dim=config.x_dim, d_model=D_CTX,
                                    n_heads=N_HEADS).to(device)
    growth_head = GrowthHead(x_dim=config.x_dim, d_ctx=D_CTX,
                             hidden=64, dropout=0.1).to(device)

    loss_fn = OTLoss(config, device)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "weight_decay": 0.0},
        {"params": attention.parameters(), "weight_decay": 1e-4},
        {"params": growth_head.parameters(), "weight_decay": 1e-3},
    ], lr=config.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    torch.save(config.__dict__, config.config_pt)

    import tqdm
    pbar = tqdm.tqdm(range(config.train_epochs))
    best_loss = np.inf
    log = open(config.train_log, "w")

    for epoch in pbar:
        for m in (model, attention, growth_head): m.zero_grad()
        losses_xy, losses_r, losses_x2 = [], [], []
        comp_acc = {"val": 0, "grad": 0, "basin": 0, "growth": 0, "n": 0}

        dat_prev = x[config.start_t]
        x_i, _ = pisde_train.p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
        r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
        x_r_i = torch.cat([x_i, r_i], dim=1).to(device)
        ts = [0] + train_t
        y_ts = [np.float64(y[ts_i]) for ts_i in ts]
        x_r_s = model(y_ts, x_r_i)

        for j in train_t:
            dat_cur = x[j]
            y_j, _ = pisde_train.p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
            position = train_t.index(j)
            a_i = torch.ones(x_i.shape[0]); a_i = a_i / a_i.sum()
            b_j = torch.ones(y_j.shape[0]); b_j = b_j / b_j.sum()

            loss_xy = loss_fn(a_i, x_r_s[position + 1][:, 0:-1], b_j, y_j)
            losses_xy.append(loss_xy.item())

            if (config.train_lambda > 0) & (j == train_t[-1]):
                loss_r = torch.mean(x_r_s[-1][:, -1] * config.train_lambda)
                losses_r.append(loss_r.item())
                loss_step = loss_xy + loss_r
            else:
                loss_step = loss_xy

            if LAMBDA_X1 > 0:
                L_x2, c2 = compute_x2_loss(
                    model, attention, growth_head,
                    centroids[j], growth_norm[j], y[j], device,
                    lam_v=LAM_V, lam_g=LAM_G, lam_b=LAM_B, lam_growth=LAM_GROWTH,
                    alpha=ALPHA_VAL, basin_sigma=BASIN_SIGMA,
                )
                loss_step = loss_step + LAMBDA_X1 * L_x2
                losses_x2.append(L_x2.item())
                for k in ("val", "grad", "basin", "growth"):
                    comp_acc[k] += c2[k]
                comp_acc["n"] += 1

            loss_step.backward(retain_graph=True)

        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(attention.parameters())
                + list(growth_head.parameters()),
                config.train_clip,
            )
        optimizer.step()
        scheduler.step()
        for m in (model, attention, growth_head): m.zero_grad()

        train_loss_xy = np.mean(losses_xy)
        train_loss_r = np.mean(losses_r) if losses_r else 0.0
        train_loss_x2 = np.mean(losses_x2) if losses_x2 else 0.0
        n = max(comp_acc["n"], 1)
        desc = (f"[ep {epoch+1}] Sink={train_loss_xy:.3f} HJ={train_loss_r:.4f} "
                f"X2={train_loss_x2:.4f} g={comp_acc['growth']/n:.4f}")
        pbar.set_description(desc)
        log.write(desc + "\n"); log.flush()

        if train_loss_xy < best_loss:
            best_loss = train_loss_xy
            torch.save({
                "model_state_dict": model.state_dict(),
                "attention_state_dict": attention.state_dict(),
                "growth_head_state_dict": growth_head.state_dict(),
                "epoch": epoch + 1,
            }, config.train_pt.format("best"))

        if (epoch + 1) % config.save == 0:
            ep_str = str(epoch + 1).rjust(6, "0")
            torch.save({
                "model_state_dict": model.state_dict(),
                "attention_state_dict": attention.state_dict(),
                "growth_head_state_dict": growth_head.state_dict(),
                "epoch": epoch + 1,
            }, config.train_pt.format(f"epoch_{ep_str}"))

    log.close()
    return config


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    args = make_args()
    config = init_config(args)

    data = torch.load(args.data_path, weights_only=False)
    x = data["xp"]
    y = data["y"]
    centroids = data["centroids"]
    growth_norm = data["growth_norm"]
    growth_raw = data["growth"]
    topic_names = data.get("topic_names", [])
    config.x_dim = x[0].shape[-1]
    config.n_topics = data["n_topics"]

    print("=" * 78)
    print("  X2 PI-SDE  (Multi-Task + Cross-Topic Attention + MC Uncertainty)")
    print(f"  seed={SEED}  epochs={EPOCHS}  leaveout={LEAVEOUT_T}")
    print(f"  λ_X1={LAMBDA_X1}  (val={LAM_V}, grad={LAM_G}, basin={LAM_B}, growth={LAM_GROWTH})")
    print(f"  d_ctx={D_CTX}  n_heads={N_HEADS}  n_MC={N_MC}")
    print(f"  out_dir={config.out_dir}")
    print("=" * 78)

    config.save = min(EPOCHS, 500)
    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")
    if os.path.exists(existing_ckpt):
        print(f"  ✅ existing ckpt found, skip training: {existing_ckpt}")
    else:
        run_x2(args, config, x, y, centroids, growth_norm)
        if not os.path.exists(existing_ckpt):
            existing_ckpt = config.train_pt.format("best")

    # ── Evaluation ──
    print(f"\n=== Evaluation ===\n  ckpt: {existing_ckpt}")
    from src.model import ForwardSDE
    import src.train as pisde_train
    from geomloss import SamplesLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    attention = TopicCrossAttention(x_dim=config.x_dim, d_model=D_CTX, n_heads=N_HEADS).to(device)
    growth_head = GrowthHead(x_dim=config.x_dim, d_ctx=D_CTX, hidden=64, dropout=0.1).to(device)
    ckpt = torch.load(existing_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    attention.load_state_dict(ckpt["attention_state_dict"])
    growth_head.load_state_dict(ckpt["growth_head_state_dict"])
    model.eval(); attention.eval(); growth_head.eval()

    ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                            scaling=config.sinkhorn_scaling)

    def eval_sinkhorn(t_cur):
        n_eval = min(2000, x[0].shape[0])
        x_0, _ = pisde_train.p_samp(x[0], n_eval)
        r_0 = torch.zeros(n_eval).unsqueeze(1)
        x_r_0 = torch.cat([x_0, r_0], dim=1).to(device).requires_grad_()
        x_r_s = model([np.float64(y[0])] + [np.float64(y[t_cur])], x_r_0)
        x_pred = x_r_s[-1][:, :-1].detach()
        x_obs = x[t_cur].to(device)
        n_obs = min(2000, x_obs.shape[0])
        idx = np.random.choice(x_obs.shape[0], n_obs, replace=False)
        return ot_solver(x_pred.contiguous(), x_obs[idx].detach().contiguous()).item()

    def eval_combined(t_cur):
        """Φ + g_pred + MC uncertainty を計算"""
        c_cpu = centroids[t_cur]
        active_cpu = c_cpu.abs().sum(dim=-1) > 1e-6
        c_a = c_cpu[active_cpu].to(device).requires_grad_()
        g_a = growth_raw[t_cur][active_cpu].numpy()
        g_norm_a = growth_norm[t_cur][active_cpu].to(device)
        t_col = torch.ones(c_a.shape[0], 1, device=device) * float(y[t_cur])
        xt = torch.cat([c_a, t_col], dim=1)

        # Φ (point estimate)
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()

        # Growth head prediction
        ctx = attention(c_a, g_norm_a, float(y[t_cur]))
        g_pred = growth_head(c_a, float(y[t_cur]), ctx).detach().cpu().numpy()

        # Monte Carlo: perturb input slightly to get Φ uncertainty
        sigma_mc = 0.05
        phi_samples = []
        for _ in range(N_MC):
            c_pert = c_a + torch.randn_like(c_a) * sigma_mc
            xt_p = torch.cat([c_pert, t_col], dim=1)
            phi_p = model._func._pot(xt_p).squeeze(-1).detach().cpu().numpy()
            phi_samples.append(phi_p)
        phi_std = np.std(np.stack(phi_samples), axis=0)

        # Rank metric (Φ-based, X1 style)
        r_phi, p_phi = stats.spearmanr(phi, g_a)
        order_phi = np.argsort(phi)
        K = min(10, len(g_a))
        rel = np.maximum(g_a, 0.0)
        dcg = sum(rel[order_phi[k]] / np.log2(k + 2) for k in range(K))
        ideal = np.argsort(-rel)
        idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
        ndcg_phi = dcg / (idcg + 1e-10) if idcg > 0 else float("nan")
        prec_phi = (g_a[order_phi[:K]] > 0).mean()

        # Rank metric (growth_head-based, X2 NEW)
        r_g, p_g = stats.spearmanr(g_pred, g_a)
        order_g = np.argsort(-g_pred)
        dcg_g = sum(rel[order_g[k]] / np.log2(k + 2) for k in range(K))
        ndcg_g = dcg_g / (idcg + 1e-10) if idcg > 0 else float("nan")
        prec_g = (g_a[order_g[:K]] > 0).mean()

        # Regression metric (g_pred vs g_norm_actual)
        g_norm_actual = g_norm_a.cpu().numpy()
        mse = float(((g_pred - g_norm_actual) ** 2).mean())
        mae = float(np.abs(g_pred - g_norm_actual).mean())

        return {
            "phi_spearman_r": float(r_phi), "phi_spearman_p": float(p_phi),
            "phi_ndcg": float(ndcg_phi), "phi_prec_at_10": float(prec_phi),
            "growth_spearman_r": float(r_g), "growth_spearman_p": float(p_g),
            "growth_ndcg": float(ndcg_g), "growth_prec_at_10": float(prec_g),
            "growth_mse": mse, "growth_mae": mae,
            "phi_std_mean": float(phi_std.mean()),
            "n_active": int(active_cpu.sum()),
        }

    results = []
    print(f"\n  {'t':<4} {'split':<6} {'Sink':<8} {'φ-Spear':<11} {'φ-P@10':<8} "
          f"{'g-Spear':<11} {'g-P@10':<8} {'g-MSE':<8} {'φ-std':<8}")
    print("  " + "-" * 80)
    for t in range(1, len(x)):
        _loset = set(int(s.strip()) for s in LEAVEOUT_T.split(",") if s.strip()) if LEAVEOUT_T else set()
        split = "test" if (t in _loset) else "train"
        sink = eval_sinkhorn(t)
        r = eval_combined(t)
        sig_phi = "*" if r["phi_spearman_p"] < 0.05 else " "
        sig_g = "*" if r["growth_spearman_p"] < 0.05 else " "
        print(f"  {t:<4} {split:<6} {sink:<8.3f} "
              f"{r['phi_spearman_r']:+.3f}{sig_phi:<3} {r['phi_prec_at_10']:<8.2f} "
              f"{r['growth_spearman_r']:+.3f}{sig_g:<3} {r['growth_prec_at_10']:<8.2f} "
              f"{r['growth_mse']:<8.3f} {r['phi_std_mean']:<8.3f}")
        results.append({"t": t, "split": split, "sinkhorn": sink, **r})

    out_eval = Path(config.out_dir) / "evaluation_x2.json"
    json.dump({"results": results, "settings": {
        "seed": SEED, "epochs": EPOCHS, "leaveout_t": LEAVEOUT_T,
        "lambda_x1": LAMBDA_X1, "lam_v": LAM_V, "lam_g": LAM_G, "lam_b": LAM_B,
        "lam_growth": LAM_GROWTH, "d_ctx": D_CTX, "n_heads": N_HEADS, "n_mc": N_MC,
    }}, out_eval.open("w"), indent=2)
    print(f"\nSaved -> {out_eval}")


if __name__ == "__main__":
    main()
