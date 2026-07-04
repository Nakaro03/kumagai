"""
PI-SDE + X1 + Direct Growth Head 実装。

経営者向け数値予測のため、Φ landscape に加えて
直接成長率を出力する head h_ψ を追加:

    ĝ_j(t) = h_ψ( c_j(t), t )    ∈ ℝ

評価指標 (追加):
    MSE, MAE, MAPE, R²

数式:
    h_ψ : MLP[D+1 → 64 → 64 → 1]
    L_growth = (1/T) Σ_j (ĝ_j - g_j)²
    L_total  = L_Sinkhorn + λ_HJ·L_HJ + λ_X1·L_X1 + λ_growth·L_growth

使用:
  PNODE_DOMAIN_TARGET=paper PNODE_SEED=42 PNODE_EPOCHS=300 \
    python -m pnode_patent_runner.run_pisde_growth_head
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 300))
LEAVEOUT_T = os.environ.get("PNODE_LEAVEOUT_T", "")
LAMBDA_X1  = float(os.environ.get("PNODE_LAMBDA_X1", 1.0))
LAMBDA_GROWTH = float(os.environ.get("PNODE_LAMBDA_GROWTH", 1.0))
GROWTH_HIDDEN = int(os.environ.get("PNODE_GROWTH_HIDDEN", 64))
# 過学習対策パラメータ (X1+Growth Head の汎化向上)
GROWTH_DROPOUT      = float(os.environ.get("PNODE_GROWTH_DROPOUT", 0.0))
GROWTH_WEIGHT_DECAY = float(os.environ.get("PNODE_GROWTH_WD", 0.0))
GROWTH_NLAYERS      = int(os.environ.get("PNODE_GROWTH_NLAYERS", 2))    # 1 or 2
# tag 名 (出力ディレクトリ識別用)
REG_TAG = ""
if GROWTH_DROPOUT > 0 or GROWTH_WEIGHT_DECAY > 0 or GROWTH_HIDDEN != 64 or GROWTH_NLAYERS != 2:
    REG_TAG = f"_h{GROWTH_HIDDEN}_l{GROWTH_NLAYERS}_d{GROWTH_DROPOUT}_wd{GROWTH_WEIGHT_DECAY}"

DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
if DOMAIN == "paper":
    DATA_NAME = "PNode_Paper_X1"
    DATA_DIR  = "data/PNode_Paper_X1"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "patent_energy_top50":
    DATA_NAME = "PNode_Patent_Energy_X1_top50"
    DATA_DIR  = "data/PNode_Patent_Energy_X1_top50"
    TRAIN_T   = list(range(1, 12))
elif DOMAIN == "patent_construction_top50":
    DATA_NAME = "PNode_Patent_Construction_X1_top50"
    DATA_DIR  = "data/PNode_Patent_Construction_X1_top50"
    TRAIN_T   = list(range(1, 12))
elif DOMAIN == "construction_papers":
    DATA_NAME = "PNode_Construction_X1"
    DATA_DIR  = "data/PNode_Construction_X1"
    TRAIN_T   = [1, 2, 3, 4, 5]   # 6 時点 (2020-2025), t=0 が初期
elif DOMAIN == "arxiv_construction":
    DATA_NAME = "PNode_ArXiv_Construction_X1_v2"
    DATA_DIR  = "data/PNode_ArXiv_Construction_X1_v2"
    TRAIN_T   = list(range(1, 11))   # 11 時点 (2015-2025), t=0..10
else:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")

OUT_TAG = f"x1_v1.0_g0.1_b0.01_GROWTH{REG_TAG}"


# ─────────────────────────────────────────────────────────────────
# Direct Growth Head
# ─────────────────────────────────────────────────────────────────
class GrowthHead(nn.Module):
    """
    数式:
        ĝ_j(t) = h_ψ( c_j(t), t )

    正則化版 (overfitting 対策):
      - hidden 縮小 (default 64 → option)
      - n_layers (1 or 2)
      - Dropout (default 0)
      - + weight_decay (optimizer 側で設定)
    """
    def __init__(self, d, hidden=64, n_layers=2, dropout=0.0):
        super().__init__()
        assert n_layers in (1, 2), "n_layers must be 1 or 2"
        layers = []
        # 第1層
        layers.append(nn.Linear(d + 1, hidden))
        layers.append(nn.Tanh())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        # 第2層 (option)
        if n_layers == 2:
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        # 出力層
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, c, t_val):
        t_col = torch.full((c.shape[0], 1), float(t_val), device=c.device)
        xt = torch.cat([c, t_col], dim=1)
        return self.mlp(xt).squeeze(-1)


# ─────────────────────────────────────────────────────────────────
# 評価指標 (経営者向け)
# ─────────────────────────────────────────────────────────────────
def regression_metrics(g_pred, g_true):
    """
    MSE, MAE, MAPE, R² を返す。
    """
    g_pred = np.asarray(g_pred)
    g_true = np.asarray(g_true)
    mask = np.isfinite(g_pred) & np.isfinite(g_true)
    if mask.sum() < 2:
        return {"mse": float("nan"), "mae": float("nan"),
                "mape": float("nan"), "r2": float("nan")}
    p, t = g_pred[mask], g_true[mask]
    mse  = float(((p - t) ** 2).mean())
    mae  = float(np.abs(p - t).mean())
    # MAPE: |真値| が小さいトピックは除外 (>0.01)
    mape_mask = np.abs(t) > 0.01
    if mape_mask.sum() > 1:
        mape = float((np.abs(p[mape_mask] - t[mape_mask]) /
                       (np.abs(t[mape_mask]) + 1e-8)).mean() * 100)
    else:
        mape = float("nan")
    # R²
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
    return {"mse": mse, "mae": mae, "mape": mape, "r2": r2}


def make_args():
    if LEAVEOUT_T:
        t = int(LEAVEOUT_T)
        eff_train_t = [tt for tt in TRAIN_T if tt != t]
        leaveout_tag = f"leaveout{t}"
    else:
        eff_train_t = TRAIN_T
        leaveout_tag = "alltime"

    ns = argparse.Namespace(
        seed=SEED, use_cuda=True, device=0,
        out_dir=f"RESULTS/{DATA_NAME}_GROWTH",
        data=DATA_NAME,
        data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        k_dims=[400, 400], activation="softplus",
        sigma_type="const", sigma_const=0.1,
        train_epochs=EPOCHS, train_lr=0.005,
        train_lambda=0.5, train_batch=0.1, train_clip=0.1,
        save=min(EPOCHS, 500),
        evaluate_n=10000, evaluate_data=None, evaluate_baseline=False,
        task="leaveout" if LEAVEOUT_T else "fate",
        train=False, evaluate=None, config=None,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, ns=2000,
        start_t=0, train_t=eff_train_t,
        leaveout_t=leaveout_tag if LEAVEOUT_T else "",
        test_t=[int(LEAVEOUT_T)] if LEAVEOUT_T else [],
    )
    ns.layers = len(ns.k_dims)
    return ns


def init_config(args):
    args.layers = len(args.k_dims)
    args.kDims = '_'.join(map(str, args.k_dims))
    name = (f"{args.activation}-{args.kDims}-"
            f"{args.train_lambda}-{args.sigma_type}-{args.sigma_const}-"
            f"{args.train_clip}-{args.train_lr}-{OUT_TAG}")
    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}")
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")
    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt  = os.path.join(args.out_dir, "train.{}.pt")
    args.growth_pt = os.path.join(args.out_dir, "growth_head.pt")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


# ─────────────────────────────────────────────────────────────────
# X1 損失計算 (run_pisde_x1 から流用)
# ─────────────────────────────────────────────────────────────────
def compute_x1_loss(model, centroids_t, growth_norm_t, t_val, device,
                    lam_v=1.0, lam_g=0.1, lam_b=0.01,
                    alpha=1.0, basin_sigma=0.1, n_eps=8):
    T_topic, D = centroids_t.shape
    active = centroids_t.abs().sum(dim=-1) > 1e-6
    if active.sum() < 2:
        return torch.tensor(0.0, device=device)
    c = centroids_t[active].to(device).requires_grad_(True)
    g_norm = growth_norm_t[active].to(device)
    t_col = torch.ones(c.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([c, t_col], dim=1)
    phi_c = model._func._pot(xt).squeeze(-1)
    target = -alpha * g_norm
    L_val = ((phi_c - target) ** 2).mean()

    grad_xt = torch.autograd.grad(phi_c.sum(), xt, create_graph=True)[0]
    grad_x  = grad_xt[:, :-1]
    L_grad = (grad_x ** 2).sum(dim=-1).mean()

    eps = torch.randn(n_eps, c.shape[0], c.shape[1], device=device) * basin_sigma
    c_perturbed = c.unsqueeze(0) + eps
    c_pert_flat = c_perturbed.view(-1, c.shape[1])
    t_pert = torch.ones(c_pert_flat.shape[0], 1, device=device) * float(t_val)
    xt_pert = torch.cat([c_pert_flat, t_pert], dim=1)
    phi_pert = model._func._pot(xt_pert).squeeze(-1).view(n_eps, c.shape[0])
    diff = phi_pert - phi_c.unsqueeze(0)
    L_basin = torch.nn.functional.relu(-diff).mean()

    return lam_v * L_val + lam_g * L_grad + lam_b * L_basin


# ─────────────────────────────────────────────────────────────────
# 学習ループ (X1 + Growth)
# ─────────────────────────────────────────────────────────────────
def train_with_growth(args, config, x, y, centroids, growth_norm, growth_raw,
                       leaveouts=None):
    from src.model import ForwardSDE
    import src.train as pisde_train
    from src.train import OTLoss
    import tqdm

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = ForwardSDE(config).to(device)
    growth_head = GrowthHead(
        d=config.x_dim, hidden=GROWTH_HIDDEN,
        n_layers=GROWTH_NLAYERS, dropout=GROWTH_DROPOUT,
    ).to(device)

    loss_fn = OTLoss(config, device)
    # PI-SDE 本体は通常の Adam, Growth Head のみ weight_decay
    optimizer = optim.AdamW([
        {"params": list(model.parameters()), "weight_decay": 0.0},
        {"params": list(growth_head.parameters()), "weight_decay": GROWTH_WEIGHT_DECAY},
    ], lr=config.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    torch.save(config.__dict__, config.config_pt)

    pbar = tqdm.tqdm(range(config.train_epochs))
    best_loss = np.inf
    log = open(config.train_log, "w")

    train_t = config.train_t

    for epoch in pbar:
        model.zero_grad(); growth_head.zero_grad()
        losses_xy, losses_r, losses_x1, losses_growth = [], [], [], []

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

            # X1 (Topic-Anchor) loss
            if LAMBDA_X1 > 0:
                L_x1 = compute_x1_loss(model, centroids[j], growth_norm[j], y[j], device)
                loss_step = loss_step + LAMBDA_X1 * L_x1
                losses_x1.append(L_x1.item())

            # Growth Head loss
            if LAMBDA_GROWTH > 0:
                cents_j = centroids[j]
                active = cents_j.abs().sum(dim=-1) > 1e-6
                if active.sum() > 2:
                    c_active = cents_j[active].to(device)
                    g_true = growth_raw[j][active].to(device)
                    g_pred = growth_head(c_active, y[j])
                    L_growth = ((g_pred - g_true) ** 2).mean()
                    loss_step = loss_step + LAMBDA_GROWTH * L_growth
                    losses_growth.append(L_growth.item())

            loss_step.backward(retain_graph=True)

        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(growth_head.parameters()),
                config.train_clip,
            )
        optimizer.step(); scheduler.step()
        model.zero_grad(); growth_head.zero_grad()

        train_xy = np.mean(losses_xy)
        train_x1 = np.mean(losses_x1) if losses_x1 else 0.0
        train_gr = np.mean(losses_growth) if losses_growth else 0.0

        desc = f"[ep {epoch+1}] Sink={train_xy:.3f} X1={train_x1:.3f} Grow={train_gr:.4f}"
        pbar.set_description(desc)
        log.write(desc + "\n"); log.flush()

        if train_xy < best_loss:
            best_loss = train_xy
            torch.save({"model_state_dict": model.state_dict(),
                         "growth_head_state_dict": growth_head.state_dict(),
                         "epoch": epoch + 1}, config.train_pt.format("best"))

        if (epoch + 1) % config.save == 0:
            ep_str = str(epoch + 1).rjust(6, "0")
            torch.save({"model_state_dict": model.state_dict(),
                         "growth_head_state_dict": growth_head.state_dict(),
                         "epoch": epoch + 1}, config.train_pt.format(f"epoch_{ep_str}"))

    log.close()
    # 最終 growth head を別ファイルに保存
    torch.save(growth_head.state_dict(), config.growth_pt)
    return config, model, growth_head


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    args = make_args()
    config = init_config(args)
    data = torch.load(args.data_path, weights_only=False)
    x = data["xp"]; y = data["y"]
    centroids = data["centroids"]
    growth_norm = data["growth_norm"]
    growth_raw  = data["growth"]
    config.x_dim = x[0].shape[-1]

    print("=" * 70)
    print(f"  PI-SDE + X1 + Direct Growth Head")
    print(f"  domain={DOMAIN}  seed={SEED}  epochs={EPOCHS}  leaveout={LEAVEOUT_T}")
    print(f"  λ_X1={LAMBDA_X1}  λ_growth={LAMBDA_GROWTH}  h={GROWTH_HIDDEN}")
    print(f"  out_dir={config.out_dir}")
    print("=" * 70)

    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")
    if os.path.exists(existing_ckpt):
        print(f"  ✅ checkpoint 存在: {existing_ckpt}")
        from src.model import ForwardSDE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ForwardSDE(config).to(device)
        growth_head = GrowthHead(d=config.x_dim, hidden=GROWTH_HIDDEN).to(device)
        ckpt = torch.load(existing_ckpt, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        growth_head.load_state_dict(ckpt["growth_head_state_dict"])
    else:
        config, model, growth_head = train_with_growth(
            args, config, x, y, centroids, growth_norm, growth_raw,
            leaveouts=[int(LEAVEOUT_T)] if LEAVEOUT_T else None,
        )

    model.eval(); growth_head.eval()

    # ── 評価: Sinkhorn + ranking + 直接予測指標 ────────────────
    from geomloss import SamplesLoss
    from scipy import stats
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                             scaling=config.sinkhorn_scaling)

    def eval_t(t_cur):
        # Sinkhorn
        import src.train as pisde_train
        n_eval = min(2000, x[0].shape[0])
        x_0, _ = pisde_train.p_samp(x[0], n_eval)
        r_0 = torch.zeros(n_eval).unsqueeze(1)
        x_r_0 = torch.cat([x_0, r_0], dim=1).to(device).requires_grad_()
        x_r_s = model([np.float64(y[0])] + [np.float64(y[t_cur])], x_r_0)
        x_pred = x_r_s[-1][:, :-1].detach()
        x_obs = x[t_cur].to(device)
        n_obs = min(2000, x_obs.shape[0])
        idx = np.random.choice(x_obs.shape[0], n_obs, replace=False)
        sink = ot_solver(x_pred.contiguous(), x_obs[idx].detach().contiguous()).item()

        # Φ at centroids → ranking
        c_cpu = centroids[t_cur]
        active = c_cpu.abs().sum(dim=-1) > 1e-6
        c_a = c_cpu[active].to(device).requires_grad_()
        g_true = growth_raw[t_cur][active].numpy()
        t_col = torch.ones(c_a.shape[0], 1, device=device) * float(y[t_cur])
        xt = torch.cat([c_a, t_col], dim=1)
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()

        # Direct Growth prediction
        with torch.no_grad():
            g_pred = growth_head(c_a.detach(), y[t_cur]).cpu().numpy()

        # Spearman / NDCG / P@10
        r, p_sp = stats.spearmanr(phi, g_true)
        K = min(10, len(g_true))
        order = np.argsort(phi)
        rel = np.maximum(g_true, 0.0)
        dcg = sum(rel[order[k]] / np.log2(k + 2) for k in range(K))
        ideal = np.argsort(-rel)
        idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
        ndcg = dcg / (idcg + 1e-10) if idcg > 0 else float("nan")
        prec = float((g_true[order[:K]] > 0).mean())

        # Direct growth metrics
        reg_m = regression_metrics(g_pred, g_true)

        # Spearman for direct prediction (ĝ vs g)
        r_dir, p_dir = stats.spearmanr(g_pred, g_true)

        return {
            "sinkhorn": sink,
            "spearman_phi": float(r), "spearman_phi_p": float(p_sp),
            "ndcg": float(ndcg), "prec_at_10": prec,
            "spearman_growth": float(r_dir), "spearman_growth_p": float(p_dir),
            **reg_m,
        }

    results = []
    print(f"\n  {'t':<3} {'split':<6} | {'Sink':<7} {'Sp(Φ)':<10} {'Sp(ĝ)':<10} | {'MSE':<8} {'MAE':<7} {'MAPE%':<8} {'R²':<8}")
    print("  " + "-" * 90)
    for t in range(1, len(x)):
        split = "test" if (LEAVEOUT_T and t == int(LEAVEOUT_T)) else "train"
        m = eval_t(t)
        sig_phi = "*" if m["spearman_phi_p"] < 0.05 else " "
        sig_dir = "*" if m["spearman_growth_p"] < 0.05 else " "
        print(f"  {t:<3} {split:<6} | {m['sinkhorn']:<7.3f} "
              f"{m['spearman_phi']:+.3f}{sig_phi}  "
              f"{m['spearman_growth']:+.3f}{sig_dir}  | "
              f"{m['mse']:<8.4f} {m['mae']:<7.4f} {m['mape']:<8.2f} {m['r2']:<+8.3f}")
        results.append({"t": t, "split": split, **m})

    out = Path(config.out_dir) / "evaluation_growth.json"
    json.dump({"results": results, "settings": {
        "domain": DOMAIN, "seed": SEED, "epochs": EPOCHS, "leaveout_t": LEAVEOUT_T,
    }}, out.open("w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
