"""
PI-SDE + X1 (Topic-Anchored Potential) 学習スクリプト。

PI-SDE 元コードベース (/tmp/PI-SDE/) を流用し、L_X1 を追加。

L_X1 = λ_v · L_anchor_val + λ_g · L_anchor_grad + λ_b · L_basin
     L_anchor_val  = Σ_j (Φ(c_j(t), t) + α·g̃_j(t))²
     L_anchor_grad = Σ_j ‖∇_x Φ(c_j(t), t)‖²
     L_basin       = Σ_j E_ε[Φ(c_j) − Φ(c_j+ε)]_+

使用:
  PNODE_SEED=42 PNODE_EPOCHS=500 PNODE_LEAVEOUT_T="" python -m pnode_patent_runner.run_pisde_x1
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import optim

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

# ── 設定 ─────────────────────────────────────────────────────────
SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 500))
LEAVEOUT_T = os.environ.get("PNODE_LEAVEOUT_T", "")
LAMBDA_X1  = float(os.environ.get("PNODE_LAMBDA_X1", 1.0))
LAM_V      = float(os.environ.get("PNODE_LAM_V", 1.0))
LAM_G      = float(os.environ.get("PNODE_LAM_G", 0.1))
LAM_B      = float(os.environ.get("PNODE_LAM_B", 0.01))
ALPHA_VAL  = float(os.environ.get("PNODE_ALPHA_VAL", 1.0))
BASIN_SIGMA = float(os.environ.get("PNODE_BASIN_SIGMA", 0.1))

# ドメイン切替 (paper / patent_energy)
DOMAIN = os.environ.get("PNODE_DOMAIN_TARGET", "paper")
if DOMAIN == "paper":
    DATA_NAME = "PNode_Paper_X1"
    DATA_DIR  = "data/PNode_Paper_X1"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "paper_pca25":
    DATA_NAME = "PNode_Paper_X1_pca25"
    DATA_DIR  = "data/PNode_Paper_X1_pca25"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "paper_pca100":
    DATA_NAME = "PNode_Paper_X1_pca100"
    DATA_DIR  = "data/PNode_Paper_X1_pca100"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "patent_energy":
    DATA_NAME = "PNode_Patent_Energy_X1"
    DATA_DIR  = "data/PNode_Patent_Energy_X1"
    TRAIN_T   = list(range(1, 12))   # 12 年, t=1..11
elif DOMAIN == "patent_energy_top50":
    DATA_NAME = "PNode_Patent_Energy_X1_top50"
    DATA_DIR  = "data/PNode_Patent_Energy_X1_top50"
    TRAIN_T   = list(range(1, 12))
elif DOMAIN == "patent_construction_top50":
    DATA_NAME = "PNode_Patent_Construction_X1_top50"
    DATA_DIR  = "data/PNode_Patent_Construction_X1_top50"
    TRAIN_T   = list(range(1, 12))
elif DOMAIN == "arxiv_construction":
    DATA_NAME = "PNode_ArXiv_Construction_X1_v2"
    DATA_DIR  = "data/PNode_ArXiv_Construction_X1_v2"
    TRAIN_T   = list(range(1, 11))   # 11 時点 (2015-2025), t=0..10
elif DOMAIN == "jp_construction":
    DATA_NAME = "PNode_JP_Construction_X1"
    DATA_DIR  = "data/PNode_JP_Construction_X1"
    TRAIN_T   = list(range(1, 11))   # 11 時点 (2015-2025), t=0..10
else:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")


def make_args():
    if LEAVEOUT_T:
        # Comma-separated multi-leaveout supported, e.g. "10,11" for k-step ahead
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
        out_dir=f"RESULTS/{DATA_NAME}",
        data=DATA_NAME,
        data_path=f"{DATA_DIR}/alltime/fate_train.pt",
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
    name = ("{activation}-{kDims}-"
            "{train_lambda}-{sigma_type}-{sigma_const}-"
            "{train_clip}-{train_lr}").format(**args.__dict__)
    name += f"-x1_v{LAM_V}_g{LAM_G}_b{LAM_B}"
    # B1 sensitivity: λ_X1 と α が default 以外なら tag に追加
    if LAMBDA_X1 != 1.0:
        name += f"_lx{LAMBDA_X1}"
    if ALPHA_VAL != 1.0:
        name += f"_a{ALPHA_VAL}"
    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}")
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")
    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt  = os.path.join(args.out_dir, "train.{}.pt")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


# ─────────────────────────────────────────────────────────────────
# X1 損失計算
# ─────────────────────────────────────────────────────────────────
def compute_x1_loss(model, centroids_t, growth_norm_t, t_val, device,
                    lam_v=1.0, lam_g=0.1, lam_b=0.01,
                    alpha=1.0, basin_sigma=0.1, n_eps=8):
    """
    centroids_t:    (T, D) — 時刻 t における各トピック centroid
    growth_norm_t:  (T,)   — 時刻 t における各トピック標準化成長率
    t_val:          float  — 時刻 (config.y[t_idx])

    Returns: (L_X1 total, dict of components)
    """
    T, D = centroids_t.shape
    # active topics (centroid が non-zero) のみ
    active = centroids_t.abs().sum(dim=-1) > 1e-6
    if active.sum() < 2:
        return torch.tensor(0.0, device=device), {"val": 0.0, "grad": 0.0, "basin": 0.0}

    c = centroids_t[active].to(device).requires_grad_(True)              # (T_act, D)
    g_norm = growth_norm_t[active].to(device)                              # (T_act,)
    t_col = torch.ones(c.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([c, t_col], dim=1)                                       # (T_act, D+1)

    # Φ(c_j, t) を計算
    phi_c = model._func._pot(xt).squeeze(-1)                                # (T_act,)

    # (1) L_anchor_val: Φ ≈ -α·g̃
    target = -alpha * g_norm
    L_val = ((phi_c - target) ** 2).mean()

    # (2) L_anchor_grad: ∇_x Φ(c_j) ≈ 0
    grad_xt = torch.autograd.grad(phi_c.sum(), xt, create_graph=True)[0]
    grad_x  = grad_xt[:, :-1]   # 時間方向は無視、空間勾配のみ
    L_grad = (grad_x ** 2).sum(dim=-1).mean()

    # (3) L_basin: 周辺 ε で Φ が下がらない (谷形状)
    # c_j_perturbed = c_j + ε,  ε ~ N(0, σ_b² I)
    # 計算量を抑えるため n_eps=8 サンプルで MC 推定
    eps = torch.randn(n_eps, c.shape[0], c.shape[1], device=device) * basin_sigma
    c_perturbed = c.unsqueeze(0) + eps                                     # (n_eps, T_act, D)
    c_pert_flat = c_perturbed.view(-1, c.shape[1])                          # (n_eps*T_act, D)
    t_pert      = torch.ones(c_pert_flat.shape[0], 1, device=device) * float(t_val)
    xt_pert     = torch.cat([c_pert_flat, t_pert], dim=1)
    phi_pert    = model._func._pot(xt_pert).squeeze(-1).view(n_eps, c.shape[0])
    # Φ(c+ε) - Φ(c) > 0 が望ましい (谷)
    diff = phi_pert - phi_c.unsqueeze(0)                                     # (n_eps, T_act)
    L_basin = torch.nn.functional.relu(-diff).mean()

    L_total = lam_v * L_val + lam_g * L_grad + lam_b * L_basin
    return L_total, {
        "val":   L_val.item(),
        "grad":  L_grad.item(),
        "basin": L_basin.item(),
    }


# ─────────────────────────────────────────────────────────────────
# 学習ループ (PI-SDE の run/run_leaveout を base に X1 を追加)
# ─────────────────────────────────────────────────────────────────
def run_x1(args, config, x, y, centroids, growth_norm, leaveouts=None):
    from src.model import ForwardSDE
    import src.train as pisde_train
    from src.train import OTLoss

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_t = config.train_t  # leaveout 反映済み
    print(f"  train_t = {train_t}")

    model = ForwardSDE(config).to(device)
    loss_fn = OTLoss(config, device)
    optimizer = optim.Adam(list(model.parameters()), lr=config.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    torch.save(config.__dict__, config.config_pt)

    import tqdm
    pbar = tqdm.tqdm(range(config.train_epochs))
    best_loss = np.inf

    log = open(config.train_log, "w")

    for epoch in pbar:
        model.zero_grad()
        losses_xy, losses_r, losses_x1 = [], [], []

        # PI-SDE 元: x_0 サンプル + SDE rollout
        dat_prev = x[config.start_t]
        x_i, _ = pisde_train.p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
        r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
        x_r_i = torch.cat([x_i, r_i], dim=1).to(device)
        ts = [0] + train_t
        y_ts = [np.float64(y[ts_i]) for ts_i in ts]
        x_r_s = model(y_ts, x_r_i)

        # 各時点で Sinkhorn + X1 を計算
        for j in train_t:
            dat_cur = x[j]
            y_j, _ = pisde_train.p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
            position = train_t.index(j)
            a_i = torch.ones(x_i.shape[0]); a_i = a_i / a_i.sum()
            b_j = torch.ones(y_j.shape[0]); b_j = b_j / b_j.sum()

            # Sinkhorn loss
            loss_xy = loss_fn(a_i, x_r_s[position + 1][:, 0:-1], b_j, y_j)
            losses_xy.append(loss_xy.item())

            # HJ residual (PI-SDE 元: 最終時点のみ)
            if (config.train_lambda > 0) & (j == train_t[-1]):
                loss_r = torch.mean(x_r_s[-1][:, -1] * config.train_lambda)
                losses_r.append(loss_r.item())
                loss_step = loss_xy + loss_r
            else:
                loss_step = loss_xy

            # X1 (Topic-Anchor) loss
            if LAMBDA_X1 > 0:
                L_x1, x1_comp = compute_x1_loss(
                    model, centroids[j], growth_norm[j], y[j], device,
                    lam_v=LAM_V, lam_g=LAM_G, lam_b=LAM_B,
                    alpha=ALPHA_VAL, basin_sigma=BASIN_SIGMA,
                )
                loss_step = loss_step + LAMBDA_X1 * L_x1
                losses_x1.append(L_x1.item())

            loss_step.backward(retain_graph=True)

        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train_clip)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

        train_loss_xy = np.mean(losses_xy)
        train_loss_r  = np.mean(losses_r) if losses_r else 0.0
        train_loss_x1 = np.mean(losses_x1) if losses_x1 else 0.0

        desc = f"[ep {epoch+1}] Sink={train_loss_xy:.3f} HJ={train_loss_r:.4f} X1={train_loss_x1:.4f}"
        pbar.set_description(desc)
        log.write(desc + "\n"); log.flush()

        if train_loss_xy < best_loss:
            best_loss = train_loss_xy
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1},
                       config.train_pt.format("best"))

        if (epoch + 1) % config.save == 0:
            ep_str = str(epoch + 1).rjust(6, "0")
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1},
                       config.train_pt.format(f"epoch_{ep_str}"))

    log.close()
    return config


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────
def main():
    args = make_args()
    config = init_config(args)

    # データロード
    data = torch.load(args.data_path, weights_only=False)
    x = data["xp"]
    y = data["y"]
    centroids   = data["centroids"]
    growth_norm = data["growth_norm"]
    config.x_dim = x[0].shape[-1]
    config.n_topics = data["n_topics"]

    print("=" * 70)
    print(f"  PI-SDE + X1 (Topic-Anchor)")
    print(f"  seed={SEED}  epochs={EPOCHS}  leaveout={LEAVEOUT_T}")
    print(f"  λ_X1={LAMBDA_X1}  (val={LAM_V}, grad={LAM_G}, basin={LAM_B})")
    print(f"  α_val={ALPHA_VAL}  σ_basin={BASIN_SIGMA}")
    print(f"  out_dir={config.out_dir}")
    print("=" * 70)

    # save 頻度を EPOCHS に合わせる
    config.save = min(EPOCHS, 500)

    # スキップ判定: 学習完了の証である最終 epoch ckpt が存在する場合のみ skip
    # (部分学習で残った best.pt は信頼できないので必ず再学習する)
    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")
    best_ckpt = config.train_pt.format("best")
    if os.path.exists(existing_ckpt):
        print(f"  ✅ 既存 checkpoint: {existing_ckpt}  → 学習スキップ")
    else:
        config = run_x1(args, config, x, y, centroids, growth_norm,
                        leaveouts=[int(t) for t in LEAVEOUT_T.split(",") if t.strip()] if LEAVEOUT_T else None)
        if not os.path.exists(existing_ckpt) and os.path.exists(best_ckpt):
            existing_ckpt = best_ckpt

    # ── 評価: Sinkhorn + Spearman + NDCG@10 ─────────────────
    print(f"\n=== Evaluation ===\nckpt: {existing_ckpt}")
    from src.model import ForwardSDE
    import src.train as pisde_train
    from geomloss import SamplesLoss
    from scipy import stats

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpt = torch.load(existing_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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

    def eval_phi_at_centroids(t_cur):
        """Φ(c_j(t), t) を取得し、g_j(t) との Spearman/NDCG を計算"""
        c_cpu = centroids[t_cur]
        active_cpu = c_cpu.abs().sum(dim=-1) > 1e-6   # CPU mask
        c_a = c_cpu[active_cpu].to(device).requires_grad_()
        g_a = data["growth"][t_cur][active_cpu].numpy()
        t_col = torch.ones(c_a.shape[0], 1, device=device) * float(y[t_cur])
        xt = torch.cat([c_a, t_col], dim=1)
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        # Spearman: 低 Φ ⇄ 高 g
        r, p_val = stats.spearmanr(phi, g_a)
        # NDCG@10
        order = np.argsort(phi)        # Φ 昇順 = 上位
        rel = np.maximum(g_a, 0.0)
        K = min(10, len(rel))
        dcg = sum(rel[order[k]] / np.log2(k + 2) for k in range(K))
        ideal = np.argsort(-rel)
        idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
        ndcg = dcg / (idcg + 1e-10) if idcg > 0 else float("nan")
        # Precision@10
        prec = (g_a[order[:K]] > 0).mean()
        return {"spearman_r": float(r), "spearman_p": float(p_val),
                "ndcg": float(ndcg), "prec_at_10": float(prec),
                "n_active": int(active_cpu.sum())}

    results = []
    print(f"\n  {'t':<4} {'split':<8} {'Sinkhorn':<12} {'Spearman':<14} {'NDCG@10':<10} {'P@10':<8}")
    print("  " + "-" * 65)
    for t in range(1, len(x)):
        _loset = set(int(s.strip()) for s in LEAVEOUT_T.split(",") if s.strip()) if LEAVEOUT_T else set()
        split = "test" if (t in _loset) else "train"
        sink = eval_sinkhorn(t)
        phi_eval = eval_phi_at_centroids(t)
        sig = "*" if phi_eval["spearman_p"] < 0.05 else " "
        print(f"  {t:<4} {split:<8} {sink:<12.4f} "
              f"{phi_eval['spearman_r']:+.4f}{sig}p={phi_eval['spearman_p']:.3f}"
              f"  {phi_eval['ndcg']:.3f}     {phi_eval['prec_at_10']:.2f}")
        results.append({"t": t, "split": split, "sinkhorn": sink, **phi_eval})

    out_eval = Path(config.out_dir) / "evaluation_x1.json"
    json.dump({"results": results, "settings": {
        "seed": SEED, "epochs": EPOCHS, "leaveout_t": LEAVEOUT_T,
        "lambda_x1": LAMBDA_X1, "lam_v": LAM_V, "lam_g": LAM_G, "lam_b": LAM_B,
    }}, out_eval.open("w"), indent=2)
    print(f"\nSaved -> {out_eval}")


if __name__ == "__main__":
    main()
