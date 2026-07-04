"""
X3 Ablation — Data leak hypothesis test.

X3 の MinimalPredictor は入力に target (g_n) を含むため、g_pred ≈ g_n の
identity 模倣でも ρ=0.99 を達成しうる。これが真の予測能力か leak かを
3 モードで切り分ける。

GINPUT_MODE:
  baseline  : g_n[t]  そのまま (= 現行 X3, leak 有)
  mask      : g_n_input を 0 で置換 (centroid + t のみから予測)
  shuffle   : g_n を topic 方向にシャッフル (入力情報量は保持 / identity 不可)
  lag       : g_n[t-1] を入力 (= 本来意図された autoregressive 予測)

eval も同じ mode で実施 (train/eval 一貫性のため)。

出力: RESULTS_X3_ABLATION/{DATA}/{MODE}/seed_{SEED}/...

Usage:
  PNODE_GINPUT_MODE=mask    PNODE_SEED=42 python pnode_patent_runner/run_pisde_x3_ablation.py
  PNODE_GINPUT_MODE=shuffle PNODE_SEED=42 python pnode_patent_runner/run_pisde_x3_ablation.py
  PNODE_GINPUT_MODE=lag     PNODE_SEED=42 python pnode_patent_runner/run_pisde_x3_ablation.py
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

SEED       = int(os.environ.get("PNODE_SEED", 42))
EPOCHS     = int(os.environ.get("PNODE_EPOCHS", 200))
LAM_G      = float(os.environ.get("PNODE_LAM_G", 0.5))
D_CTX      = int(os.environ.get("PNODE_D_CTX", 32))
MODE       = os.environ.get("PNODE_GINPUT_MODE", "baseline")
LEAVEOUT_T = os.environ.get("PNODE_LEAVEOUT_T", "")
assert MODE in {"baseline", "mask", "shuffle", "lag"}, f"unknown MODE={MODE}"

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
        out_dir=f"RESULTS_X3_ABLATION/{DATA_NAME}/{MODE}",
        data=DATA_NAME, data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        k_dims=[400, 400], activation="softplus",
        sigma_type="const", sigma_const=0.1,
        train_epochs=EPOCHS, train_lr=0.005,
        train_lambda=0.0,
        train_batch=0.1, train_clip=0.1, save=500,
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
    args.kDims = '_'.join(map(str, args.k_dims))
    name = f"x3abl_{MODE}_g{LAM_G}"
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


class MinimalPredictor(nn.Module):
    def __init__(self, x_dim, d_ctx=32):
        super().__init__()
        self.embed = nn.Linear(x_dim + 2, d_ctx)
        self.attn = nn.MultiheadAttention(d_ctx, num_heads=2, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(d_ctx)
        self.out = nn.Linear(d_ctx + x_dim + 1, 1)

    def forward(self, c, g_norm_input, t_val):
        K = c.shape[0]
        t_col = torch.full((K, 1), float(t_val), device=c.device)
        g_col = g_norm_input.unsqueeze(-1)
        feat = torch.cat([c, g_col, t_col], dim=-1)
        emb = self.embed(feat).unsqueeze(0)
        att, _ = self.attn(emb, emb, emb)
        ctx = self.norm(emb + att).squeeze(0)
        inp = torch.cat([ctx, c, t_col], dim=-1)
        return self.out(inp).squeeze(-1)


def transform_g_input(g_n_true: torch.Tensor, g_n_prev: torch.Tensor | None, mode: str,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """g_n_input を mode に従って変換"""
    if mode == "baseline":
        return g_n_true
    if mode == "mask":
        return torch.zeros_like(g_n_true)
    if mode == "shuffle":
        K = g_n_true.shape[0]
        if generator is None:
            perm = torch.randperm(K, device=g_n_true.device)
        else:
            perm = torch.randperm(K, generator=generator).to(g_n_true.device)
        return g_n_true[perm]
    if mode == "lag":
        if g_n_prev is None:
            return torch.zeros_like(g_n_true)   # t=1 では t=0 を使う (= zeros)
        # active 集合がズレ得るので、サイズが合わなければ zero pad
        if g_n_prev.shape[0] == g_n_true.shape[0]:
            return g_n_prev.to(g_n_true.device)
        # サイズが違うなら共通インデックス分だけ使う (シンプルに zero)
        return torch.zeros_like(g_n_true)
    raise ValueError(mode)


def compute_x3_loss(model, predictor, centroids_t, growth_norm_t, growth_norm_prev,
                    t_val, device, lam_growth, mode):
    active = centroids_t.abs().sum(dim=-1) > 1e-6
    if active.sum() < 2:
        zero = torch.tensor(0.0, device=device)
        return zero, {"val": 0.0, "growth": 0.0}

    c = centroids_t[active].to(device).requires_grad_(True)
    g_n = growth_norm_t[active].to(device)
    g_n_prev = None
    if growth_norm_prev is not None:
        # 同じ active mask を仮定 (PI-SDE の前提では centroid 0 のトピックは常時非アクティブ)
        if growth_norm_prev.shape[0] == active.shape[0]:
            g_n_prev = growth_norm_prev[active].to(device)
    t_col = torch.ones(c.shape[0], 1, device=device) * float(t_val)
    xt = torch.cat([c, t_col], dim=1)

    phi = model._func._pot(xt).squeeze(-1)
    L_val = ((phi + g_n) ** 2).mean()

    g_input = transform_g_input(g_n, g_n_prev, mode)
    g_pred = predictor(c, g_input, t_val)
    L_growth = ((g_pred - g_n) ** 2).mean()

    L_total = L_val + lam_growth * L_growth
    return L_total, {"val": L_val.item(), "growth": L_growth.item()}


def run_x3(args, config, x, y, centroids, growth_norm):
    from src.model import ForwardSDE
    import src.train as pisde_train
    from src.train import OTLoss

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    train_t = config.train_t
    print(f"  train_t = {train_t}")
    print(f"  λ_growth = {LAM_G}, MODE = {MODE}")

    model = ForwardSDE(config).to(device)
    predictor = MinimalPredictor(x_dim=config.x_dim, d_ctx=D_CTX).to(device)

    loss_fn = OTLoss(config, device)
    optimizer = optim.AdamW([
        {"params": model.parameters(), "weight_decay": 0.0},
        {"params": predictor.parameters(), "weight_decay": 1e-4},
    ], lr=config.train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    torch.save(config.__dict__, config.config_pt)

    import tqdm
    pbar = tqdm.tqdm(range(config.train_epochs))
    best_loss = np.inf
    log = open(config.train_log, "w")

    for epoch in pbar:
        model.zero_grad(); predictor.zero_grad()
        losses_xy, losses_x3 = [], []
        comp = {"val": 0, "growth": 0, "n": 0}

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

            g_prev = growth_norm[j - 1] if (j - 1) >= 0 else None
            L_x3, c3 = compute_x3_loss(model, predictor, centroids[j], growth_norm[j],
                                       g_prev, y[j], device,
                                       lam_growth=LAM_G, mode=MODE)
            loss_step = loss_xy + L_x3
            losses_x3.append(L_x3.item())
            for k in ("val", "growth"):
                comp[k] += c3[k]
            comp["n"] += 1
            loss_step.backward(retain_graph=True)

        if config.train_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(predictor.parameters()),
                config.train_clip,
            )
        optimizer.step()
        scheduler.step()
        model.zero_grad(); predictor.zero_grad()

        l_xy = np.mean(losses_xy)
        l_x3 = np.mean(losses_x3)
        n = max(comp["n"], 1)
        desc = f"[ep {epoch+1}] mode={MODE} Sink={l_xy:.3f} val={comp['val']/n:.4f} growth={comp['growth']/n:.4f}"
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


def main():
    args = make_args()
    config = init_config(args)

    data = torch.load(args.data_path, weights_only=False)
    x = data["xp"]; y = data["y"]
    centroids = data["centroids"]
    growth_norm = data["growth_norm"]
    growth_raw = data["growth"]
    config.x_dim = x[0].shape[-1]
    config.n_topics = data["n_topics"]

    print("=" * 78)
    print(f"  X3 ABLATION  (mode={MODE})")
    print(f"  seed={SEED}  epochs={EPOCHS}  λ_growth={LAM_G}")
    print(f"  out_dir={config.out_dir}")
    print("=" * 78)

    config.save = min(EPOCHS, 500)
    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")
    if os.path.exists(existing_ckpt):
        print(f"  ✅ existing ckpt found, skip training")
    else:
        run_x3(args, config, x, y, centroids, growth_norm)
        if not os.path.exists(existing_ckpt):
            existing_ckpt = config.train_pt.format("best")

    # ── Eval ──
    print(f"\n=== Evaluation (mode={MODE}) ===\n  ckpt: {existing_ckpt}")
    from src.model import ForwardSDE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    predictor = MinimalPredictor(x_dim=config.x_dim, d_ctx=D_CTX).to(device)
    ckpt = torch.load(existing_ckpt, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    model.eval(); predictor.eval()

    def eval_t(t_cur):
        c_cpu = centroids[t_cur]
        active = c_cpu.abs().sum(dim=-1) > 1e-6
        c_a = c_cpu[active].to(device).requires_grad_()
        g_a = growth_raw[t_cur][active].numpy()
        g_n = growth_norm[t_cur][active].to(device)
        # 前時点
        g_n_prev = None
        if (t_cur - 1) >= 0:
            gp = growth_norm[t_cur - 1]
            if gp.shape[0] == active.shape[0]:
                g_n_prev = gp[active].to(device)
        g_input = transform_g_input(g_n, g_n_prev, MODE)
        t_col = torch.ones(c_a.shape[0], 1, device=device) * float(y[t_cur])
        xt = torch.cat([c_a, t_col], dim=1)
        phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
        g_pred = predictor(c_a, g_input, float(y[t_cur])).detach().cpu().numpy()

        r_phi, p_phi = stats.spearmanr(phi, g_a)
        r_g,   p_g   = stats.spearmanr(g_pred, g_a)
        K = min(10, len(g_a))
        rel = np.maximum(g_a, 0.0)
        ideal = np.argsort(-rel)
        idcg = sum(rel[ideal[k]] / np.log2(k + 2) for k in range(K))
        order_phi = np.argsort(phi)
        order_g = np.argsort(-g_pred)
        ndcg_phi = sum(rel[order_phi[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
        ndcg_g = sum(rel[order_g[k]] / np.log2(k + 2) for k in range(K)) / (idcg + 1e-10)
        prec_phi = (g_a[order_phi[:K]] > 0).mean()
        prec_g = (g_a[order_g[:K]] > 0).mean()
        g_n_np = g_n.cpu().numpy()
        return {
            "phi_spearman_r": float(r_phi), "phi_spearman_p": float(p_phi),
            "phi_ndcg": float(ndcg_phi), "phi_prec_at_10": float(prec_phi),
            "growth_spearman_r": float(r_g), "growth_spearman_p": float(p_g),
            "growth_ndcg": float(ndcg_g), "growth_prec_at_10": float(prec_g),
            "growth_mse": float(((g_pred - g_n_np) ** 2).mean()),
            "growth_mae": float(np.abs(g_pred - g_n_np).mean()),
            "n_active": int(active.sum()),
        }

    results = []
    print(f"\n  {'t':<4} {'split':<6} {'φ-Sp':<10} {'φ-P@10':<8} {'g-Sp':<10} {'g-P@10':<8} {'g-MSE':<10}")
    print("  " + "-" * 65)
    _loset = set(int(s.strip()) for s in LEAVEOUT_T.split(",") if s.strip()) if LEAVEOUT_T else set()
    for t in range(1, len(x)):
        split = "test" if (t in _loset) else "train"
        r = eval_t(t)
        sig_p = "*" if r["phi_spearman_p"] < 0.05 else " "
        sig_g = "*" if r["growth_spearman_p"] < 0.05 else " "
        print(f"  {t:<4} {split:<6} "
              f"{r['phi_spearman_r']:+.3f}{sig_p:<3} {r['phi_prec_at_10']:<8.2f} "
              f"{r['growth_spearman_r']:+.3f}{sig_g:<3} {r['growth_prec_at_10']:<8.2f} "
              f"{r['growth_mse']:<10.4f}")
        results.append({"t": t, "split": split, **r})

    out_eval = Path(config.out_dir) / f"evaluation_x3_ablation_{MODE}.json"
    json.dump({"results": results, "settings": {
        "seed": SEED, "epochs": EPOCHS, "lam_growth": LAM_G,
        "mode": MODE, "leaveout_t": LEAVEOUT_T,
        "model": f"X3 ablation {MODE}",
    }}, out_eval.open("w"), indent=2)
    print(f"\nSaved -> {out_eval}")


if __name__ == "__main__":
    main()
