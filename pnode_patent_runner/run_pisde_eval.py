"""
PI-SDE 学習済みモデルに対し、leaveout 評価を行う。

評価:
  - 各時点 t について、t=0 から SDE 積分して x_r_s[t] を生成
  - 観測 x[t] との Wasserstein (Sinkhorn) 距離を計算
  - leaveout は train_t に入ってない t について計算 (held-out 性能)

ベースライン:
  - "Naive": x_pred = x[0] (時間不変仮定)
  - "Last seen": x_pred = x[t-1] (直前年が同じと仮定)

論文準拠の評価表を作る:
  | t | observed | PI-SDE (Sinkhorn) | Naive (t=0) | Last seen (t-1) |
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

# PI-SDE source
sys.path.insert(0, "/tmp/PI-SDE")

# ── 設定 ─────────────────────────────────────────────────────────
DOMAIN = os.environ.get("PNODE_DOMAIN", "paper")
SEED   = int(os.environ.get("PNODE_SEED", 42))
EPOCHS = int(os.environ.get("PNODE_EPOCHS", 500))
LEAVEOUT_T = os.environ.get("PNODE_LEAVEOUT_T", "")    # "" or "3"

if DOMAIN == "paper":
    DATA_NAME = "PNode_Paper"
    DATA_DIR  = "data/PNode_Paper"
    TRAIN_T   = [1, 2, 3]
elif DOMAIN == "patent_energy":
    DATA_NAME = "PNode_Patent_Energy"
    DATA_DIR  = "data/PNode_Patent_Energy"
    TRAIN_T   = list(range(1, 12))
else:
    raise ValueError(f"unknown DOMAIN={DOMAIN}")

os.chdir("/home/nakamuraroi/kumagai")


def make_args(epochs, leaveout_t=""):
    import argparse
    if leaveout_t:
        t = int(leaveout_t)
        eff_train_t = [tt for tt in TRAIN_T if tt != t]
        leaveout_tag = f"leaveout{t}"
    else:
        eff_train_t = TRAIN_T
        leaveout_tag = "alltime"

    ns = argparse.Namespace(
        seed=SEED,
        use_cuda=True, device=0,
        out_dir=f"RESULTS/{DATA_NAME}",
        data=DATA_NAME,
        data_path=f"{DATA_DIR}/alltime/fate_train.pt",
        data_dir=DATA_DIR,
        k_dims=[400, 400], activation="softplus",
        sigma_type="const", sigma_const=0.1,
        train_epochs=epochs, train_lr=0.005,
        train_lambda=0.5, train_batch=0.1, train_clip=0.1,
        save=500,
        evaluate_n=10000, evaluate_data=None, evaluate_baseline=False,
        task="leaveout" if leaveout_t else "fate",
        train=False, evaluate=None, config=None,
        sinkhorn_scaling=0.7, sinkhorn_blur=0.1, ns=2000,
        start_t=0, train_t=eff_train_t,
        leaveout_t=leaveout_tag if leaveout_t else "",
        test_t=[int(leaveout_t)] if leaveout_t else [],
    )
    ns.layers = len(ns.k_dims)
    return ns


def init_config(args):
    args.layers = len(args.k_dims)
    args.kDims = '_'.join(map(str, args.k_dims))
    name = ("{activation}-{kDims}-"
            "{train_lambda}-{sigma_type}-{sigma_const}-"
            "{train_clip}-{train_lr}").format(**args.__dict__)
    args.out_dir = os.path.join(args.out_dir, name, f"seed_{args.seed}")
    if args.task == "leaveout":
        args.out_dir = os.path.join(args.out_dir, args.leaveout_t)
    else:
        args.out_dir = os.path.join(args.out_dir, "alltime")
    os.makedirs(args.out_dir, exist_ok=True)
    args.train_pt  = os.path.join(args.out_dir, "train.{}.pt")
    args.done_log  = os.path.join(args.out_dir, "done.log")
    args.config_pt = os.path.join(args.out_dir, "config.pt")
    args.train_log = os.path.join(args.out_dir, "train.log")
    return args


def main():
    print(f"\n{'='*60}\n  PI-SDE on {DATA_NAME}  seed={SEED}  epochs={EPOCHS}")
    if LEAVEOUT_T:
        print(f"  leaveout t={LEAVEOUT_T}")
    print('='*60)

    import src.train as pisde_train
    from src.model import ForwardSDE

    args = make_args(EPOCHS, LEAVEOUT_T)
    config = init_config(args)
    epoch_pad = str(EPOCHS).rjust(6, "0")
    existing_ckpt = config.train_pt.format(f"epoch_{epoch_pad}")

    # 学習スキップ判定
    if os.path.exists(existing_ckpt):
        print(f"  ✅ checkpoint 既存: {existing_ckpt}  → 学習スキップ")
        # config.pt も保存する必要がある
        if not os.path.exists(config.config_pt):
            torch.save(config.__dict__, config.config_pt)
    else:
        if LEAVEOUT_T:
            config = pisde_train.run_leaveout(args, init_config, leaveouts=[int(LEAVEOUT_T)])
        else:
            config = pisde_train.run(args, init_config)

    # ── 評価: Sinkhorn / Wasserstein 距離 ─────────────────────────
    print(f"\n=== Evaluation ===\nLoading from {config.out_dir}")

    import glob
    train_pts = sorted(glob.glob(config.train_pt.format("epoch_*")))
    if not train_pts:
        # fallback to best
        bestp = config.train_pt.format("best")
        if os.path.exists(bestp):
            train_pts = [bestp]
    if not train_pts:
        print("No checkpoint found!")
        return
    use_ckpt = train_pts[-1]
    print(f"Using checkpoint: {use_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # データ先 → x_dim を設定
    from src.config_Veres import load_data
    x, y, config = load_data(config)

    model = ForwardSDE(config).to(device)
    ckpt = torch.load(use_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Sinkhorn 評価関数
    from geomloss import SamplesLoss
    ot_solver = SamplesLoss("sinkhorn", p=2, blur=config.sinkhorn_blur,
                            scaling=config.sinkhorn_scaling)

    def evaluate_t(t_cur):
        # x[0] からスタートして t_cur まで SDE 積分
        # PI-SDE の drift 計算には autograd が必要なので no_grad は使えない
        n_eval = min(2000, x[0].shape[0])
        x_0, _ = pisde_train.p_samp(x[0], n_eval)
        r_0 = torch.zeros(n_eval).unsqueeze(1)
        x_r_0 = torch.cat([x_0, r_0], dim=1).to(device).requires_grad_()

        x_r_s = model([np.float64(y[0])] + [np.float64(y[t_cur])], x_r_0)
        x_pred = x_r_s[-1][:, :-1].detach()
        x_obs  = x[t_cur].to(device)
        n_obs = min(2000, x_obs.shape[0])
        idx = np.random.choice(x_obs.shape[0], n_obs, replace=False)
        x_obs = x_obs[idx].detach()
        d = ot_solver(x_pred.contiguous(), x_obs.contiguous()).item()
        return d

    # ベースライン
    def baseline_naive(t_cur):
        """ĝ_t = x_0 (時間不変)"""
        n = min(2000, x[t_cur].shape[0])
        x_obs_idx = np.random.choice(x[t_cur].shape[0], n, replace=False)
        x_obs = x[t_cur][x_obs_idx].to(device)
        x_pred_idx = np.random.choice(x[0].shape[0], n, replace=False)
        x_pred = x[0][x_pred_idx].to(device)
        with torch.no_grad():
            return ot_solver(x_pred.contiguous(), x_obs.contiguous()).item()

    def baseline_lastseen(t_cur):
        """ĝ_t = x_{t-1}"""
        if t_cur == 0:
            return float("nan")
        prev_t = t_cur - 1
        n = min(2000, x[t_cur].shape[0], x[prev_t].shape[0])
        idx_obs = np.random.choice(x[t_cur].shape[0], n, replace=False)
        idx_prev = np.random.choice(x[prev_t].shape[0], n, replace=False)
        x_obs = x[t_cur][idx_obs].to(device)
        x_pred = x[prev_t][idx_prev].to(device)
        with torch.no_grad():
            return ot_solver(x_pred.contiguous(), x_obs.contiguous()).item()

    # 各時点で評価
    results = []
    all_t = list(range(1, len(x)))
    print(f"\n  {'t':<4} {'y':<8} {'split':<10} {'PI-SDE':<14} {'Naive(x0)':<14} {'Last(x_t-1)':<14}")
    print("  " + "-" * 70)
    for t in all_t:
        split = "test" if (LEAVEOUT_T and t == int(LEAVEOUT_T)) else "train"
        d_pisde = evaluate_t(t)
        d_naive = baseline_naive(t)
        d_last  = baseline_lastseen(t)
        results.append({
            "t": t, "y": y[t], "split": split,
            "pi_sde": d_pisde, "naive": d_naive, "last_seen": d_last,
        })
        print(f"  {t:<4} {y[t]:<8} {split:<10} {d_pisde:<14.4f} {d_naive:<14.4f} {d_last:<14.4f}")

    out = {"results": results, "settings": {
        "domain": DOMAIN, "seed": SEED, "epochs": EPOCHS,
        "leaveout_t": LEAVEOUT_T, "ckpt": use_ckpt,
    }}
    out_file = Path(config.out_dir) / "evaluation.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
