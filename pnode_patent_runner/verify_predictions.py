"""
具体的検証: leaveout モデルで予測した TOP-K vs 実際の TOP-K の照合.
"""
from __future__ import annotations

import os, sys, glob
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE
from types import SimpleNamespace

DOMAINS = {
    "Paper (arXiv CS)": {
        "data": "data/PNode_Paper_X1/alltime/fate_train.pt",
        "root": "RESULTS/PNode_Paper_X1",
        "last_t": 3, "year": "2025",
        "leaveout_tag": "leaveout3",
    },
    "Patent Energy (CPC Y02)": {
        "data": "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "root": "RESULTS/PNode_Patent_Energy_X1_top50",
        "last_t": 11, "year": "2024",
        "leaveout_tag": "leaveout11",
    },
    "JP Construction (J-STAGE)": {
        "data": "data/PNode_JP_Construction_X1/alltime/fate_train.pt",
        "root": "RESULTS/PNode_JP_Construction_X1",
        "last_t": 10, "year": "2025",
        "leaveout_tag": "leaveout10",
    },
}
TAG_SUFFIX = "-x1_v1.0_g0.1_b0.01"
SEEDS = [0, 1, 42, 123, 999]
TOP_K = 10


def main():
    for dname, cfg in DOMAINS.items():
        print(f"\n{'='*100}")
        print(f"  {dname}  — leaveout {cfg['year']} (モデル未学習の真の未来)")
        print(f"{'='*100}")

        data = torch.load(cfg["data"], weights_only=False)
        topic_names = data["topic_names"]
        n_topics = data["n_topics"]
        last_t = cfg["last_t"]
        centroids = data["centroids"]
        growth = data["growth"]

        cent = centroids[last_t].numpy()
        active = cent.sum(axis=-1) != 0
        cent_act = cent[active]
        names_act = [topic_names[i] for i in range(n_topics) if active[i]]
        g_act = growth[last_t].numpy()[active]

        # Actual TOP-3 ground truth
        order_actual = np.argsort(-g_act)
        print(f"\n  ▼ 実際の TOP-3 成長 topic ({cfg['year']} に最も伸びた)")
        for k, i in enumerate(order_actual[:3]):
            print(f"    [{k+1}] {names_act[i]:<35}  実 g = {g_act[i]:+.3f}")

        # Aggregate predictions across 5 leaveout seeds
        phi_per_seed = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for s in SEEDS:
            pat = f"{cfg['root']}/*{TAG_SUFFIX}/seed_{s}/{cfg['leaveout_tag']}"
            cands = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
            if not cands: continue
            out_dir = cands[0]
            config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
            config.x_dim = data["xp"][0].shape[-1]
            model = ForwardSDE(config).to(device)
            ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
            ckpt = ckpts[-1] if ckpts else str(out_dir / "train.best.pt")
            try:
                state = torch.load(ckpt, weights_only=False, map_location=device)
                model.load_state_dict(state["model_state_dict"])
            except Exception as e:
                continue
            model.eval()
            cent_dev = torch.tensor(cent_act, dtype=torch.float32, device=device)
            y_val = data["y"][last_t]
            t_col = torch.full((cent_dev.shape[0], 1), float(y_val), device=device)
            with torch.enable_grad():
                xt = torch.cat([cent_dev, t_col], dim=1).requires_grad_()
                phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
            phi_per_seed.append(phi)

        if not phi_per_seed:
            print("    (leaveout モデルが見つからない)")
            continue
        phi_mean = np.mean(phi_per_seed, axis=0)
        order_pred = np.argsort(phi_mean)   # 低 Φ = 成長予測

        print(f"\n  ▼ X1 PI-SDE 予測 TOP-3 成長 topic  ({len(phi_per_seed)} seed mean)")
        for k, i in enumerate(order_pred[:3]):
            actual_rank = int(np.where(order_actual == i)[0][0]) + 1
            actual_g = g_act[i]
            mark = "✓" if actual_rank <= 3 else ("~" if actual_rank <= 10 else "✗")
            print(f"    [{k+1}] {names_act[i]:<35}  Φ_avg = {phi_mean[i]:+.3f}  "
                  f"(実 rank {actual_rank:>2}/{len(g_act)}, 実 g={actual_g:+.3f}) {mark}")

        # TOP-K overlap
        K = min(TOP_K, len(g_act))
        pred_set = set(order_pred[:K].tolist())
        actual_set = set(order_actual[:K].tolist())
        overlap = pred_set & actual_set
        print(f"\n  ◆ Precision@{K}: 予測 TOP-{K} のうち実 TOP-{K} に含まれる数")
        print(f"    {len(overlap)}/{K} = {len(overlap)/K*100:.0f}% hit")

        # TOP-3 specific
        K3 = 3
        pred3 = set(order_pred[:K3].tolist())
        actual3 = set(order_actual[:K3].tolist())
        overlap3 = pred3 & actual3
        print(f"  ◆ Precision@3: {len(overlap3)}/3 = {len(overlap3)/3*100:.0f}% hit")

        # Spearman
        from scipy import stats
        rho, p = stats.spearmanr(phi_mean, g_act)
        print(f"  ◆ Spearman ρ(Φ, g) = {rho:+.4f}  p = {p:.4g}")
        print(f"    解釈: 強い負相関 = 順位予測良好 (|ρ| 大きいほど良い)")


if __name__ == "__main__":
    main()
