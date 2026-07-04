"""
B3 Case Study: X1 PI-SDE TOP-K predicted growing topics

3 ドメイン × seed=42 (代表) で
  - 学習済み Φ_θ(c_j, t_last) を計算
  - 値が低い topic = 成長期待大
  - 実成長率 g_j(t_last) と並べて表示
  - TOP10 / BOTTOM5 を人手検証用に出力
"""
from __future__ import annotations

import os, sys, glob, json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/tmp/PI-SDE")
os.chdir("/home/nakamuraroi/kumagai")

from src.model import ForwardSDE
from types import SimpleNamespace


DOMAINS = {
    "Paper (arXiv CS)": {
        "data":  "data/PNode_Paper_X1/alltime/fate_train.pt",
        "root":  "RESULTS/PNode_Paper_X1",
        "tag_suffix":  "-x1_v1.0_g0.1_b0.01",
        "last_t": 3,
        "year_label": "2025",
    },
    "Patent Energy (CPC)": {
        "data":  "data/PNode_Patent_Energy_X1_top50/alltime/fate_train.pt",
        "root":  "RESULTS/PNode_Patent_Energy_X1_top50",
        "tag_suffix":  "-x1_v1.0_g0.1_b0.01",
        "last_t": 11,
        "year_label": "2024",
    },
    "arXiv Construction": {
        "data":  "data/PNode_ArXiv_Construction_X1_v2/alltime/fate_train.pt",
        "root":  "RESULTS/PNode_ArXiv_Construction_X1_v2",
        "tag_suffix":  "-x1_v1.0_g0.1_b0.01",
        "last_t": 10,
        "year_label": "2025",
    },
    "JP Construction (J-STAGE)": {
        "data":  "data/PNode_JP_Construction_X1/alltime/fate_train.pt",
        "root":  "RESULTS/PNode_JP_Construction_X1",
        "tag_suffix":  "-x1_v1.0_g0.1_b0.01",
        "last_t": 10,
        "year_label": "2025",
    },
}

SEED = 42
TOP_K = 10
BOT_K = 5


def load_model(root, tag_suffix, seed):
    """Find the X1 default-setting (D) checkpoint."""
    pat = f"{root}/*{tag_suffix}/seed_{seed}/alltime"
    candidates = [Path(p) for p in glob.glob(pat) if "_lx" not in Path(p).parent.name]
    if not candidates:
        return None, None
    out_dir = candidates[0]

    config = SimpleNamespace(**torch.load(out_dir / "config.pt", weights_only=False))
    ckpt_p = out_dir / "train.best.pt"
    if not ckpt_p.exists():
        ckpts = sorted(glob.glob(str(out_dir / "train.epoch_*.pt")))
        ckpt_p = Path(ckpts[-1]) if ckpts else None
        if ckpt_p is None:
            return None, None
    return config, ckpt_p


def evaluate_topics(config, ckpt_p, data_path, last_t):
    data = torch.load(data_path, weights_only=False)
    centroids   = data["centroids"]
    growth_raw  = data["growth"]
    topic_names = data["topic_names"]
    y_vals      = data["y"]

    config.x_dim = data["xp"][0].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardSDE(config).to(device)
    ckpt = torch.load(ckpt_p, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    cent_t = centroids[last_t].numpy()
    active = cent_t.sum(axis=-1) != 0
    cent_active = cent_t[active]
    g_active    = growth_raw[last_t].numpy()[active]
    names_active = [topic_names[i] for i in range(len(active)) if active[i]]

    xt = torch.cat([
        torch.tensor(cent_active, dtype=torch.float32),
        torch.full((len(cent_active), 1), float(y_vals[last_t]))
    ], dim=1).to(device).requires_grad_()
    phi = model._func._pot(xt).squeeze(-1).detach().cpu().numpy()
    return phi, g_active, names_active


def main():
    overall = {}
    for dname, cfg in DOMAINS.items():
        print(f"\n{'='*100}")
        print(f"  {dname}  —  seed={SEED}, t_last={cfg['last_t']} ({cfg['year_label']})")
        print(f"{'='*100}")

        config, ckpt = load_model(cfg["root"], cfg["tag_suffix"], SEED)
        if config is None:
            print(f"  ⚠ ckpt not found for {dname}")
            continue

        phi, g, names = evaluate_topics(config, ckpt, cfg["data"], cfg["last_t"])

        # Sort: low Φ ⇒ predicted growing (rank 1)
        order = np.argsort(phi)
        r_spearman, p_val = stats.spearmanr(phi, g)

        print(f"  Spearman ρ(Φ, g) = {r_spearman:+.4f}  (p = {p_val:.4f})")
        print(f"  active topics = {len(names)}")

        # TOP-K predicted growing
        print(f"\n  ▼ TOP-{TOP_K} predicted GROWING (lowest Φ):")
        print(f"  {'#':<3} {'Topic':<32} {'Φ':>10} {'g (actual)':>12} {'g-rank':>8}")
        print("  " + "-" * 76)
        g_ranks = stats.rankdata(-g, method="min")   # 1 = highest g
        for i, idx in enumerate(order[:TOP_K]):
            mark = "✓" if g_ranks[idx] <= TOP_K else ("~" if g_ranks[idx] <= 15 else "✗")
            print(f"  {i+1:<3} {names[idx]:<32} {phi[idx]:>10.3f} {g[idx]:>+12.3f} {int(g_ranks[idx]):>5}/{len(g):<3} {mark}")

        # BOTTOM (predicted declining)
        print(f"\n  ▽ BOTTOM-{BOT_K} predicted DECLINING (highest Φ):")
        for i, idx in enumerate(order[-BOT_K:][::-1]):
            mark = "✓" if g_ranks[idx] >= len(g) - BOT_K + 1 else "✗"
            print(f"  {i+1:<3} {names[idx]:<32} {phi[idx]:>10.3f} {g[idx]:>+12.3f} {int(g_ranks[idx]):>5}/{len(g):<3} {mark}")

        # Hit metrics
        top_k_pred = set(order[:TOP_K].tolist())
        top_k_actual = set(np.argsort(-g)[:TOP_K].tolist())
        precision_k = len(top_k_pred & top_k_actual) / TOP_K
        print(f"\n  ▶ Precision@{TOP_K} (Φ TOP{TOP_K} ∩ g TOP{TOP_K}) / {TOP_K} = {precision_k:.2f}")

        overall[dname] = {
            "spearman": float(r_spearman),
            "spearman_p": float(p_val),
            "precision_at_10": precision_k,
            "n_topics": len(names),
        }

    # Save summary
    out_p = Path("RESULTS/b3_case_study_summary.json")
    out_p.write_text(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_p}")
    print("\n判定マーク: ✓ TOP10 hit  ~ TOP15 hit  ✗ miss")


if __name__ == "__main__":
    main()
