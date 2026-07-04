"""
試験スクリプト: HJ正則化 + L_trend detach修正 + 著者フローベクトル場

実データ不要。トイデータで以下を検証:
  1. L_HJ (Hamilton-Jacobi) の勾配計算が通るか
  2. detach()なし L_trend で encoder まで gradient が流れるか
  3. 著者フローベクトル場 v = -∇Φ の可視化

python -m pnode_patent_runner.test_hj_flow
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

# ─────────────────────────────────────────────────────────────────────────────
# 1. トイモデル定義
# ─────────────────────────────────────────────────────────────────────────────

class ToyEncoder(nn.Module):
    """入力特徴 → 潜在空間 z (勾配テスト用)"""
    def __init__(self, in_dim=8, latent_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
    def forward(self, x):
        return self.fc(x)


class ToyPotentialNet(nn.Module):
    """Φ(z) → スカラー  (Softplus で≥0 保証)"""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    def forward(self, z):
        return self.net(z).squeeze(-1)  # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# 2. L_HJ 損失の実装
# ─────────────────────────────────────────────────────────────────────────────

def compute_l_hj(
    pot: ToyPotentialNet,
    z_t: torch.Tensor,        # (B, D) 時刻 t での潜在点
    z_t1: torch.Tensor,       # (B, D) 時刻 t+1 での潜在点
    delta_t: float = 1.0,
) -> torch.Tensor:
    """
    Hamilton-Jacobi 正則化損失:
      L_HJ = mean( |∂_t Φ + ½ ||∇_z Φ||²|² )

    ∂_t Φ ≈ (Φ(z_{t+1}) - Φ(z_t)) / Δt  (有限差分)
    ∇_z Φ を autograd で計算
    """
    # ∂_t Φ の有限差分近似
    phi_t  = pot(z_t)           # (B,)
    phi_t1 = pot(z_t1)          # (B,)
    dphi_dt = (phi_t1 - phi_t) / delta_t   # (B,)

    # ∇_z Φ(z_t) を autograd で計算
    z_for_grad = z_t.detach().requires_grad_(True)
    phi_for_grad = pot(z_for_grad)               # (B,)
    grad_phi = torch.autograd.grad(
        phi_for_grad.sum(), z_for_grad,
        create_graph=True,
    )[0]                                          # (B, D)
    grad_norm_sq = (grad_phi ** 2).sum(dim=-1)   # (B,)

    # HJ 残差: ∂_t Φ + ½ ||∇Φ||²  (理想は 0)
    hj_residual = dphi_dt + 0.5 * grad_norm_sq   # (B,)
    l_hj = (hj_residual ** 2).mean()
    return l_hj


# ─────────────────────────────────────────────────────────────────────────────
# 3. L_trend (detach なし版) の実装
# ─────────────────────────────────────────────────────────────────────────────

def compute_l_trend_no_detach(
    z_all: torch.Tensor,       # (N_authors + N_topics, D)
    num_authors: int,
    pot: ToyPotentialNet,
    growth_rates: torch.Tensor, # (N_topics,)
) -> torch.Tensor:
    """
    L_trend (detach なし):
      Φ(z_topic_j) → -g_j_norm

    detach() を外すことで gradient が encoder まで流れる。
    """
    z_topics = z_all[num_authors:]              # slice: gradient 保持
    phi = pot(z_topics)                          # (N_topics,)
    phi_centered = phi - phi.mean()

    g_norm = (growth_rates - growth_rates.mean()) / (growth_rates.std() + 1e-8)
    return F.mse_loss(phi_centered, -g_norm)


# ─────────────────────────────────────────────────────────────────────────────
# 4. テスト実行
# ─────────────────────────────────────────────────────────────────────────────

def test_l_hj_gradient():
    print("=" * 55)
    print("TEST 1: L_HJ 勾配計算")
    print("=" * 55)

    pot = ToyPotentialNet(latent_dim=2)
    opt = torch.optim.Adam(pot.parameters(), lr=1e-3)

    B = 32
    losses = []
    for step in range(50):
        z_t  = torch.randn(B, 2)
        # z_{t+1} = z_t + drift(-∇Φ) + noise
        with torch.no_grad():
            z_t1 = z_t - 0.1 * torch.randn(B, 2) + 0.05 * torch.randn(B, 2)

        opt.zero_grad()
        loss = compute_l_hj(pot, z_t, z_t1, delta_t=1.0)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    print(f"  初期 L_HJ: {losses[0]:.6f}")
    print(f"  最終 L_HJ: {losses[-1]:.6f}")
    print(f"  {'✅ 損失が減少' if losses[-1] < losses[0] else '⚠️  損失が増加（初期値依存）'}")
    return losses


def test_l_trend_gradient_flow():
    print("\n" + "=" * 55)
    print("TEST 2: L_trend detach なし → encoder に gradient 流れるか")
    print("=" * 55)

    N_authors, N_topics = 20, 10
    encoder = ToyEncoder(in_dim=8, latent_dim=2)
    pot     = ToyPotentialNet(latent_dim=2)
    growth  = torch.randn(N_topics)       # 仮の成長率

    # encoder の 1 パラメータを取り出してチェック
    enc_param = list(encoder.parameters())[0]

    x_all = torch.randn(N_authors + N_topics, 8)

    # ── detach あり (旧実装) ──────────────────────────────────────
    z_all_detach = encoder(x_all).detach().requires_grad_(False)
    z_topics_d   = z_all_detach[N_authors:]
    phi_d = pot(z_topics_d)
    phi_centered_d = phi_d - phi_d.mean()
    g_norm = (growth - growth.mean()) / (growth.std() + 1e-8)
    loss_d = F.mse_loss(phi_centered_d, -g_norm)
    loss_d.backward()
    grad_with_detach = enc_param.grad.clone() if enc_param.grad is not None else None
    encoder.zero_grad()

    # ── detach なし (修正版) ─────────────────────────────────────
    z_all_live = encoder(x_all)
    loss_nd = compute_l_trend_no_detach(z_all_live, N_authors, pot, growth)
    loss_nd.backward()
    grad_without_detach = enc_param.grad.clone() if enc_param.grad is not None else None

    detach_grad_norm   = grad_with_detach.norm().item() if grad_with_detach is not None else 0.0
    nodetach_grad_norm = grad_without_detach.norm().item() if grad_without_detach is not None else 0.0

    print(f"  encoder grad norm (detach あり):  {detach_grad_norm:.6f}")
    print(f"  encoder grad norm (detach なし):  {nodetach_grad_norm:.6f}")
    if nodetach_grad_norm > 1e-8 and detach_grad_norm < 1e-8:
        print("  ✅ detach 除去で encoder に gradient が流れることを確認")
    elif nodetach_grad_norm > 1e-8:
        print("  ✅ gradient 流れあり（detach あり版も一部流れている）")
    else:
        print("  ❌ gradient が流れていない（実装を再確認）")


def test_flow_visualization():
    print("\n" + "=" * 55)
    print("TEST 3: 著者フローベクトル場 v = -∇Φ 可視化")
    print("=" * 55)

    # ポテンシャルを手動で双極子型に設定（谷2つ = 成長技術2つ）
    class BowlPotential(nn.Module):
        """2つの谷を持つ手作りポテンシャル（テスト用）"""
        def forward(self, z):
            # 谷: (-1, 0) と (+1, 0) → 山: (0, 0)
            d1 = ((z - torch.tensor([-1.5, 0.0])) ** 2).sum(dim=-1)
            d2 = ((z - torch.tensor([+1.5, 0.0])) ** 2).sum(dim=-1)
            phi = torch.log(torch.exp(-2 * d1) + torch.exp(-2 * d2) + 1e-6) * (-1) + 3.0
            return phi

    pot_viz = BowlPotential()

    # グリッド生成
    res = 30
    xs = torch.linspace(-3, 3, res)
    ys = torch.linspace(-2, 2, res)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1).requires_grad_(True)

    phi_grid = pot_viz(grid)
    grads = torch.autograd.grad(phi_grid.sum(), grid)[0]  # (res², 2)

    U = -grads[:, 0].detach().reshape(res, res).numpy()
    V = -grads[:, 1].detach().reshape(res, res).numpy()
    Phi = phi_grid.detach().reshape(res, res).numpy()

    # ── 描画 ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左: ポテンシャルとベクトル場
    ax = axes[0]
    cf = ax.contourf(X.numpy(), Y.numpy(), Phi, levels=20, cmap="RdYlGn_r", alpha=0.7)
    plt.colorbar(cf, ax=ax, label="Φ(z)")
    ax.quiver(X.numpy()[::3, ::3], Y.numpy()[::3, ::3],
              U[::3, ::3], V[::3, ::3],
              alpha=0.8, color="white", scale=15)

    # トピックノード（成長=谷、衰退=山）
    topics = {
        "Topic A\n(成長)":  (-1.5, 0.0),
        "Topic B\n(衰退)":  ( 0.0, 0.0),
        "Topic C\n(成長)":  ( 1.5, 0.0),
    }
    colors = ["#22c55e", "#ef4444", "#22c55e"]
    for (label, (tx, ty)), col in zip(topics.items(), colors):
        ax.scatter(tx, ty, s=120, color=col, zorder=5, edgecolors="white", lw=1.5)
        ax.annotate(label, (tx, ty), textcoords="offset points",
                    xytext=(5, 8), fontsize=8, color="white", fontweight="bold")

    # 著者ノード（ランダム配置）
    np.random.seed(42)
    authors_xy = np.random.randn(15, 2) * 0.8
    authors_xy[:, 0] *= 1.5
    ax.scatter(authors_xy[:, 0], authors_xy[:, 1], s=40, color="cyan",
               zorder=4, alpha=0.7, label="著者")
    ax.set_title("ポテンシャル場 Φ(z) と著者フロー v = -∇Φ\n"
                 "緑=成長(谷) 赤=衰退(山)  矢印=著者移動方向", fontsize=9)
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂")
    ax.legend(fontsize=8)

    # 右: 著者の実際の移動ベクトル（SDE ステップ）
    ax2 = axes[1]
    ax2.contourf(X.numpy(), Y.numpy(), Phi, levels=20, cmap="RdYlGn_r", alpha=0.5)

    z_a = torch.tensor(authors_xy, dtype=torch.float32).requires_grad_(True)
    phi_a = pot_viz(z_a)
    grad_a = torch.autograd.grad(phi_a.sum(), z_a)[0]
    drift = -0.5 * grad_a.detach().numpy()

    ax2.scatter(authors_xy[:, 0], authors_xy[:, 1], s=50, color="cyan",
                zorder=5, label="著者 t")
    ax2.quiver(authors_xy[:, 0], authors_xy[:, 1],
               drift[:, 0], drift[:, 1],
               color="yellow", scale=3, width=0.005, label="drift = -α∇Φ")

    next_pos = authors_xy + drift * 0.8
    ax2.scatter(next_pos[:, 0], next_pos[:, 1], s=30, color="orange",
                zorder=4, alpha=0.7, label="著者 t+1 (予測)")

    for i in range(len(authors_xy)):
        ax2.plot([authors_xy[i, 0], next_pos[i, 0]],
                 [authors_xy[i, 1], next_pos[i, 1]],
                 color="orange", alpha=0.4, lw=0.8)

    for (label, (tx, ty)), col in zip(topics.items(), colors):
        ax2.scatter(tx, ty, s=120, color=col, zorder=5, edgecolors="white", lw=1.5)
        ax2.annotate(label, (tx, ty), textcoords="offset points",
                     xytext=(5, 8), fontsize=8, color="black")

    ax2.set_title("著者移動 SDE ステップ: dz = -α∇Φ dt\n"
                  "シアン=現在位置  オレンジ=次ステップ予測", fontsize=9)
    ax2.set_xlabel("z₁"); ax2.legend(fontsize=8)

    out = Path("pnode_patent_runner/outputs/test_hj_flow.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("TEST: HJ 正則化 + 著者フローベクトル場の数値検証", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  ✅ 可視化保存: {out}")
    return str(out)


def test_combined_loss():
    print("\n" + "=" * 55)
    print("TEST 4: L_recon + L_trend + L_HJ 合算損失の学習ステップ")
    print("=" * 55)

    N_authors, N_topics = 30, 15
    encoder = ToyEncoder(in_dim=8, latent_dim=2)
    pot     = ToyPotentialNet(latent_dim=2)
    opt     = torch.optim.Adam(list(encoder.parameters()) + list(pot.parameters()), lr=1e-3)

    growth  = torch.randn(N_topics)
    x_all   = torch.randn(N_authors + N_topics, 8)

    history = {"recon": [], "trend": [], "hj": [], "total": []}

    for step in range(80):
        opt.zero_grad()

        z_t  = encoder(x_all)
        z_t1 = encoder(x_all + 0.1 * torch.randn_like(x_all))  # 次時刻の擬似入力

        # L_recon (簡易: reconstruction as identity)
        l_recon = F.mse_loss(z_t, torch.zeros_like(z_t))

        # L_trend (detach なし)
        l_trend = compute_l_trend_no_detach(z_t, N_authors, pot, growth)

        # L_HJ
        z_t_topics  = z_t[N_authors:]
        z_t1_topics = z_t1[N_authors:]
        l_hj = compute_l_hj(pot, z_t_topics, z_t1_topics)

        total = l_recon + 0.5 * l_trend + 0.05 * l_hj
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(pot.parameters()), 1.0
        )
        opt.step()

        for k, v in [("recon", l_recon), ("trend", l_trend),
                     ("hj", l_hj), ("total", total)]:
            history[k].append(v.item())

    print(f"  L_recon: {history['recon'][0]:.4f} → {history['recon'][-1]:.4f}")
    print(f"  L_trend: {history['trend'][0]:.4f} → {history['trend'][-1]:.4f}")
    print(f"  L_HJ:    {history['hj'][0]:.4f} → {history['hj'][-1]:.4f}")
    print(f"  L_total: {history['total'][0]:.4f} → {history['total'][-1]:.4f}")
    print(f"  {'✅ 全損失が減少傾向' if history['total'][-1] < history['total'][0] else '⚠️  損失確認要'}")

    # 学習曲線
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (key, vals) in zip(axes, history.items()):
        ax.plot(vals, lw=1.5)
        ax.set_title(f"L_{key}", fontsize=9)
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    fig.suptitle("合算損失学習曲線 (L_recon + L_trend + L_HJ)", fontsize=10)
    fig.tight_layout()
    out2 = Path("pnode_patent_runner/outputs/test_hj_losses.png")
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    print(f"  ✅ 損失曲線保存: {out2}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[P-NODE 試験スクリプト] HJ正則化 + 著者フロー")
    print("  Python:", sys.version.split()[0])
    print("  Torch:", torch.__version__)
    print("  Device: CPU (toy data)\n")

    l_hj_history = test_l_hj_gradient()
    test_l_trend_gradient_flow()
    test_flow_visualization()
    test_combined_loss()

    print("\n" + "=" * 55)
    print("全テスト完了")
    print("出力: pnode_patent_runner/outputs/test_hj_flow.png")
    print("出力: pnode_patent_runner/outputs/test_hj_losses.png")
    print("=" * 55)
