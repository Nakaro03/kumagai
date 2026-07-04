"""
P-NODE 一点集中シミュレーション
dz/dτ = -α∇Φ(z) が多ステップで分布を崩壊させる様子を数式で確認する。

出力:
  simulate_collapse.png  — 4パネル可視化
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

# ── 設定 ──────────────────────────────────────────────────────────────────────
N      = 300   # 発明者数
D      = 2     # latent 次元（可視化のため 2D）
ALPHA  = 0.8   # ODE スケール α
DT     = 1.0   # Δτ（1 年ステップ）
N_STEPS = 5    # rollout ステップ数
SEED   = 42

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ── ① 解析的ポテンシャル Φ(z) = ½||z||²（調和ポテンシャル）─────────────────

def phi_harmonic(z: np.ndarray) -> np.ndarray:
    return 0.5 * (z ** 2).sum(axis=-1)

def grad_phi_harmonic(z: np.ndarray) -> np.ndarray:
    return z  # ∇Φ = z

def ode_step_harmonic(z: np.ndarray, alpha: float, dt: float) -> np.ndarray:
    # Euler: z_{t+1} = z_t - α·∇Φ·dt = z_t - α·z_t·dt = z_t·(1 - α·dt)
    return z * (1.0 - alpha * dt)

def jacobian_det_harmonic(alpha: float, dt: float, d: int) -> float:
    # J = (1-α·dt)·I → det = (1-α·dt)^d
    return (1.0 - alpha * dt) ** d


# ── ② 学習済みポテンシャル Φ_θ(z)（SpectralNorm + Softplus MLP）─────────────

class PotentialNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(2, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1)),
            nn.Softplus(),
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

def grad_phi_mlp(model: PotentialNet2D, z: np.ndarray) -> np.ndarray:
    zt = torch.tensor(z, dtype=torch.float32, requires_grad=True)
    phi = model(zt).sum()
    phi.backward()
    return zt.grad.detach().numpy()

def ode_step_mlp(model: PotentialNet2D, z: np.ndarray, alpha: float, dt: float) -> np.ndarray:
    grad = grad_phi_mlp(model, z)
    return z - alpha * grad * dt

def jacobian_det_mlp(model: PotentialNet2D, z: np.ndarray, alpha: float, dt: float) -> float:
    """平均ヤコビアン行列式 (有限差分近似)"""
    eps = 1e-3
    dets = []
    for zi in z[:50]:  # サブサンプルで計算
        zi_t = torch.tensor(zi, dtype=torch.float32, requires_grad=False)
        J = np.zeros((D, D))
        for j in range(D):
            z_plus  = zi.copy(); z_plus[j]  += eps
            z_minus = zi.copy(); z_minus[j] -= eps
            f_plus  = zi.copy() - alpha * dt * grad_phi_mlp(model, z_plus[None])[0]
            f_minus = zi.copy() - alpha * dt * grad_phi_mlp(model, z_minus[None])[0]
            J[:, j] = (f_plus - f_minus) / (2 * eps)
        dets.append(abs(np.linalg.det(J)))
    return float(np.mean(dets))


# ── 初期分布：3 クラスタ Gaussian（3 技術パラダイム）─────────────────────────

def make_initial_z() -> np.ndarray:
    centers = np.array([[2.0, 1.0], [-2.0, 1.0], [0.0, -2.0]])
    z_list = []
    for c in centers:
        z_list.append(rng.normal(loc=c, scale=0.5, size=(N // 3, D)))
    return np.vstack(z_list).astype(np.float32)


# ── 連続方程式残差の計算 ─────────────────────────────────────────────────────

def continuity_residual(z_t: np.ndarray, z_t1: np.ndarray,
                        grad_fn, alpha: float, dt: float) -> float:
    """
    ∂ρ/∂t + ∇·(ρv) を KDE + 有限差分で近似

    手順:
      1. KDE で ρ_t, ρ_{t+1} を推定
      2. ∂ρ/∂t ≈ (ρ_{t+1} - ρ_t) / dt
      3. ∇·(ρ·v) を数値微分
      4. 残差ノルム
    """
    from scipy.stats import gaussian_kde
    bw = 0.4

    # グリッド上で評価
    xs = np.linspace(-5, 5, 30)
    ys = np.linspace(-5, 5, 30)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    kde_t  = gaussian_kde(z_t.T,  bw_method=bw)
    kde_t1 = gaussian_kde(z_t1.T, bw_method=bw)
    rho_t  = kde_t(pts.T).reshape(30, 30)
    rho_t1 = kde_t1(pts.T).reshape(30, 30)

    drho_dt = (rho_t1 - rho_t) / dt

    # ∇·(ρ·v) の数値近似
    v = -alpha * grad_fn(pts)           # velocity field on grid
    rho_v_x = rho_t * v[:, 0].reshape(30, 30)
    rho_v_y = rho_t * v[:, 1].reshape(30, 30)

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    div_rho_v = (np.gradient(rho_v_x, dx, axis=1) +
                 np.gradient(rho_v_y, dy, axis=0))

    residual = drho_dt + div_rho_v
    return float(np.mean(residual ** 2))


# ── メインシミュレーション ────────────────────────────────────────────────────

def run_simulation():
    z0 = make_initial_z()

    # 学習済み MLP Φ（小規模で初期化）
    mlp = PotentialNet2D()
    # 多少「山あり谷あり」に見せるためランダム初期化のまま使用

    # ─ rollout ─
    traj_harm = [z0.copy()]
    traj_mlp  = [z0.copy()]
    dets_harm, dets_mlp = [1.0], []
    cont_harm, cont_mlp = [], []

    for step in range(1, N_STEPS + 1):
        z_harm_prev = traj_harm[-1]
        z_mlp_prev  = traj_mlp[-1]

        z_harm_next = ode_step_harmonic(z_harm_prev, ALPHA, DT)
        z_mlp_next  = ode_step_mlp(mlp, z_mlp_prev, ALPHA, DT)

        traj_harm.append(z_harm_next)
        traj_mlp.append(z_mlp_next)

        dets_harm.append(jacobian_det_harmonic(ALPHA, DT, D) ** step)
        dets_mlp.append(jacobian_det_mlp(mlp, z_mlp_prev, ALPHA, DT))

        cont_harm.append(continuity_residual(
            z_harm_prev, z_harm_next, grad_phi_harmonic, ALPHA, DT))
        cont_mlp.append(continuity_residual(
            z_mlp_prev, z_mlp_next,
            lambda pts: grad_phi_mlp(mlp, pts), ALPHA, DT))

    # ─ 平均ペアワイズ距離 ─
    def mean_pairwise_dist(z):
        idx = rng.choice(len(z), size=100, replace=False)
        sub = z[idx]
        diffs = sub[:, None] - sub[None, :]
        return float(np.sqrt((diffs ** 2).sum(axis=-1)).mean())

    pw_harm = [mean_pairwise_dist(z) for z in traj_harm]
    pw_mlp  = [mean_pairwise_dist(z) for z in traj_mlp]

    # ── プロット ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "P-NODE 一点集中シミュレーション: dz/dτ = -α∇Φ(z)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, N_STEPS + 1, figure=fig,
                           hspace=0.45, wspace=0.3)

    steps_to_plot = list(range(0, N_STEPS + 1))
    colors_harm = plt.cm.Blues(np.linspace(0.3, 0.9, N_STEPS + 1))
    colors_mlp  = plt.cm.Reds (np.linspace(0.3, 0.9, N_STEPS + 1))

    # 行 0: 調和ポテンシャル 散布図
    for col, step in enumerate(steps_to_plot):
        ax = fig.add_subplot(gs[0, col])
        z = traj_harm[step]
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5,
                   color=colors_harm[col])
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_title(f"k={step}", fontsize=8)
        if col == 0:
            ax.set_ylabel("Harmonic Φ=½‖z‖²", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 行 1: 学習済み MLP 散布図
    for col, step in enumerate(steps_to_plot):
        ax = fig.add_subplot(gs[1, col])
        z = traj_mlp[step]
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5,
                   color=colors_mlp[col])
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        if col == 0:
            ax.set_ylabel("MLP Φ (SpectralNorm)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 行 2 左: ヤコビアン行列式 det(J^k) と ペアワイズ距離
    ax_det = fig.add_subplot(gs[2, :3])
    ks = list(range(N_STEPS + 1))
    ax_det.plot(ks, dets_harm, "b-o", label="det(J^k) Harmonic (解析)", linewidth=2)
    ax_det.plot(ks[1:], dets_mlp, "r-^", label="det(J) MLP (数値平均/ステップ)", linewidth=2)
    ax_det.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_det.set_xlabel("rollout step k")
    ax_det.set_ylabel("Jacobian det  (体積要素)")
    ax_det.set_title("体積要素の収縮: det(J)^k → 0", fontsize=9)
    ax_det.legend(fontsize=7)
    ax_det.grid(True, linestyle=":", alpha=0.5)

    # 行 2 中: ペアワイズ距離
    ax_pw = fig.add_subplot(gs[2, 3:5])
    ax_pw.plot(ks, pw_harm, "b-o", label="Harmonic", linewidth=2)
    ax_pw.plot(ks, pw_mlp,  "r-^", label="MLP",      linewidth=2)
    ax_pw.set_xlabel("rollout step k")
    ax_pw.set_ylabel("平均ペアワイズ距離")
    ax_pw.set_title("発明者間距離の崩壊", fontsize=9)
    ax_pw.legend(fontsize=7)
    ax_pw.grid(True, linestyle=":", alpha=0.5)

    # 行 2 右: 連続方程式残差
    ax_cont = fig.add_subplot(gs[2, 5:])
    ax_cont.plot(range(1, N_STEPS + 1), cont_harm, "b-o",
                 label="Harmonic", linewidth=2)
    ax_cont.plot(range(1, N_STEPS + 1), cont_mlp,  "r-^",
                 label="MLP",      linewidth=2)
    ax_cont.set_xlabel("rollout step k")
    ax_cont.set_ylabel("連続方程式残差 MSE")
    ax_cont.set_title("∂ρ/∂t + ∇·(ρv) の違反量", fontsize=9)
    ax_cont.legend(fontsize=7)
    ax_cont.grid(True, linestyle=":", alpha=0.5)

    # 解説テキスト（調和の解析解）
    theory_lines = [
        "【解析解】調和ポテンシャル Φ=½‖z‖²",
        "  ∇Φ = z  →  dz/dτ = -α·z",
        "  解: z(τ) = z(0)·exp(-ατ)",
        "  Euler 1 ステップ: z' = z·(1-αΔt)",
        f"  α={ALPHA}, Δt={DT}  →  z' = z·{1-ALPHA*DT:.2f}",
        "  J = (1-αΔt)·I",
        f"  det(J) = {(1-ALPHA*DT)**D:.4f}  (d={D})",
        f"  k=5 後: det(J)^5 = {(1-ALPHA*DT)**(D*N_STEPS):.4f}",
        "  ↑ 体積 99% 消失 → 分布が 1 点に潰れる",
    ]
    fig.text(0.01, 0.04, "\n".join(theory_lines),
             fontsize=7.5, family="monospace",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    out = "pnode_patent_runner/simulate_collapse.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    out_pdf = "pnode_patent_runner/simulate_collapse.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved → {out}")
    print(f"Saved → {out_pdf}")

    # ─ 数値サマリー ─
    print("\n" + "="*55)
    print(" 一点集中シミュレーション 数値サマリー")
    print("="*55)
    print(f"{'Step':>4}  {'det(J)^k':>10}  {'PW dist (H)':>12}  {'Cont res (H)':>13}")
    print("-"*55)
    for k in range(N_STEPS + 1):
        d_val = (1 - ALPHA * DT) ** (D * k)
        pw_val = pw_harm[k]
        cr_val = cont_harm[k - 1] if k > 0 else 0.0
        print(f"{k:>4}  {d_val:>10.5f}  {pw_val:>12.4f}  {cr_val:>13.6f}")
    print("="*55)


if __name__ == "__main__":
    run_simulation()
