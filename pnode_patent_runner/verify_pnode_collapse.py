"""
学習済み P-NODE（pnode_seed42.pt）で崩壊現象を実測する。

同じ初期分布 z₀（3クラスタ Gaussian）を：
  1. 学習済み P-NODE Φ_θ（trained）で積分
  2. 調和ポテンシャル Φ=½‖z‖²（解析基準）で積分
の両方に適用し、collapse 指標を比較する。

出力:
  pnode_patent_runner/verify_pnode_collapse.png
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

CKPT      = "pnode_patent_runner/outputs/ablation_pnode/ckpt/pnode_seed42.pt"
N         = 300   # 発明者数
D         = 2     # latent 次元（チェックポイントも latent_dim=2）
DT        = 1.0   # Δτ（1 年ステップ）
N_STEPS   = 5
SEED      = 42

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ── チェックポイントに合わせた軽量 PotentialNet ───────────────────────────────
# pnode_seed42.pt は spectral_norm なし（weight/bias のみ）+ scale パラメータ版
# head: 0=Linear(128,256), 1=LeakyReLU, 2=Dropout, 3=Linear(256,128),
#       4=LeakyReLU, 5=Linear(128,1), 6=Softplus
# in_proj: 0=Linear(2,128), 1=LeakyReLU, 2=Dropout

class _PotentialNetCompat(nn.Module):
    """State-dict 互換の軽量 Φ（学習済み重みの読み込み専用）。"""
    def __init__(self, latent_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.in_proj(z))


def load_pnode(ckpt_path: str):
    """
    チェックポイントから Φ_θ と α を復元する。
    旧 API: ode_func.scale、α = tanh(scale)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # potential_net の重みを抽出
    prefix = "temporal_predictor.ode_func.potential_net."
    pot_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    pot_net = _PotentialNetCompat(latent_dim=D, hidden_dim=128)
    pot_net.load_state_dict(pot_sd, strict=True)
    pot_net.eval()

    # 旧コードは alpha = tanh(scale)
    scale_val = float(sd["temporal_predictor.ode_func.scale"].item())
    alpha = float(torch.tanh(torch.tensor(scale_val)).item())
    print(f"[ckpt] scale={scale_val:.4f}  →  α = tanh(scale) = {alpha:.4f}")

    return pot_net, alpha


# ── 勾配計算 ─────────────────────────────────────────────────────────────────

def grad_phi_trained(model: nn.Module, z_np: np.ndarray) -> np.ndarray:
    """学習済み Φ_θ の ∇Φ（numpy → numpy）。"""
    zt = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
    phi = model(zt).sum()
    phi.backward()
    return zt.grad.detach().numpy()


def grad_phi_harmonic(z_np: np.ndarray) -> np.ndarray:
    """解析 ∇Φ = z（調和ポテンシャル）。"""
    return z_np.copy()


# ── Euler 1 ステップ（torchdiffeq なしで完結させる） ─────────────────────────

def ode_step(grad_fn, z: np.ndarray, alpha: float, dt: float) -> np.ndarray:
    return z - alpha * grad_fn(z) * dt


# ── ヤコビアン行列式の数値近似 ────────────────────────────────────────────────

def jacobian_det_numerical(grad_fn, z: np.ndarray, alpha: float, dt: float,
                            n_sample: int = 50) -> float:
    """Euler マップ f(z) = z - α·dt·∇Φ(z) の数値ヤコビアン平均行列式。"""
    eps = 1e-3
    idx = rng.choice(len(z), size=min(n_sample, len(z)), replace=False)
    dets = []
    for i in idx:
        zi = z[i].copy()
        J = np.zeros((D, D))
        for j in range(D):
            zp = zi.copy(); zp[j] += eps
            zm = zi.copy(); zm[j] -= eps
            fp = zp - alpha * dt * grad_fn(zp[None])[0]
            fm = zm - alpha * dt * grad_fn(zm[None])[0]
            J[:, j] = (fp - fm) / (2 * eps)
        dets.append(abs(np.linalg.det(J)))
    return float(np.mean(dets))


def jacobian_det_harmonic(alpha: float, dt: float, d: int) -> float:
    return (1.0 - alpha * dt) ** d


# ── 連続方程式残差 ─────────────────────────────────────────────────────────────

def continuity_residual(z_t: np.ndarray, z_t1: np.ndarray,
                        grad_fn, alpha: float, dt: float) -> float:
    from scipy.stats import gaussian_kde
    xs = np.linspace(-5, 5, 30)
    ys = np.linspace(-5, 5, 30)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    kde_t  = gaussian_kde(z_t.T,  bw_method=0.4)
    kde_t1 = gaussian_kde(z_t1.T, bw_method=0.4)
    rho_t  = kde_t(pts.T).reshape(30, 30)
    rho_t1 = kde_t1(pts.T).reshape(30, 30)
    drho_dt = (rho_t1 - rho_t) / dt

    v = -alpha * grad_fn(pts)
    rho_v_x = rho_t * v[:, 0].reshape(30, 30)
    rho_v_y = rho_t * v[:, 1].reshape(30, 30)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    div_rho_v = (np.gradient(rho_v_x, dx, axis=1) +
                 np.gradient(rho_v_y, dy, axis=0))
    return float(np.mean((drho_dt + div_rho_v) ** 2))


# ── 初期分布 ──────────────────────────────────────────────────────────────────

def make_z0() -> np.ndarray:
    """3 クラスタ Gaussian（simulate_collapse.py と同一）。"""
    centers = np.array([[2.0, 1.0], [-2.0, 1.0], [0.0, -2.0]])
    parts = [rng.normal(loc=c, scale=0.5, size=(N // 3, D)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def mean_pw_dist(z: np.ndarray) -> float:
    idx = rng.choice(len(z), size=100, replace=False)
    sub = z[idx]
    diffs = sub[:, None] - sub[None, :]
    return float(np.sqrt((diffs ** 2).sum(axis=-1)).mean())


# ── メイン ────────────────────────────────────────────────────────────────────

def run():
    pot_net, alpha_trained = load_pnode(CKPT)
    alpha_harm = 0.8   # simulate_collapse.py と同一の解析基準

    grad_trained = lambda z: grad_phi_trained(pot_net, z)
    grad_harm    = grad_phi_harmonic

    z0 = make_z0()

    traj = {"trained": [z0.copy()], "harmonic": [z0.copy()]}
    dets = {"trained": [], "harmonic": [1.0]}
    cont = {"trained": [], "harmonic": []}
    pw   = {"trained": [mean_pw_dist(z0)], "harmonic": [mean_pw_dist(z0)]}

    for step in range(1, N_STEPS + 1):
        for key, gfn, alpha in [
            ("trained", grad_trained, alpha_trained),
            ("harmonic", grad_harm,   alpha_harm),
        ]:
            z_prev = traj[key][-1]
            z_next = ode_step(gfn, z_prev, alpha, DT)
            traj[key].append(z_next)
            pw[key].append(mean_pw_dist(z_next))

            # ヤコビアン
            if key == "trained":
                dets["trained"].append(
                    jacobian_det_numerical(gfn, z_prev, alpha, DT))
            else:
                dets["harmonic"].append(
                    jacobian_det_harmonic(alpha, DT, D) ** step)

            # 連続方程式
            cont[key].append(
                continuity_residual(z_prev, z_next, gfn, alpha, DT))

    # ── ポテンシャル景観 ──
    xs_grid = np.linspace(-5, 5, 60)
    ys_grid = np.linspace(-5, 5, 60)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    pts_grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        phi_grid = pot_net(torch.tensor(pts_grid)).numpy().reshape(60, 60)

    # ── プロット ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "学習済み P-NODE (Φ_θ, α≈{:.3f}) vs 調和ポテンシャル (α=0.8)\n"
        "崩壊検証: dz/dτ = -α∇Φ(z), Euler 積分".format(alpha_trained),
        fontsize=11, fontweight="bold", y=0.99,
    )
    ks = list(range(N_STEPS + 1))

    gs = gridspec.GridSpec(3, N_STEPS + 2, figure=fig, hspace=0.45, wspace=0.25)

    col_colors = {"trained": plt.cm.Reds(np.linspace(0.35, 0.9, N_STEPS + 1)),
                  "harmonic": plt.cm.Blues(np.linspace(0.35, 0.9, N_STEPS + 1))}

    # 行 0: trained P-NODE 散布図
    for col in range(N_STEPS + 1):
        ax = fig.add_subplot(gs[0, col])
        z = traj["trained"][col]
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5, color=col_colors["trained"][col])
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.set_title(f"k={col}", fontsize=8)
        if col == 0:
            ax.set_ylabel("Trained Φ_θ\n(P-NODE ckpt)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 行 1: 調和ポテンシャル 散布図
    for col in range(N_STEPS + 1):
        ax = fig.add_subplot(gs[1, col])
        z = traj["harmonic"][col]
        ax.scatter(z[:, 0], z[:, 1], s=4, alpha=0.5, color=col_colors["harmonic"][col])
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        if col == 0:
            ax.set_ylabel("Harmonic Φ=½‖z‖²\n(α=0.8 解析基準)", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # ポテンシャル景観（最終列）
    ax_phi = fig.add_subplot(gs[0:2, -1])
    c = ax_phi.contourf(XX, YY, phi_grid, levels=20, cmap="viridis")
    ax_phi.scatter(z0[:, 0], z0[:, 1], s=3, c="white", alpha=0.4, label="z₀")
    plt.colorbar(c, ax=ax_phi, fraction=0.04)
    ax_phi.set_title("Φ_θ(z) landscape", fontsize=8)
    ax_phi.set_xlabel("z₁"); ax_phi.set_ylabel("z₂")

    # ── 行 2 左: ヤコビアン行列式 ──
    ax_det = fig.add_subplot(gs[2, :2])
    ax_det.plot(ks, dets["harmonic"], "b-o", label=f"det(J)^k Harmonic (α={alpha_harm})", lw=2)
    ax_det.plot(ks[1:], dets["trained"], "r-^", label=f"det(J) Trained Φ_θ (α≈{alpha_trained:.3f})", lw=2)
    ax_det.axhline(0, color="k", lw=0.8, ls="--")
    ax_det.set_xlabel("rollout step k")
    ax_det.set_ylabel("Jacobian det (体積要素)")
    ax_det.set_title("体積要素の収縮", fontsize=9)
    ax_det.legend(fontsize=7)
    ax_det.grid(True, linestyle=":", alpha=0.5)

    # ── 行 2 中: ペアワイズ距離 ──
    ax_pw = fig.add_subplot(gs[2, 2:4])
    ax_pw.plot(ks, pw["harmonic"], "b-o", label="Harmonic", lw=2)
    ax_pw.plot(ks, pw["trained"],  "r-^", label="Trained Φ_θ", lw=2)
    ax_pw.set_xlabel("rollout step k")
    ax_pw.set_ylabel("平均ペアワイズ距離")
    ax_pw.set_title("発明者間距離の崩壊", fontsize=9)
    ax_pw.legend(fontsize=7)
    ax_pw.grid(True, linestyle=":", alpha=0.5)

    # ── 行 2 右: 連続方程式残差 ──
    ax_cont = fig.add_subplot(gs[2, 4:])
    ax_cont.plot(range(1, N_STEPS + 1), cont["harmonic"], "b-o", label="Harmonic", lw=2)
    ax_cont.plot(range(1, N_STEPS + 1), cont["trained"],  "r-^", label="Trained Φ_θ", lw=2)
    ax_cont.set_xlabel("rollout step k")
    ax_cont.set_ylabel("連続方程式残差 MSE")
    ax_cont.set_title("∂ρ/∂t + ∇·(ρv) 違反量", fontsize=9)
    ax_cont.legend(fontsize=7)
    ax_cont.grid(True, linestyle=":", alpha=0.5)

    # ── 数値サマリー ──
    print("\n" + "=" * 60)
    print(" 学習済み P-NODE 崩壊検証 サマリー")
    print("=" * 60)
    print(f"  α_trained (tanh(scale)) = {alpha_trained:.4f}")
    print(f"  α_harmonic (解析基準)    = {alpha_harm:.4f}")
    print(f"  初期 PW dist = {pw['trained'][0]:.4f}")
    print("-" * 60)
    print(f"{'Step':>4}  {'PW(trained)':>12}  {'PW(harm)':>10}  "
          f"{'det(J) trained':>16}  {'det(J) harm':>13}")
    print("-" * 60)
    for k in range(N_STEPS + 1):
        det_t = dets["trained"][k - 1] if k > 0 else 1.0
        det_h = dets["harmonic"][k]
        print(f"{k:>4}  {pw['trained'][k]:>12.5f}  {pw['harmonic'][k]:>10.5f}  "
              f"{det_t:>16.5f}  {det_h:>13.5f}")
    collapse_ratio = pw["trained"][0] / max(pw["trained"][-1], 1e-12)
    print("=" * 60)
    print(f"  P-NODE 崩壊倍率 (k=0→5): PW dist {collapse_ratio:.1f}x 圧縮")

    out = "pnode_patent_runner/verify_pnode_collapse.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    run()
