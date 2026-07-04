"""
学習済み P-NODE の Φ(z) ポテンシャル景観 + ∇Φ ベクトル場を可視化する。
latent_dim=2 なので直接 2D 平面に描画できる（UMAP 不要）。

出力: pnode_patent_runner/phi_vector_field.png
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import torch.nn as nn

sys.path.insert(0, ".")

CKPT    = "pnode_patent_runner/outputs/ablation_pnode/ckpt/pnode_seed42.pt"
D       = 2
SEED    = 42
rng     = np.random.default_rng(SEED)
torch.manual_seed(SEED)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


# ── 学習済み PotentialNet（spectral_norm なし版） ──────────────────────────────

class _PotentialNetCompat(nn.Module):
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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    prefix = "temporal_predictor.ode_func.potential_net."
    pot_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    pot_net = _PotentialNetCompat(latent_dim=D, hidden_dim=128)
    pot_net.load_state_dict(pot_sd, strict=True)
    pot_net.eval()
    scale = float(sd["temporal_predictor.ode_func.scale"].item())
    alpha = float(torch.tanh(torch.tensor(scale)).item())
    return pot_net, alpha


# ── グリッド上で Φ と ∇Φ を計算 ──────────────────────────────────────────────

def compute_phi_and_grad(model: nn.Module, x_range=(-5, 5), y_range=(-5, 5),
                         res_phi: int = 80, res_grad: int = 20):
    """
    fine grid (res_phi²)  : Φ の等高線用
    coarse grid (res_grad²): ∇Φ のベクトル場用
    """
    # fine grid
    xs = np.linspace(*x_range, res_phi)
    ys = np.linspace(*y_range, res_phi)
    XX, YY = np.meshgrid(xs, ys)
    pts = torch.tensor(
        np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32
    )
    with torch.no_grad():
        phi_vals = model(pts).numpy().reshape(res_phi, res_phi)

    # coarse grid for quiver
    xq = np.linspace(*x_range, res_grad)
    yq = np.linspace(*y_range, res_grad)
    XQ, YQ = np.meshgrid(xq, yq)
    ptsq = torch.tensor(
        np.stack([XQ.ravel(), YQ.ravel()], axis=1), dtype=torch.float32,
        requires_grad=False,
    )
    ptsq.requires_grad_(True)
    phi_q = model(ptsq).sum()
    phi_q.backward()
    grad_np = ptsq.grad.detach().numpy().reshape(res_grad, res_grad, 2)
    GX = grad_np[:, :, 0]   # ∂Φ/∂z₁
    GY = grad_np[:, :, 1]   # ∂Φ/∂z₂

    return XX, YY, phi_vals, XQ, YQ, GX, GY


# ── 初期分布 3 クラスタ + 軌跡 ────────────────────────────────────────────────

def make_z0() -> np.ndarray:
    centers = np.array([[2.0, 1.0], [-2.0, 1.0], [0.0, -2.0]])
    parts = [rng.normal(loc=c, scale=0.5, size=(30, D)) for c in centers]
    return np.vstack(parts).astype(np.float32)


def grad_phi_numpy(model: nn.Module, z_np: np.ndarray) -> np.ndarray:
    zt = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
    model(zt).sum().backward()
    return zt.grad.detach().numpy()


def rollout(model: nn.Module, z0: np.ndarray, alpha: float,
            dt: float = 1.0, n_steps: int = 6):
    traj = [z0.copy()]
    z = z0.copy()
    for _ in range(n_steps):
        z = z - alpha * grad_phi_numpy(model, z) * dt
        traj.append(z.copy())
    return traj   # list of (N, 2)


# ── メインプロット ─────────────────────────────────────────────────────────────

def run():
    pot_net, alpha = load_pnode(CKPT)
    print(f"alpha = {alpha:.4f}")

    XX, YY, phi_vals, XQ, YQ, GX, GY = compute_phi_and_grad(pot_net)

    z0   = make_z0()
    traj = rollout(pot_net, z0, alpha, n_steps=6)

    # 速度場 v = -α∇Φ（発明者が実際に動く方向）
    VX = -alpha * GX
    VY = -alpha * GY
    speed = np.sqrt(VX**2 + VY**2) + 1e-8

    # ── 3 パネル ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Trained P-NODE: Phi landscape + gradient vector field  (alpha={alpha:.3f})",
        fontsize=11, fontweight="bold",
    )

    # ─ パネル 1: Φ の等高線 + ベクトル場 ──────────────────────────────────────
    ax = axes[0]
    cf = ax.contourf(XX, YY, phi_vals, levels=30, cmap="viridis", alpha=0.85)
    plt.colorbar(cf, ax=ax, fraction=0.046, label="Phi(z)")
    # quiver: 正規化して方向だけ示す
    ax.quiver(XQ, YQ, -GX / (speed / alpha), -GY / (speed / alpha),
              alpha=0.7, color="white", scale=25, width=0.003)
    ax.set_title("Phi(z) + velocity field  v = -alpha*grad_Phi", fontsize=9)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")

    # ─ パネル 2: Φ 等高線 + 発明者軌跡（200 ステップ極小点まで）──────────────
    ax = axes[1]
    ax.contourf(XX, YY, phi_vals, levels=30, cmap="viridis", alpha=0.6)
    colors_c = ["#e74c3c", "#3498db", "#2ecc71"]
    n_each = len(z0) // 3
    # 各クラスタ中心の長期軌跡（200 step）
    centers = np.array([[2.0, 1.0], [-2.0, 1.0], [0.0, -2.0]])
    attractor_pts = []
    for ci, c in enumerate(centers):
        z_long = torch.tensor([c], dtype=torch.float32)
        long_path = [c.tolist()]
        for _ in range(200):
            z_long = z_long.detach().requires_grad_(True)
            pot_net(z_long).sum().backward()
            z_long = (z_long - alpha * z_long.grad).detach()
            if _ % 20 == 0:
                long_path.append(z_long[0].numpy().tolist())
        attractor_pts.append(z_long[0].numpy())
        lp = np.array(long_path)
        ax.plot(lp[:, 0], lp[:, 1], color=colors_c[ci], lw=1.5, alpha=0.7,
                zorder=3, label=f"cluster {ci} path")

    for ci in range(3):
        sl = slice(ci * n_each, (ci + 1) * n_each)
        ax.scatter(z0[sl, 0], z0[sl, 1], s=20, color=colors_c[ci],
                   edgecolors="white", linewidths=0.4, zorder=4)
    # 収束先（極小点）を大きなマーカーで表示
    for ci, ap in enumerate(attractor_pts):
        ax.scatter(ap[0], ap[1], s=150, marker="*", color=colors_c[ci],
                   edgecolors="black", linewidths=0.8, zorder=5,
                   label=f"attractor {ci} ({ap[0]:.1f},{ap[1]:.1f})")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("Long-run trajectories to local minima (200 steps)", fontsize=9)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")

    # ─ パネル 3: |∇Φ| の大きさ（どこで力が強いか）──────────────────────────
    ax = axes[2]
    grad_mag = np.sqrt(GX**2 + GY**2)
    # fine grid で magnitude
    xf = np.linspace(-5, 5, 80)
    yf = np.linspace(-5, 5, 80)
    XF, YF = np.meshgrid(xf, yf)
    ptsf = torch.tensor(
        np.stack([XF.ravel(), YF.ravel()], axis=1), dtype=torch.float32,
        requires_grad=False,
    )
    ptsf.requires_grad_(True)
    pot_net(ptsf).sum().backward()
    gmag = ptsf.grad.detach().norm(dim=1).numpy().reshape(80, 80)
    cf3 = ax.contourf(XF, YF, gmag, levels=30, cmap="hot_r", alpha=0.9)
    plt.colorbar(cf3, ax=ax, fraction=0.046, label="|grad_Phi|")
    ax.quiver(XQ, YQ, -GX / (speed / alpha), -GY / (speed / alpha),
              alpha=0.6, color="white", scale=25, width=0.003)
    ax.set_title("|grad Phi(z)|  (brighter = stronger pull)", fontsize=9)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")

    for ax in axes:
        ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

    out = "pnode_patent_runner/phi_vector_field.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")

    # 数値サマリー
    print(f"\nPhi range:        [{phi_vals.min():.3f}, {phi_vals.max():.3f}]")
    print(f"|grad Phi| range: [{gmag.min():.4f}, {gmag.max():.4f}]")
    print(f"Mean speed |v|:   {(alpha * gmag).mean():.4f}")
    minima_idx = np.unravel_index(phi_vals.argmin(), phi_vals.shape)
    print(f"Approx min of Phi: z1={XX[minima_idx]:.2f}, z2={YY[minima_idx]:.2f}")


if __name__ == "__main__":
    run()
