"""
P-NODE vs P-NODE+Continuity の学習方法と分布変化を可視化する。

【学習の違い】
  P-NODE            : L = L_recon + λ_traj · L_traj
  P-NODE+Continuity : L = L_recon + λ_traj · L_traj + λ_cont · L_cont

【L_cont（連続方程式残差）】
  L_cont = ||∂ρ/∂t + ∇·(ρv)||²

【推論時の ODE】
  P-NODE            : dz/dτ = -α∇Φ(z)
  P-NODE+Continuity : dz/dτ = -α∇Φ(z) + γ·∇log ρ̂(z)
    ∇log ρ̂(z) = KDE スコア関数 → 高密度領域から押し出す力

出力: pnode_patent_runner/pnode_continuity_comparison.png
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import torch
import torch.nn as nn

sys.path.insert(0, ".")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

CKPT    = "pnode_patent_runner/outputs/ablation_pnode/ckpt/pnode_seed42.pt"
N       = 300
D       = 2
DT      = 1.0
N_STEPS = 5
SEED    = 42
GAMMA   = 0.8    # continuity 項の係数
BW      = 0.4    # KDE bandwidth

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


# ── モデル ──────────────────────────────────────────────────────────────────

class _PotentialNetCompat(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=128):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(0.1))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), nn.LeakyReLU(0.2), nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1), nn.Softplus())
    def forward(self, z): return self.head(self.in_proj(z))


def load_pnode(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["state_dict"]
    prefix = "temporal_predictor.ode_func.potential_net."
    pot_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    pot = _PotentialNetCompat(); pot.load_state_dict(pot_sd); pot.eval()
    alpha = float(torch.tanh(sd["temporal_predictor.ode_func.scale"]).item())
    return pot, alpha


# ── KDE スコア関数 ∇log ρ̂(z) ────────────────────────────────────────────────

def kde_score(z_np: np.ndarray, anchors_np: np.ndarray, bw: float) -> np.ndarray:
    """
    ∇_z log ρ̂(z) を解析的に計算。
    ρ̂(z) = (1/M) Σ_i exp(-||z-μ_i||²/(2h²))
    ∇log ρ̂(z) = -Σ_i w_i · (z - μ_i) / h²
    w_i = κ(z,μ_i) / Σ_j κ(z,μ_j)   (ソフトマックス重み)
    """
    z = z_np[:, None, :]          # (N, 1, D)
    a = anchors_np[None, :, :]    # (1, M, D)
    diff   = z - a                # (N, M, D)
    sq     = (diff**2).sum(-1)    # (N, M)
    log_k  = -sq / (2 * bw**2)
    log_k -= log_k.max(axis=1, keepdims=True)
    w = np.exp(log_k)
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    grad = -(w[:, :, None] * diff).sum(axis=1) / (bw**2)  # (N, D)
    return grad


# ── ∇Φ 計算 ─────────────────────────────────────────────────────────────────

def grad_phi(pot, z_np):
    zt = torch.tensor(z_np, dtype=torch.float32, requires_grad=True)
    pot(zt).sum().backward()
    return zt.grad.detach().numpy()


# ── ODE ステップ ─────────────────────────────────────────────────────────────

def step_pnode(pot, z, alpha, dt=1.0):
    return z - alpha * grad_phi(pot, z) * dt

def step_pnode_cont(pot, z, alpha, gamma, bw, dt=1.0):
    gp  = grad_phi(pot, z)
    ks  = kde_score(z, z, bw)           # anchors = 現在の z 分布
    return z - alpha * gp * dt + gamma * ks * dt


# ── 平均ペアワイズ距離 ────────────────────────────────────────────────────────

def mean_pw(z):
    idx = rng.choice(len(z), size=min(100, len(z)), replace=False)
    s = z[idx]; d = s[:, None] - s[None, :]
    return float(np.sqrt((d**2).sum(-1)).mean())


# ── 初期分布 ────────────────────────────────────────────────────────────────

def make_z0():
    centers = [[2., 1.], [-2., 1.], [0., -2.]]
    return np.vstack([rng.normal(c, 0.5, (N//3, D)) for c in centers]).astype(np.float32)


# ── rollout ─────────────────────────────────────────────────────────────────

def rollout(pot, alpha, gamma, bw, use_cont=False):
    z = make_z0()
    traj = [z.copy()]
    pw   = [mean_pw(z)]
    for _ in range(N_STEPS):
        if use_cont:
            z = step_pnode_cont(pot, z, alpha, gamma, bw)
        else:
            z = step_pnode(pot, z, alpha)
        traj.append(z.copy())
        pw.append(mean_pw(z))
    return traj, pw


# ── KDE 密度グリッド ─────────────────────────────────────────────────────────

def kde_density_grid(z_np, bw=0.4, res=50, xlim=(-6,6), ylim=(-6,6)):
    from scipy.stats import gaussian_kde
    xs = np.linspace(*xlim, res); ys = np.linspace(*ylim, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()])
    try:
        kde = gaussian_kde(z_np.T, bw_method=bw)
        rho = kde(pts).reshape(res, res)
    except Exception:
        rho = np.zeros((res, res))
    return XX, YY, rho


# ── メイン ───────────────────────────────────────────────────────────────────

def run():
    pot, alpha = load_pnode(CKPT)
    print(f"alpha={alpha:.4f}, gamma={GAMMA}")

    traj_a, pw_a = rollout(pot, alpha, GAMMA, BW, use_cont=False)  # P-NODE のみ
    traj_b, pw_b = rollout(pot, alpha, GAMMA, BW, use_cont=True)   # +Continuity

    ks = list(range(N_STEPS + 1))

    # ── レイアウト ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d0d0d")

    gs_top  = gridspec.GridSpec(1, 1,  top=0.97, bottom=0.82,
                                left=0.02, right=0.98)
    gs_mid  = gridspec.GridSpec(2, N_STEPS+1, top=0.79, bottom=0.40,
                                left=0.03, right=0.97,
                                hspace=0.08, wspace=0.08)
    gs_bot  = gridspec.GridSpec(1, 3,  top=0.35, bottom=0.06,
                                left=0.04, right=0.97, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # パネル 0: 学習方法の説明（テキストボックス）
    # ─────────────────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs_top[0])
    ax_title.set_facecolor("#0d0d0d")
    ax_title.axis("off")

    # タイトル
    ax_title.text(0.5, 0.92,
        "P-NODE vs P-NODE + Continuity:  Training Method Comparison",
        transform=ax_title.transAxes, ha="center", va="top",
        fontsize=13, fontweight="bold", color="white")

    # 2列のボックス
    box_kw = dict(transform=ax_title.transAxes, va="top",
                  fontsize=9.5, family="monospace")

    # 左ボックス: P-NODE
    ax_title.text(0.22, 0.72,
        "P-NODE  (existing)",
        ha="center", color="#ff6b6b", fontsize=10, fontweight="bold",
        transform=ax_title.transAxes)
    lines_a = [
        "L = L_recon",
        "  + λ_traj · L_traj",
        "",
        "ODE:  dz/dτ = -α ∇Φ(z)",
        "",
        "Problem: density can pile up",
        "  → all nodes → same valley",
    ]
    ax_title.text(0.22, 0.60, "\n".join(lines_a), ha="center",
        color="#ffaaaa", **box_kw,
        bbox=dict(boxstyle="round,pad=0.6", fc="#2a0a0a", ec="#ff6b6b", lw=1.5))

    # 中央の矢印
    ax_title.annotate("", xy=(0.53, 0.38), xytext=(0.47, 0.38),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax_title.text(0.5, 0.44, "+ L_cont", ha="center", color="white",
        fontsize=10, fontweight="bold", transform=ax_title.transAxes)

    # 右ボックス: P-NODE+Continuity
    ax_title.text(0.77, 0.72,
        "P-NODE + Continuity  (proposed)",
        ha="center", color="#6bffb8", fontsize=10, fontweight="bold",
        transform=ax_title.transAxes)
    lines_b = [
        "L = L_recon",
        "  + λ_traj · L_traj",
        "  + λ_cont · L_cont          ← NEW",
        "",
        "L_cont = ||∂ρ/∂t + ∇·(ρv)||²",
        "",
        "ODE:  dz/dτ = -α∇Φ(z)",
        "            + γ·∇log ρ̂(z)   ← density score",
    ]
    ax_title.text(0.77, 0.60, "\n".join(lines_b), ha="center",
        color="#aaffdd", **box_kw,
        bbox=dict(boxstyle="round,pad=0.6", fc="#0a2a1a", ec="#6bffb8", lw=1.5))

    # ─────────────────────────────────────────────────────────────────────────
    # パネル 1〜2: 散布図（k=0〜5）
    # ─────────────────────────────────────────────────────────────────────────
    clrs_a = plt.cm.Reds (np.linspace(0.35, 0.9, N_STEPS+1))
    clrs_b = plt.cm.Greens(np.linspace(0.35, 0.9, N_STEPS+1))

    for row, (traj, clrs, label) in enumerate([
        (traj_a, clrs_a, "P-NODE"),
        (traj_b, clrs_b, "P-NODE + Continuity"),
    ]):
        for col in range(N_STEPS + 1):
            ax = fig.add_subplot(gs_mid[row, col])
            ax.set_facecolor("#111111")
            z = traj[col]

            # KDE 等高線
            XX, YY, rho = kde_density_grid(z, bw=BW)
            ax.contourf(XX, YY, rho, levels=8, cmap="hot" if row==0 else "cool",
                        alpha=0.35)
            ax.scatter(z[:, 0], z[:, 1], s=3, color=clrs[col],
                       alpha=0.6, linewidths=0)
            ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_color("#444444")

            if col == 0:
                ax.set_ylabel(label, fontsize=8, color="white")
            ax.set_title(f"k={col}", fontsize=8, color="#aaaaaa", pad=2)
            ax.yaxis.label.set_color("white")

    # ─────────────────────────────────────────────────────────────────────────
    # パネル 3: ペアワイズ距離 / 密度分散 / 比較
    # ─────────────────────────────────────────────────────────────────────────

    # 左: PW distance
    ax_pw = fig.add_subplot(gs_bot[0])
    ax_pw.set_facecolor("#111111")
    ax_pw.plot(ks, pw_a, "o-", color="#ff6b6b", lw=2, ms=6, label="P-NODE")
    ax_pw.plot(ks, pw_b, "s-", color="#6bffb8", lw=2, ms=6,
               label="P-NODE+Continuity")
    ax_pw.set_xlabel("rollout step k", color="white")
    ax_pw.set_ylabel("Mean pairwise distance", color="white")
    ax_pw.set_title("Embedding Collapse", color="white", fontsize=10)
    ax_pw.legend(fontsize=8)
    ax_pw.tick_params(colors="white"); ax_pw.spines[:].set_color("#444")
    ax_pw.set_facecolor("#111111")

    # 中: KDE エントロピー（分散の代理指標）
    def kde_entropy(z):
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(z.T, bw_method=BW)
            pts = z
            log_rho = np.log(kde(pts.T) + 1e-12)
            return -float(log_rho.mean())
        except Exception:
            return 0.0

    ent_a = [kde_entropy(t) for t in traj_a]
    ent_b = [kde_entropy(t) for t in traj_b]

    ax_ent = fig.add_subplot(gs_bot[1])
    ax_ent.set_facecolor("#111111")
    ax_ent.plot(ks, ent_a, "o-", color="#ff6b6b", lw=2, ms=6, label="P-NODE")
    ax_ent.plot(ks, ent_b, "s-", color="#6bffb8", lw=2, ms=6,
                label="P-NODE+Continuity")
    ax_ent.set_xlabel("rollout step k", color="white")
    ax_ent.set_ylabel("-E[log rho]   (higher = more spread)", color="white")
    ax_ent.set_title("Distribution Entropy", color="white", fontsize=10)
    ax_ent.legend(fontsize=8)
    ax_ent.tick_params(colors="white"); ax_ent.spines[:].set_color("#444")

    # 右: k=5 の密度比較（KDE プロファイル）
    ax_kde = fig.add_subplot(gs_bot[2])
    ax_kde.set_facecolor("#111111")
    z_a5 = traj_a[-1]; z_b5 = traj_b[-1]
    xs_line = np.linspace(-6, 6, 200)
    from scipy.stats import gaussian_kde as gkde
    for z5, col, lbl in [(z_a5, "#ff6b6b", "P-NODE  k=5"),
                          (z_b5, "#6bffb8", "P-NODE+Cont k=5"),
                          (traj_a[0], "#888888", "Initial k=0")]:
        try:
            kde_1d = gkde(z5[:, 0], bw_method=BW)
            ax_kde.plot(xs_line, kde_1d(xs_line), color=col, lw=2, label=lbl)
        except Exception:
            pass
    ax_kde.set_xlabel("z1  (marginal density)", color="white")
    ax_kde.set_ylabel("density", color="white")
    ax_kde.set_title("Marginal Density at k=5\n(z1 axis)", color="white", fontsize=10)
    ax_kde.legend(fontsize=8)
    ax_kde.tick_params(colors="white"); ax_kde.spines[:].set_color("#444")

    # 数値サマリー
    print("\n" + "="*55)
    print("  P-NODE vs P-NODE+Continuity  サマリー")
    print("="*55)
    print(f"  gamma (continuity weight) = {GAMMA}")
    print(f"{'Step':>4}  {'PW(P-NODE)':>12}  {'PW(+Cont)':>12}  {'H(P-NODE)':>10}  {'H(+Cont)':>10}")
    print("-"*55)
    for k in ks:
        print(f"{k:>4}  {pw_a[k]:>12.4f}  {pw_b[k]:>12.4f}  "
              f"{ent_a[k]:>10.4f}  {ent_b[k]:>10.4f}")
    print("="*55)
    collapse_a = pw_a[0] / max(pw_a[-1], 1e-9)
    collapse_b = pw_b[0] / max(pw_b[-1], 1e-9)
    print(f"  PW 崩壊倍率: P-NODE={collapse_a:.1f}x  +Cont={collapse_b:.1f}x")

    out = "pnode_patent_runner/pnode_continuity_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    run()
