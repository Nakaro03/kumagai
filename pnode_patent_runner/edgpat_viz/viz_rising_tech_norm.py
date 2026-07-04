"""
(A, normalised) Rising-technology map on EDGPAT's latent space, but with each
CPC memory vector L2-normalised BEFORE PCA so the layout reflects DIRECTION
(technological character) rather than magnitude (activity/maturity) which
otherwise dominates PC1. Momentum is directional change ||v_t/|v_t| - ...||.
"""
import warnings; warnings.filterwarnings("ignore")
import ast, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

D = "/tmp/EDGPAT/data"; TAG = "G"; OUT = Path("/tmp/EDGPAT/viz"); OUT.mkdir(exist_ok=True)

z = np.load(f"{D}/cpc_traj_{TAG}.npz", allow_pickle=True)
traj, years = z["traj"], z["years"]
cpc_str = z["cpc_str"]
T, N, dim = traj.shape

active = np.linalg.norm(traj[-1], axis=1) > 1e-6
idx = np.where(active)[0]
V = traj[:, idx, :]                                  # (T, Na, dim)
codes = cpc_str[idx]; Na = len(idx)

# --- L2-normalise each vector (direction only) -----------------------------
nrm = np.linalg.norm(V, axis=2, keepdims=True)
Vn = np.where(nrm > 1e-6, V / np.maximum(nrm, 1e-9), 0.0)

pca = PCA(n_components=2, random_state=0).fit(Vn[-1])
P = pca.transform(Vn[-1])
print("normalised PCA explained var (2D):", pca.explained_variance_ratio_.round(3).tolist())

# directional momentum
mom = np.zeros((T, Na))
for t in range(1, T):
    mom[t] = np.linalg.norm(Vn[t] - Vn[t - 1], axis=1)

# --- Moran's I -------------------------------------------------------------
nn = NearestNeighbors(n_neighbors=9).fit(P); _, knn = nn.kneighbors(P); knn = knn[:, 1:]
def moran(x):
    x = x - x.mean()
    if x.std() < 1e-9: return 0.0
    num = sum(x[i] * x[j] for i in range(Na) for j in knn[i])
    return (1.0 / 8) * (num / (x ** 2).sum())
print("\n=== Moran's I (normalised directional momentum) ===")
for t in range(1, T):
    print(f"  {years[t]}: I={moran(mom[t]):+.3f}")

print("\n=== top-5 rising CPCs (directional) ===")
for t in range(1, T):
    top = np.argsort(-mom[t])[:5]
    print(f"  {years[t]}: " + ", ".join(f"{codes[k]}" for k in top))

# --- animation with corner label box (no overlap) --------------------------
vmax = np.percentile(mom[mom > 0], 97)
fig, ax = plt.subplots(figsize=(8.5, 7))
def draw(t):
    ax.clear()
    sc = ax.scatter(P[:, 0], P[:, 1], c=mom[t],
                    s=10 + 45 * (mom[t] / (vmax + 1e-9)).clip(0, 1),
                    cmap="inferno", vmin=0, vmax=vmax, alpha=0.82, linewidths=0)
    top = np.argsort(-mom[t])[:8]
    txt = "top rising:\n" + "\n".join(f"{codes[k]}" for k in top)
    ax.text(0.015, 0.985, txt, transform=ax.transAxes, fontsize=8, va="top",
            color="white", family="monospace",
            bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    ax.set_title(f"Rising technologies (EDGPAT, direction-normalised) — {years[t]}")
    ax.set_xlabel("PC1 (direction)"); ax.set_ylabel("PC2 (direction)")
    ax.set_xticks([]); ax.set_yticks([])
    return sc,
sc = draw(T - 1); fig.colorbar(sc[0], ax=ax, label="directional momentum")
FuncAnimation(fig, draw, frames=range(1, T), interval=700).save(
    OUT / f"rising_norm_{TAG}.gif", writer=PillowWriter(fps=1.4))
for t in range(1, T):
    draw(t); fig.savefig(OUT / f"frame_norm_{years[t]}.png", dpi=110)
plt.close(fig)
print(f"\nsaved {OUT}/rising_norm_{TAG}.gif  + frame_norm_*.png")
