"""
(A, hybrid) Best of both: layout from DIRECTION-normalised PCA (rich, balanced
2D map) but heat = RAW memory velocity (the strong, magnitude-bearing 'rising'
signal). Decouples 'where the map is' from 'what is heating up'.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

D = "/tmp/EDGPAT/data"; TAG = "G"; OUT = Path("/tmp/EDGPAT/viz"); OUT.mkdir(exist_ok=True)
z = np.load(f"{D}/cpc_traj_{TAG}.npz", allow_pickle=True)
traj, years, cpc_str = z["traj"], z["years"], z["cpc_str"]
T, N, dim = traj.shape

idx = np.where(np.linalg.norm(traj[-1], axis=1) > 1e-6)[0]
V = traj[:, idx, :]; codes = cpc_str[idx]; Na = len(idx)

# layout: direction-normalised PCA (balanced 2D)
nrm = np.linalg.norm(V, axis=2, keepdims=True)
Vn = np.where(nrm > 1e-6, V / np.maximum(nrm, 1e-9), 0.0)
P = PCA(n_components=2, random_state=0).fit_transform(Vn[-1])

# heat: RAW memory velocity (strong rising signal)
mom = np.zeros((T, Na))
for t in range(1, T):
    mom[t] = np.linalg.norm(V[t] - V[t - 1], axis=1)

nn = NearestNeighbors(n_neighbors=9).fit(P); _, knn = nn.kneighbors(P); knn = knn[:, 1:]
def moran(x):
    x = x - x.mean()
    if x.std() < 1e-9: return 0.0
    return (1.0/8) * (sum(x[i]*x[j] for i in range(Na) for j in knn[i]) / (x**2).sum())
print("=== Moran's I: RAW momentum on NORMALISED layout (legibility) ===")
for t in range(1, T):
    print(f"  {years[t]}: I={moran(mom[t]):+.3f}")

vmax = np.percentile(mom[mom > 0], 97)
fig, ax = plt.subplots(figsize=(8.5, 7))
def draw(t):
    ax.clear()
    sc = ax.scatter(P[:, 0], P[:, 1], c=mom[t], s=10 + 50*(mom[t]/(vmax+1e-9)).clip(0,1),
                    cmap="inferno", vmin=0, vmax=vmax, alpha=0.82, linewidths=0)
    top = np.argsort(-mom[t])[:8]
    ax.text(0.015, 0.985, "top rising:\n" + "\n".join(codes[k] for k in top),
            transform=ax.transAxes, fontsize=8, va="top", color="white",
            family="monospace", bbox=dict(boxstyle="round", fc="black", alpha=0.6))
    ax.set_title(f"Rising technologies — EDGPAT latent map — {years[t]}")
    ax.set_xlabel("PC1 (direction)"); ax.set_ylabel("PC2 (direction)")
    ax.set_xticks([]); ax.set_yticks([]); return sc,
sc = draw(T-1); fig.colorbar(sc[0], ax=ax, label="memory velocity (rising)")
FuncAnimation(fig, draw, frames=range(1, T), interval=700).save(
    OUT / f"rising_hybrid_{TAG}.gif", writer=PillowWriter(fps=1.4))
for t in range(1, T):
    draw(t); fig.savefig(OUT / f"frame_hybrid_{years[t]}.png", dpi=110)
plt.close(fig)
print(f"saved {OUT}/rising_hybrid_{TAG}.gif + frame_hybrid_*.png")
