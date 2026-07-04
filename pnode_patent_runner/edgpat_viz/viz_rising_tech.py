"""
(A) Visualise "which technologies are heating up" on EDGPAT's learned latent map.
- Layout : PCA(2) of EDGPAT field_5 (CPC) memory  -> a fixed technology map.
- Heat   : per-year momentum  m_j^t = ||v_j^t - v_j^{t-1}||  (memory velocity),
           and (cross-check) raw filing-count growth.
- Gate   : Moran's I of the heat over the 2D map -> is "rising" spatially legible
           (clustered) or just scattered noise?
Outputs PNG frames + an animated GIF + a top-rising-CPC table.
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
traj, years = z["traj"], z["years"]                 # (T, N, dim), (T,)
cpc_str, cpc_cls = z["cpc_str"], z["cpc_cls"]
T, N, dim = traj.shape

active = np.linalg.norm(traj[-1], axis=1) > 1e-6    # CPCs ever touched (final year)
idx = np.where(active)[0]
V = traj[:, idx, :]                                 # (T, Na, dim)
codes, cls = cpc_str[idx], cpc_cls[idx]
Na = len(idx); print(f"active CPCs: {Na}")

# --- fixed technology map: PCA(2) on final-year vectors --------------------
pca = PCA(n_components=2, random_state=0).fit(V[-1])
P = pca.transform(V[-1])                             # (Na, 2) fixed positions
print("PCA explained var (2D):", pca.explained_variance_ratio_.round(3).tolist())

# --- heat signals ----------------------------------------------------------
mem_mom = np.zeros((T, Na))                          # memory velocity
for t in range(1, T):
    mem_mom[t] = np.linalg.norm(V[t] - V[t - 1], axis=1)

# raw filing-count growth per CPC per year
ev = pd.read_csv(f"{D}/patent_{TAG}.csv")
yr_list = sorted(ev["year"].unique().tolist())
cnt = np.zeros((T, N))
for ti, y in enumerate(yr_list):
    sub = ev[ev["year"] == y]["fields"]
    for s in sub:
        for c in ast.literal_eval(s):
            cnt[ti, c] += 1
cnt = cnt[:, idx]
file_mom = np.vstack([np.zeros((1, Na)), np.diff(cnt, axis=0)])   # YoY delta

# --- Moran's I (legibility gate) ------------------------------------------
nn = NearestNeighbors(n_neighbors=9).fit(P)
_, knn = nn.kneighbors(P)
knn = knn[:, 1:]                                      # drop self
def moran(x):
    x = x - x.mean()
    if x.std() < 1e-9: return 0.0
    num = sum(x[i] * x[j] for i in range(Na) for j in knn[i])
    den = (x ** 2).sum()
    return (Na / (Na * 8)) * (num / den)
print("\n=== Moran's I (spatial clustering of 'rising'; >0 = legible) ===")
for t in range(1, T):
    print(f"  {years[t]}: memory-velocity I={moran(mem_mom[t]):+.3f}   filing-growth I={moran(file_mom[t]):+.3f}")

# --- top rising CPCs per year (memory velocity) ---------------------------
print("\n=== top-5 rising CPCs by memory velocity ===")
for t in range(1, T):
    top = np.argsort(-mem_mom[t])[:5]
    print(f"  {years[t]}: " + ", ".join(f"{codes[k]}({mem_mom[t,k]:.1f})" for k in top))

# --- animation -------------------------------------------------------------
def frames(mom, name, label):
    vmax = np.percentile(mom[mom > 0], 97) if (mom > 0).any() else 1.0
    fig, ax = plt.subplots(figsize=(8, 7))
    def draw(t):
        ax.clear()
        sc = ax.scatter(P[:, 0], P[:, 1], c=mom[t], s=10 + 40 * (mom[t] / (vmax + 1e-9)).clip(0, 1),
                        cmap="inferno", vmin=0, vmax=vmax, alpha=0.8, linewidths=0)
        for k in np.argsort(-mom[t])[:6]:
            ax.annotate(codes[k], (P[k, 0], P[k, 1]), fontsize=7, color="cyan")
        ax.set_title(f"Rising technologies (EDGPAT latent map) — {label} — {years[t]}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_xticks([]); ax.set_yticks([])
        return sc,
    sc = draw(T - 1)
    fig.colorbar(sc[0], ax=ax, label=label)
    anim = FuncAnimation(fig, draw, frames=range(1, T), interval=700)
    anim.save(OUT / f"rising_{name}_{TAG}.gif", writer=PillowWriter(fps=1.4))
    for t in range(1, T):
        draw(t); fig.savefig(OUT / f"frame_{name}_{years[t]}.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT}/rising_{name}_{TAG}.gif")

frames(mem_mom, "memvel", "memory velocity")
frames(file_mom, "filegrow", "filing growth")
print("\nDONE. frames + GIFs in", OUT)
