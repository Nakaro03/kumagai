"""
PC-PNODE 景観マップ可視化 (Architecture A: TrendAdapter 込み)。

入力: チェックポイント trend_benchmark/ckpt/pnode_pc_seed{S}.pt
出力: pnode_patent_runner/outputs/trend_benchmark/landscape_map_seed{S}.png

景観の解釈:
  - z 平面 (latent_dim=2) で Φ(z_trend) の標高を等高線表示
  - 谷 = 低 Φ = 注目予測 = 成長技術が集まるべき領域
  - 山 = 高 Φ = 衰退予測
  - トピックノードは実際の成長率 g で色づけ
  - 期待: 緑 (高成長) ノードが谷に, 赤 (衰退) ノードが山に集まる
  - ベクトル場 -∇Φ: 著者がどの方向に流れるかを示す
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE
from pnode_patent_runner.checkpoint_utils import load_state_dict_skip_shape_mismatch
from pnode_patent_runner.cope_experiment import load_author_topic_graph_bundle

# 設定
SEED       = int(os.environ.get("PNODE_SEED", 0))   # 既定: r=-0.308 の seed
CKPT_DIR   = "pnode_patent_runner/outputs/trend_benchmark/ckpt"
DATA_CSV   = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
EVAL_YEAR  = 2024     # この年の z で景観を描画
YEAR_RANGE = (2022, 2025)
OUT_PNG    = f"pnode_patent_runner/outputs/trend_benchmark/landscape_map_seed{SEED}.png"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = load_author_topic_graph_bundle(DATA_CSV, min_papers=5, year_range=YEAR_RANGE)

ckpt = torch.load(f"{CKPT_DIR}/pnode_pc_seed{SEED}.pt", map_location=device)
model = BenchmarkTemporalVGAE(
    num_nodes=bundle.total_n, num_corps=bundle.num_corps,
    input_dim=bundle.in_dim, hidden_dim=128, latent_dim=2,
    initial_corp_vectors=bundle.init_vectors,
    link_score_mode="distance", variant="pnode_pc",
    pnode_potential_feature="mlp", year_min=2022, year_max=2025,
    topic_position_embedding=True,   # A+BEF 構成
).to(device)
load_state_dict_skip_shape_mismatch(model, ckpt["state_dict"])
model.eval()

# 成長率取得 (±1年平均)
growth = bundle.topic_growth_by_year or {}
all_gy = sorted(growth.keys())
smoothed = {}
for y in all_gy:
    nb = [yy for yy in all_gy if abs(yy - y) <= 1]
    smoothed[y] = torch.stack([growth[yy] for yy in nb]).mean(dim=0)

topics = [str(t) for t in bundle.right_nodes]
pot    = model.temporal_predictor.potential_net
num_corps = bundle.num_corps

with torch.no_grad():
    data_y = bundle.graphs[EVAL_YEAR].to(device)
    z, _, _ = model.encode(data_y.x, data_y.edge_index)
    pot.set_population(z.detach())

    z_authors = z[:num_corps]
    z_topics_raw = z[num_corps:]
    # Architecture A: adapter を通した z で Φ 計算
    z_topics = model.apply_trend_adapter(z_topics_raw)

z_authors_np = z_authors.cpu().numpy()
z_topics_np  = z_topics.cpu().numpy()
g_arr = smoothed[EVAL_YEAR].numpy()
n_t = min(len(topics), z_topics_np.shape[0], len(g_arr))
topics = topics[:n_t]
z_topics_np = z_topics_np[:n_t]
g_arr = g_arr[:n_t]

# Φ グリッドを adapter 出力空間で描画
all_z = np.concatenate([z_authors_np, z_topics_np], axis=0)
mn = all_z.min(axis=0) - 0.05
mx = all_z.max(axis=0) + 0.05
res = 80
xs = np.linspace(mn[0], mx[0], res)
ys = np.linspace(mn[1], mx[1], res)
X, Y = np.meshgrid(xs, ys, indexing="ij")
grid = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1),
                     dtype=torch.float32, device=device).requires_grad_(True)
phi_grid = pot(grid).squeeze(-1)

# ∇Φ for vector field
grads = torch.autograd.grad(phi_grid.sum(), grid)[0].detach().cpu().numpy()
U = -grads[:, 0].reshape(res, res)
V = -grads[:, 1].reshape(res, res)
Phi = phi_grid.detach().cpu().numpy().reshape(res, res)

fig, ax = plt.subplots(figsize=(11, 8))

# Φ 等高線 (谷=緑, 山=赤に近い色)
cf = ax.contourf(X, Y, Phi, levels=18, cmap="RdYlGn_r", alpha=0.7)
cs = ax.contour(X, Y, Phi, levels=10, colors="white", linewidths=0.4, alpha=0.5)
plt.colorbar(cf, ax=ax, label="Φ(z_trend)  (low = valley = predicted growing)", fraction=0.04)

# ベクトル場 -∇Φ (著者の流れる方向)
step = 6
ax.quiver(X[::step, ::step], Y[::step, ::step],
          U[::step, ::step], V[::step, ::step],
          color="white", alpha=0.5, scale=80, width=0.0028)

# 著者ノード (淡いシアン散布)
ax.scatter(z_authors_np[:, 0], z_authors_np[:, 1],
           s=4, color="cyan", alpha=0.18, zorder=2, label=f"Authors (n={len(z_authors_np)})")

# トピックノード: 実際の成長率で色づけ
g_clip = np.clip(g_arr, -0.5, 1.0)
sc = ax.scatter(z_topics_np[:, 0], z_topics_np[:, 1],
                s=120, c=g_arr, cmap="RdYlGn", vmin=-0.3, vmax=0.7,
                edgecolors="black", linewidths=1.0, zorder=5,
                label=f"Topics (n={n_t})")

# 全トピックにラベル
for i, t in enumerate(topics):
    ax.annotate(t, (z_topics_np[i, 0], z_topics_np[i, 1]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7.5, color="black",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, edgecolor="none"))

ax.set_title(
    f"PC-PNODE Trend Landscape  |  ArXiv CS, year {EVAL_YEAR}, seed={SEED}\n"
    f"Background = Φ(z_trend) potential  |  Topic color = actual growth rate  |  "
    f"White arrows = -∇Φ (author flow direction)",
    fontsize=10
)
ax.set_xlabel("z_trend axis 1")
ax.set_ylabel("z_trend axis 2")
ax.legend(loc="lower left", fontsize=8)

# 補助カラーバー: トピック色
cax = fig.add_axes([0.92, 0.12, 0.012, 0.30])
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(-0.3, 0.7))
sm.set_array([])
plt.colorbar(sm, cax=cax, label="Topic growth g")

# Φ ランキング上位 5 / 下位 5 を抽出
phi_topics_val = pot(torch.tensor(z_topics_np, device=device, dtype=torch.float32)).squeeze(-1).detach().cpu().numpy()
order = np.argsort(phi_topics_val)
print("Φ 最小 (谷=注目予測) TOP 5:")
for i in order[:5]:
    print(f"  {topics[i]:<8} Φ={phi_topics_val[i]:+.4f}  g_actual={g_arr[i]:+.4f}")
print("Φ 最大 (山=衰退予測) TOP 5:")
for i in order[-5:][::-1]:
    print(f"  {topics[i]:<8} Φ={phi_topics_val[i]:+.4f}  g_actual={g_arr[i]:+.4f}")

Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
print(f"\nSaved -> {OUT_PNG}")
