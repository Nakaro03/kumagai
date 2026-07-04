"""
学習済み P-NODE エンコーダで実際の著者埋め込みを取得し、
Φ(z) 景観 + ∇Φ ベクトル場の上に重ねて可視化する。

latent_dim=2 なので UMAP は不要 — 潜在空間がそのまま 2D。

出力: pnode_patent_runner/real_embeddings_vector_field.png
"""
from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from torch_geometric.nn import GATConv
from pnode_patent_runner.data_arxiv import (
    preprocess_author_topic_data,
    filter_active_authors,
    build_author_topic_graphs,
    calculate_initial_author_vectors,
)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

CKPT      = "pnode_patent_runner/outputs/ablation_pnode/ckpt/pnode_seed42.pt"
DATA_CSV  = "data/processed/arxiv_cs_embedded_2020-2026_full.csv"
VIZ_YEAR  = 2024   # 可視化する年
GRID_RES  = 80     # Φ 等高線の解像度
QUIV_RES  = 20     # ベクトル場の矢印密度
MAX_NODES = 500    # 表示する著者数の上限（多すぎると重い）


# ── 1. チェックポイントと互換のモデル再構築 ────────────────────────────────────
# BatchNorm + スキップ接続なし（旧アーキテクチャ）

class _EncoderCompat(nn.Module):
    """チェックポイント互換エンコーダ（BatchNorm, skip なし）。"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1     = GATConv(in_channels,      hidden_channels * 2, heads=2, concat=False)
        self.conv2     = GATConv(hidden_channels * 2, hidden_channels,  heads=2, concat=False)
        self.conv3     = GATConv(hidden_channels,  hidden_channels,     heads=1, concat=False)
        self.conv_mu   = GATConv(hidden_channels,  out_channels,        heads=1, concat=False)
        self.conv_logvar = GATConv(hidden_channels, out_channels,       heads=1, concat=False)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        h1 = F.relu(self.batch_norm1(self.conv1(x, edge_index)))
        h1 = self.dropout(h1)
        h2 = F.relu(self.batch_norm2(self.conv2(h1, edge_index)))
        h2 = self.dropout(h2)
        h3 = F.relu(self.conv3(h2, edge_index))
        return self.conv_mu(h3, edge_index), self.conv_logvar(h3, edge_index)


class _PotentialNetCompat(nn.Module):
    """チェックポイント互換 Φ（spectral_norm なし）。"""
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


def load_models(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # ── エンコーダ ──
    in_ch  = sd["corp_embeddings.weight"].shape[1]   # 1024
    hid_ch = ckpt["hidden_dim"]                       # 128
    lat_ch = ckpt["latent_dim"]                       # 2
    num_corps = sd["corp_embeddings.weight"].shape[0] # 3589

    encoder = _EncoderCompat(in_ch, hid_ch, lat_ch)
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc_sd, strict=True)
    encoder.eval()

    # ── corp embeddings（トピックの学習済み特徴量）──
    corp_emb = nn.Embedding(num_corps, in_ch)
    corp_emb.weight.data.copy_(sd["corp_embeddings.weight"])

    # ── PotentialNet ──
    prefix = "temporal_predictor.ode_func.potential_net."
    pot_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    pot_net = _PotentialNetCompat(latent_dim=lat_ch, hidden_dim=hid_ch)
    pot_net.load_state_dict(pot_sd, strict=True)
    pot_net.eval()

    scale = float(sd["temporal_predictor.ode_func.scale"].item())
    alpha = float(torch.tanh(torch.tensor(scale)).item())

    print(f"[ckpt] in_ch={in_ch}, hidden={hid_ch}, latent={lat_ch}, "
          f"num_corps={num_corps}, alpha={alpha:.4f}")
    return encoder, corp_emb, pot_net, alpha, in_ch


# ── 2. データ読み込みとグラフ構築 ─────────────────────────────────────────────

def load_data(csv_path: str, viz_year: int):
    print(f"Loading data from {csv_path} ...")
    df = preprocess_author_topic_data(csv_path, year_min=2020, year_max=2026)
    df = filter_active_authors(df, min_papers=5)
    graphs, authors, topics, total_n, hist_edges, input_dim = build_author_topic_graphs(df)
    print(f"  authors={len(authors)}, topics={len(topics)}, total_nodes={total_n}")
    return graphs, authors, topics, total_n, input_dim, df


# ── 3. エンコーダ実行 → 実際の z_i 取得 ──────────────────────────────────────

@torch.no_grad()
def encode_year(encoder, corp_emb, graph, num_authors: int, input_dim: int,
                df, authors):
    """指定年のグラフを encoder に通して著者ノードの μ を返す。"""
    # 著者ノード: 平均 paper_vector（author_to_idx 順）
    author_vecs = calculate_initial_author_vectors(df, num_authors, input_dim, authors)

    # ノード特徴テンソル構築: [著者 | トピック]
    # トピック部分: corp_emb（学習済み、0..num_topics-1 のインデックス）
    num_topics = graph.x.shape[0] - num_authors
    topic_feats = corp_emb.weight[:num_topics]  # (num_topics, input_dim)
    x = torch.cat([author_vecs, topic_feats], dim=0)  # (total_nodes, input_dim)

    mu, _ = encoder(x, graph.edge_index)
    return mu[:num_authors].numpy()  # 著者ノードの μ だけ返す


# ── 4. Φ と ∇Φ のグリッド計算 ────────────────────────────────────────────────

def compute_phi_grid(pot_net, z_real: np.ndarray,
                     margin: float = 1.0, res_phi: int = 80, res_grad: int = 20):
    z1_min, z1_max = z_real[:, 0].min() - margin, z_real[:, 0].max() + margin
    z2_min, z2_max = z_real[:, 1].min() - margin, z_real[:, 1].max() + margin

    xs = np.linspace(z1_min, z1_max, res_phi)
    ys = np.linspace(z2_min, z2_max, res_phi)
    XX, YY = np.meshgrid(xs, ys)
    pts = torch.tensor(np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        phi_vals = pot_net(pts).numpy().reshape(res_phi, res_phi)

    xq = np.linspace(z1_min, z1_max, res_grad)
    yq = np.linspace(z2_min, z2_max, res_grad)
    XQ, YQ = np.meshgrid(xq, yq)
    ptsq = torch.tensor(
        np.stack([XQ.ravel(), YQ.ravel()], axis=1), dtype=torch.float32,
        requires_grad=False)
    ptsq.requires_grad_(True)
    pot_net(ptsq).sum().backward()
    grad = ptsq.grad.detach().numpy().reshape(res_grad, res_grad, 2)

    return XX, YY, phi_vals, XQ, YQ, grad[:, :, 0], grad[:, :, 1]


# ── 5. プロット ────────────────────────────────────────────────────────────────

def make_plot(XX, YY, phi_vals, XQ, YQ, GX, GY,
              z_real: np.ndarray, alpha: float,
              topics_labels=None, z_topics: np.ndarray = None,
              year: int = 2024):

    speed = np.sqrt(GX**2 + GY**2) + 1e-8
    VX_norm = -GX / speed
    VY_norm = -GY / speed

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"P-NODE: Real author embeddings ({year}) + Phi landscape + velocity field\n"
        f"latent_dim=2 (no UMAP needed)   alpha={alpha:.3f}",
        fontsize=11, fontweight="bold",
    )

    for ax_i, ax in enumerate(axes):
        # Φ 等高線
        cf = ax.contourf(XX, YY, phi_vals, levels=30,
                         cmap="viridis" if ax_i == 0 else "plasma", alpha=0.75)
        plt.colorbar(cf, ax=ax, fraction=0.04, label="Phi(z)")

        # ベクトル場
        ax.quiver(XQ, YQ, VX_norm, VY_norm,
                  alpha=0.55, color="white", scale=30, width=0.0025)

        # 実際の著者位置
        sc = ax.scatter(z_real[:, 0], z_real[:, 1],
                        s=8, c="cyan", alpha=0.55, linewidths=0,
                        label=f"authors (n={len(z_real)})", zorder=4)

        # トピック位置（オプション）
        if z_topics is not None and ax_i == 1:
            ax.scatter(z_topics[:, 0], z_topics[:, 1],
                       s=25, c="orange", marker="D", alpha=0.7,
                       linewidths=0.5, edgecolors="black",
                       label="topics", zorder=5)

        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_title("Phi(z) contour + velocity field -alpha*grad_Phi", fontsize=9)
    axes[1].set_title("+ actual author (cyan) & topic (orange) positions", fontsize=9)

    out = "pnode_patent_runner/real_embeddings_vector_field.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out}")
    return out


# ── メイン ────────────────────────────────────────────────────────────────────

def run():
    encoder, corp_emb, pot_net, alpha, input_dim = load_models(CKPT)

    graphs, authors, topics, total_n, in_dim, df = load_data(DATA_CSV, VIZ_YEAR)

    if VIZ_YEAR not in graphs:
        available = sorted(graphs.keys())
        print(f"Year {VIZ_YEAR} not found. Available: {available}")
        use_year = available[-1]
        print(f"Using {use_year} instead.")
    else:
        use_year = VIZ_YEAR

    graph = graphs[use_year]
    num_authors = len(authors)
    num_topics  = len(topics)

    print(f"Encoding year {use_year} ...")
    z_authors = encode_year(encoder, corp_emb, graph, num_authors, input_dim, df, authors)
    print(f"  z_authors shape: {z_authors.shape}")

    # トピックノードも encode
    with torch.no_grad():
        topic_vecs = corp_emb.weight[:num_topics]
        z_full, _ = encoder(
            torch.cat([
                torch.tensor(
                    calculate_initial_author_vectors(df, num_authors, input_dim, authors).numpy()
                ),
                topic_vecs,
            ], dim=0),
            graph.edge_index,
        )
    z_topics = z_full[num_authors:].numpy()

    # サブサンプル（多すぎる場合）
    if len(z_authors) > MAX_NODES:
        idx = np.random.default_rng(42).choice(len(z_authors), MAX_NODES, replace=False)
        z_authors_plot = z_authors[idx]
    else:
        z_authors_plot = z_authors

    print(f"  z_topics shape: {z_topics.shape}")
    print(f"  author z range: z1=[{z_authors[:,0].min():.2f},{z_authors[:,0].max():.2f}], "
          f"z2=[{z_authors[:,1].min():.2f},{z_authors[:,1].max():.2f}]")

    XX, YY, phi_vals, XQ, YQ, GX, GY = compute_phi_grid(
        pot_net, np.vstack([z_authors_plot, z_topics]),
        margin=0.5, res_phi=GRID_RES, res_grad=QUIV_RES
    )

    make_plot(XX, YY, phi_vals, XQ, YQ, GX, GY,
              z_authors_plot, alpha,
              z_topics=z_topics, year=use_year)


if __name__ == "__main__":
    run()
