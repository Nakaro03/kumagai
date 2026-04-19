"""
P-NODE の時間依存ポテンシャル版で、力学とリンク尤度を一つの Φ(z,y) で整合させる。

数学的定式化（実装と対応）:

1) **単粒子ポテンシャル** Φ: ℝ^L × 𝒴 → ℝ（𝒴 は暦年の有限集合、`TimeDependentPotentialNet`）。

2) **潜在の時間発展（勾配流）**  
   出発年 y₀ に対し、擬時間 τ ∈ [0,1] で  
   dz/dτ = -∇_z Φ(z, y₀)  
   （摩擦なし・単位質量。ODE は `GradientNeuralODEPredictorEnergy` = 純勾配流。）

3) **二部リンクのボルツマン型エネルギー（Bernoulli ロジットと整合）**  
   エッジ (i,j) のペアエネルギーを  
   U_ij(y) = d(z_i, z_j) - λ · (Φ(z_i,y) + Φ(z_j,y))  
   とし、逆温度 β > 0 に対し  
   P(y_ij=1 | z, y) = σ(-β · U_ij(y))  
   すなわち logit = -β · U_ij。  
   距離項は d = ‖z_i - z_j‖²（`link_score_mode=distance`）、  
   または d = 2(1 - cos∠(z_i,z_j))（`cosine`、幾何的に非負）。

`BenchmarkTemporalVGAE` の P-NODE 分岐をベースに、デコーダに Φ を明示的に入れた変種。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.models import SharedVGAEEncoder
from pnode_patent_runner.time_dependent_potential import GradientNeuralODEPredictorEnergy

METHOD_SHORT_NAME = "P-NODE-Energy-TD"


class PNodeEnergyTD(nn.Module):
    """
    時間依存 Φ とボルツマン型リンク尤度・純勾配流 ODE を共有する P-NODE 拡張。
    """

    decode_requires_calendar_year: bool = True
    time_dependent_potential: bool = True
    pnode_energy_td: bool = True
    temporal_history_len: int = 1

    def __init__(
        self,
        num_nodes: int,
        num_corps: int,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 2,
        initial_corp_vectors=None,
        link_score_mode: str = "distance",
        cosine_logit_scale: float = 5.0,
        year_min: int = 2010,
        year_max: int = 2020,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)
        self.year_min = int(year_min)
        self.year_max = int(year_max)

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.temporal_predictor = GradientNeuralODEPredictorEnergy(
            latent_dim, hidden_dim, self.year_min, self.year_max
        )

        # U_ij = geom - λ (φ_i+φ_j), logit = -β U_ij  （β, λ > 0 を softplus でパラメータ化）
        self._log_boltzmann_beta = nn.Parameter(torch.tensor(0.541324855))  # softplus -> ~1
        self._log_pair_lambda = nn.Parameter(torch.tensor(-2.3025850929940455))  # softplus -> ~0.1

    def _pair_geom(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            cos_theta = (z_src * z_dst).sum(dim=1)
            return 2.0 * (1.0 - cos_theta)
        dist_sq = torch.sum((z_src - z_dst) ** 2, dim=1)
        return dist_sq

    def decode_logits(self, z: torch.Tensor, edge_index: torch.Tensor, calendar_year: int):
        """logit = -β · U_ij(y),  U_ij = d_ij - λ (φ_i+φ_j)。"""
        pn = self.temporal_predictor.potential_net
        yi = pn.year_tensor(int(calendar_year), z.size(0), z.device)
        phi = pn(z, yi)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)
        phi_e = phi[edge_index[0]] + phi[edge_index[1]]
        d_ij = self._pair_geom(z, edge_index)
        lam = F.softplus(self._log_pair_lambda)
        beta = F.softplus(self._log_boltzmann_beta)
        u_ij = d_ij - lam * phi_e
        logits = -beta * u_ij
        return torch.clamp(logits, -10, 10)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor, calendar_year: int):
        return torch.sigmoid(self.decode_logits(z, edge_index, calendar_year))

    def get_node_features(self, x, node_indices=None):
        features = x.clone()
        if node_indices is None:
            node_indices = torch.arange(self.num_nodes, device=x.device)
        corp_idx = node_indices[node_indices < self.num_corps]
        if corp_idx.numel() > 0:
            features[corp_idx] = self.corp_embeddings(corp_idx)
        return features

    def encode(self, x, edge_index, node_indices=None):
        x_features = self.get_node_features(x, node_indices)
        mu, logvar = self.encoder(x_features, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        return mu

    def predict_future(
        self,
        z_history_list: List[torch.Tensor],
        year_calendar_start: Optional[int] = None,
    ) -> torch.Tensor:
        if year_calendar_start is None:
            raise ValueError("year_calendar_start is required for PNodeEnergyTD")
        z_last = z_history_list[-1]
        return self.temporal_predictor(z_last, int(year_calendar_start))
