"""
P-NODE の時間依存ポテンシャル版で、力学とリンク尤度を一つの Φ(z,t) で整合させる。

【v2 変更点】
  - PNodeEnergyTDContinuous を追加。
    GradientNeuralODEPredictorContinuous を使い、ODE 積分中に t を連続的に進める。
    （既存の PNodeEnergyTD は後方互換のため残置）
  - predict_future シグネチャを統一: (z_history_list, year_calendar_start)

数学的定式化:

1) ポテンシャル Φ: ℝ^L × ℝ → ℝ（連続実時刻版 ContinuousTimePotentialNet）。

2) 潜在の時間発展（連続時間勾配流）
   state = (z, t_cont),  τ ∈ [0, 1]
   d(z, t_cont)/dτ = (-tanh(s)·∇_z Φ(z, t_cont),  1)
   → τ が進むにつれ t_cont も 1 増加し、Φ の地形が連続的に変化。

3) ボルツマン型リンクエネルギー（既存と同形）
   logit(i,j,t) = -β · (d_ij - λ·(Φ(z_i,t) + Φ(z_j,t)))
   β, λ > 0 を softplus でパラメータ化（正値保証）。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.models import SharedVGAEEncoder
from pnode_patent_runner.time_dependent_potential import (
    GradientNeuralODEPredictorEnergy,  # 旧実装（後方互換）
    TimeDependentPotentialNet,
)
from pnode_patent_runner.time_dependent_potential import (
    GradientNeuralODEPredictorContinuous,
    ContinuousTimePotentialNet,
)

METHOD_SHORT_NAME = "P-NODE-Energy-TD"
METHOD_SHORT_NAME_CONTINUOUS = "P-NODE-Energy-TD-Cont"


# ---------------------------------------------------------------------------
# 既存モデル（後方互換）
# ---------------------------------------------------------------------------

class PNodeEnergyTD(nn.Module):
    """
    時間依存 Φ(z, year_idx) とボルツマン型リンク尤度・純勾配流 ODE。
    ODE 積分中は年を固定（旧実装）。後方互換のため残置。
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

        self._log_boltzmann_beta = nn.Parameter(torch.tensor(0.541324855))
        self._log_pair_lambda = nn.Parameter(torch.tensor(-2.3025850929940455))

    def _pair_geom(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            return 2.0 * (1.0 - (z_src * z_dst).sum(dim=1))
        return torch.sum((z_src - z_dst) ** 2, dim=1)

    def decode_logits(self, z: torch.Tensor, edge_index: torch.Tensor, calendar_year: int):
        pn = self.temporal_predictor.potential_net
        yi = pn.year_tensor(int(calendar_year), z.size(0), z.device)
        phi = pn(z, yi)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)
        phi_e = phi[edge_index[0]] + phi[edge_index[1]]
        d_ij = self._pair_geom(z, edge_index)
        lam = F.softplus(self._log_pair_lambda)
        beta = F.softplus(self._log_boltzmann_beta)
        return torch.clamp(-beta * (d_ij - lam * phi_e), -10, 10)

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
            raise ValueError("year_calendar_start is required")
        return self.temporal_predictor(z_history_list[-1], int(year_calendar_start))


# ---------------------------------------------------------------------------
# 【新規】PNodeEnergyTDContinuous
# ---------------------------------------------------------------------------

class PNodeEnergyTDContinuous(nn.Module):
    """
    連続時間 ODE で Φ(z, t) を積分する P-NODE 変種。

    【既存 PNodeEnergyTD との違い】
        ODE 積分中に t_cont が 1 ずつ連続的に増加する。
        Φ は ContinuousTimePotentialNet（Fourier 時刻埋め込み）を使用。
        デコーダでは calendar_year を float 変換して Φ(z, t) を評価。

    【シグネチャ互換性】
        predict_future(z_history_list, year_calendar_start) — 既存訓練ループと同じ。
        decode / decode_logits(z, edge_index, calendar_year) — 同じ。
    """

    decode_requires_calendar_year: bool = True
    time_dependent_potential: bool = True
    pnode_energy_td_continuous: bool = True   # 識別フラグ
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
        year_min: int = 2010,
        year_max: int = 2020,
        time_fourier_K: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.year_min = int(year_min)
        self.year_max = int(year_max)

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)

        # 連続時間 ODE 予測子（Φ も ContinuousTimePotentialNet）
        self.temporal_predictor = GradientNeuralODEPredictorContinuous(
            latent_dim, hidden_dim, year_min, year_max, time_fourier_K
        )

        # ボルツマン型パラメータ（softplus で正値保証）
        self._log_boltzmann_beta = nn.Parameter(torch.tensor(0.541324855))
        self._log_pair_lambda = nn.Parameter(torch.tensor(-2.3025850929940455))

    @property
    def potential_net(self) -> ContinuousTimePotentialNet:
        return self.temporal_predictor.potential_net

    def _pair_geom(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            return 2.0 * (1.0 - (z_src * z_dst).sum(dim=1))
        return torch.sum((z_src - z_dst) ** 2, dim=1)

    def decode_logits(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        calendar_year: int,
    ) -> torch.Tensor:
        """
        logit = -β · (d_ij - λ·(Φ(z_i,t) + Φ(z_j,t)))
        t = calendar_year - year_min （連続値として渡す）
        """
        pn = self.potential_net
        t_offset = float(calendar_year) - float(pn.year_min)
        t_scalar = torch.tensor(t_offset, dtype=torch.float32, device=z.device)

        phi = pn(z, t_scalar)                          # (N, 1)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)                      # (N,)

        phi_e = phi[edge_index[0]] + phi[edge_index[1]]
        d_ij = self._pair_geom(z, edge_index)

        lam = F.softplus(self._log_pair_lambda)
        beta = F.softplus(self._log_boltzmann_beta)
        return torch.clamp(-beta * (d_ij - lam * phi_e), -10, 10)

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        calendar_year: int,
    ) -> torch.Tensor:
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
        """
        既存訓練ループ（compute_loss_standardized_td 等）と同じシグネチャ。
        """
        if year_calendar_start is None:
            raise ValueError("year_calendar_start is required for PNodeEnergyTDContinuous")
        return self.temporal_predictor(
            z_history_list[-1], year_start=int(year_calendar_start)
        )


# ---------------------------------------------------------------------------
# 【新規②】PNodeEnergyTDBasis — 基底分解型 Φ(z,t) を使う PNODE
# ---------------------------------------------------------------------------

class PNodeEnergyTDBasis(nn.Module):
    """
    基底分解型ポテンシャル Φ(z,t) = α(t)ᵀ Φ_basis(z) を使う P-NODE 変種。

    【PNodeEnergyTDContinuous との違い】
        - Φ が BasisDecomposedPotentialNet（基底分解）
        - α(t) を get_alpha() で取得 → Figure 1 の「年別基底重みの推移」に直結
        - 基底数 num_basis はハイパーパラメータ（デフォルト 8）

    【訓練ループとの互換性】
        predict_future / decode / decode_logits のシグネチャは
        PNodeEnergyTD / PNodeEnergyTDContinuous と同一。
        compute_loss_standardized_td をそのまま使用可能。

    【論文上の位置づけ】
        アブレーション表の「Basis-K」列として掲載。
        K=1 は ContinuousTimePotentialNet に近い退化ケース、
        K=8 が提案の標準設定。
    """

    decode_requires_calendar_year: bool = True
    time_dependent_potential: bool = True
    pnode_energy_td_basis: bool = True        # 識別フラグ
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
        year_min: int = 2010,
        year_max: int = 2020,
        num_basis: int = 8,
        time_fourier_K: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        self.num_basis = int(num_basis)

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)

        # 基底分解型予測子をインポート
        from pnode_patent_runner.time_dependent_potential import (
            GradientNeuralODEPredictorBasis,
        )
        self.temporal_predictor = GradientNeuralODEPredictorBasis(
            latent_dim, hidden_dim, year_min, year_max, num_basis, time_fourier_K
        )

        # ボルツマン型パラメータ
        self._log_boltzmann_beta = nn.Parameter(torch.tensor(0.541324855))
        self._log_pair_lambda = nn.Parameter(torch.tensor(-2.3025850929940455))

    @property
    def potential_net(self):
        return self.temporal_predictor.potential_net

    def _pair_geom(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            return 2.0 * (1.0 - (z_src * z_dst).sum(dim=1))
        return torch.sum((z_src - z_dst) ** 2, dim=1)

    def decode_logits(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        calendar_year: int,
    ) -> torch.Tensor:
        """
        logit = -β · (d_ij - λ·(Φ(z_i,t) + Φ(z_j,t)))
        Φ は基底分解型: α(t)ᵀ Φ_basis(z)
        """
        pn = self.potential_net
        t_offset = float(calendar_year) - float(pn.year_min)
        t_scalar = torch.tensor(t_offset, dtype=torch.float32, device=z.device)

        phi = pn(z, t_scalar)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)

        phi_e = phi[edge_index[0]] + phi[edge_index[1]]
        d_ij = self._pair_geom(z, edge_index)

        lam = F.softplus(self._log_pair_lambda)
        beta = F.softplus(self._log_boltzmann_beta)
        return torch.clamp(-beta * (d_ij - lam * phi_e), -10, 10)

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        calendar_year: int,
    ) -> torch.Tensor:
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
        z_history_list,
        year_calendar_start=None,
    ) -> torch.Tensor:
        if year_calendar_start is None:
            raise ValueError("year_calendar_start is required for PNodeEnergyTDBasis")
        return self.temporal_predictor(
            z_history_list[-1], year_start=int(year_calendar_start)
        )

    def get_basis_weights_over_time(self, year_range, device: str = "cpu"):
        """
        year_range: list of calendar years (e.g. range(2010, 2021))
        返り値: shape (len(years), K) の numpy 配列
        → 論文 Figure 1 下段「基底重み α(t) の時系列」プロット用
        """
        import numpy as np
        alphas = []
        for y in year_range:
            alpha = self.potential_net.get_alpha(float(y), device=device)
            alphas.append(alpha.numpy())
        return __import__("numpy").stack(alphas, axis=0)
