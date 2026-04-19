"""
時系列ベンチマーク用 VGAE（Static / RNN+VGAE / NeuralODE / P-NODE）。
CoPE-VGAE 本体は `unified_vgae.UnifiedVGAE` をそのまま利用する。
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.models import (
    GradientNeuralODEPredictor,
    NeuralODEPredictor,
    RNNLatentPredictor,
    SharedVGAEEncoder,
    StaticLatentPredictor,
)
from pnode_patent_runner.time_dependent_potential import GradientNeuralODEPredictorTime

BENCHMARK_VARIANT_KEYS = ("static", "rnn", "neural_ode", "pnode")


class BenchmarkTemporalVGAE(nn.Module):
    """
    - static: 潜在の時間シフトなし
    - rnn: LSTM による系列予測（`temporal_history_len` 年分の履歴）
    - neural_ode: 素の Neural ODE
    - pnode: ポテンシャル勾配流 ODE（デコーダには Φ を入れない = w_pot 相当を幾何項のみ）
    """

    def __init__(
        self,
        num_nodes: int,
        num_corps: int,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 2,
        initial_corp_vectors=None,
        w_pot_init: float = 0.0,
        link_score_mode: str = "cosine",
        cosine_logit_scale: float = 5.0,
        variant: str = "static",
        rnn_history_len: int = 4,
        time_dependent_potential: bool = False,
        year_min: int = 2010,
        year_max: int = 2020,
    ):
        super().__init__()
        if variant not in BENCHMARK_VARIANT_KEYS:
            raise ValueError(f"variant must be one of {BENCHMARK_VARIANT_KEYS}")
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        self.variant = variant
        self.time_dependent_potential = bool(
            time_dependent_potential and variant == "pnode"
        )
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)

        self.temporal_history_len = rnn_history_len if variant == "rnn" else 1

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.r = nn.Parameter(torch.tensor(1.0))
        self.w_pot = nn.Parameter(torch.tensor(float(w_pot_init)))

        if variant == "static":
            self.temporal_predictor = StaticLatentPredictor()
        elif variant == "rnn":
            self.temporal_predictor = RNNLatentPredictor(latent_dim, hidden_dim)
        elif variant == "neural_ode":
            self.temporal_predictor = NeuralODEPredictor(latent_dim, hidden_dim)
        else:
            if self.time_dependent_potential:
                self.temporal_predictor = GradientNeuralODEPredictorTime(
                    latent_dim, hidden_dim, self.year_min, self.year_max
                )
            else:
                self.temporal_predictor = GradientNeuralODEPredictor(latent_dim, hidden_dim)

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

    def decode_logits(self, z, edge_index):
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            cos_theta = (z_src * z_dst).sum(dim=1)
            logits = self.cosine_logit_scale * cos_theta
        else:
            dist_sq = torch.sum((z_src - z_dst) ** 2, dim=1)
            logits = self.r - dist_sq
        return torch.clamp(logits, -10, 10)

    def decode(self, z, edge_index):
        return torch.sigmoid(self.decode_logits(z, edge_index))

    def predict_future(
        self,
        z_history_list: List[torch.Tensor],
        year_calendar_start: Optional[int] = None,
    ) -> torch.Tensor:
        if self.variant == "rnn":
            return self.temporal_predictor(z_history_list)
        z_last = z_history_list[-1]
        if self.variant == "pnode" and self.time_dependent_potential:
            if year_calendar_start is None:
                raise ValueError(
                    "year_calendar_start is required for time-dependent P-NODE"
                )
            return self.temporal_predictor(z_last, int(year_calendar_start))
        return self.temporal_predictor(z_last)
