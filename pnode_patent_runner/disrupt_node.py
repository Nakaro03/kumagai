"""
DisruptNODE: Trend-Conditioned Dual-Force Potential Neural ODE

P-NODE との主な違い:
1. TrendModulatedPotentialNet: Φ(z) = Φ_base(z) に対して
   - 成長トピック → ガウス引力項 (-Σ w⁺_k·G(z,c_k)) を加算 → z が c_k へ流れる
   - 衰退トピック → ガウス斥力項 (+Σ w⁻_k·G(z,c_k)) を加算 → z が c_k から遠ざかる
   G(z,c_k) = exp(-||z-c_k||²/(2σ²))   (σ は学習可能)
2. トレンド信号 (w⁺, w⁻) は各年のトピックごとの論文数増減から算出
3. デコーダも Φ を共有（consistent energy, UnifiedVGAE と同型）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from pnode_patent_runner.models import (
    GradientODEFunc,
    PotentialNet,
    SharedVGAEEncoder,
    integrate_gradient_flow_ode,
)


class TrendModulatedPotentialNet(nn.Module):
    """
    Dual-force potential conditioned on per-topic trend signals.

        Φ(z) = Φ_base(z)
               - Σ_k w⁺_k · G(z, c_k)     # 成長トピック → 引力（谷）
               + Σ_k w⁻_k · G(z, c_k)     # 衰退トピック → 斥力（山）

    G(z, c_k) = exp(-||z - c_k||² / (2σ²))   σ: 学習可能
    w⁺_k = softplus(log_w_plus) * trend_plus_k    (non-negative)
    w⁻_k = softplus(log_w_minus) * trend_minus_k  (non-negative)

    dz/dτ = -∇Φ = -∇Φ_base
                  + Σ_k w⁺_k · G(z,c_k) · (c_k - z)/σ²   (成長方向へ加速)
                  - Σ_k w⁻_k · G(z,c_k) · (c_k - z)/σ²   (衰退方向から減速)

    set_topic_info() で呼び出し元が c_k, trend_plus, trend_minus をセットする。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_topics: int,
        sigma_init: float = 0.5,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_topics = int(num_topics)

        self.log_sigma = nn.Parameter(torch.tensor(math.log(float(sigma_init))))
        self.log_w_plus = nn.Parameter(torch.tensor(0.0))
        self.log_w_minus = nn.Parameter(torch.tensor(0.0))

        self.phi_base = PotentialNet(latent_dim, hidden_dim, feature_mode="mlp")

        self._topic_centers: Optional[torch.Tensor] = None
        self._trend_plus: Optional[torch.Tensor] = None
        self._trend_minus: Optional[torch.Tensor] = None

    def set_topic_info(
        self,
        z_topics: torch.Tensor,
        trend_plus: torch.Tensor,
        trend_minus: torch.Tensor,
    ) -> None:
        """ODE 積分前および損失計算前に呼び出す。"""
        self._topic_centers = z_topics.detach()
        tp_max = trend_plus.max().clamp(min=1e-8)
        tm_max = trend_minus.max().clamp(min=1e-8)
        self._trend_plus = (trend_plus / tp_max).detach().float()
        self._trend_minus = (trend_minus / tm_max).detach().float()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, D) → Φ(z): (N, 1)"""
        phi = self.phi_base(z)

        if self._topic_centers is not None and self._topic_centers.size(0) > 0:
            c = self._topic_centers.to(device=z.device, dtype=z.dtype)
            sigma = self.log_sigma.exp().clamp(min=0.05)

            diff = z.unsqueeze(1) - c.unsqueeze(0)         # (N, K, D)
            sq_dist = (diff ** 2).sum(dim=-1)               # (N, K)
            gauss = torch.exp(-sq_dist / (2.0 * sigma ** 2))  # (N, K)

            if self._trend_plus is not None:
                tp = self._trend_plus.to(device=z.device, dtype=z.dtype)
                w_plus = F.softplus(self.log_w_plus) * tp                     # (K,)
                phi = phi - (w_plus.unsqueeze(0) * gauss).sum(1, keepdim=True)

            if self._trend_minus is not None:
                tm = self._trend_minus.to(device=z.device, dtype=z.dtype)
                w_minus = F.softplus(self.log_w_minus) * tm                   # (K,)
                phi = phi + (w_minus.unsqueeze(0) * gauss).sum(1, keepdim=True)

        return phi


class TrendAwareDualForcePredictor(nn.Module):
    """TrendModulatedPotentialNet を勾配流 ODE で駆動する予測子。"""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_topics: int,
        sigma_init: float = 0.5,
        ode_method: str = "rk4",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        self.potential_net = TrendModulatedPotentialNet(
            latent_dim, hidden_dim, num_topics, sigma_init
        )
        self.ode_func = GradientODEFunc(self.potential_net)
        self.ode_method = str(ode_method)
        self.ode_n_steps = int(ode_n_steps)

    def set_topic_info(
        self,
        z_topics: torch.Tensor,
        trend_plus: torch.Tensor,
        trend_minus: torch.Tensor,
    ) -> None:
        self.potential_net.set_topic_info(z_topics, trend_plus, trend_minus)

    def forward(self, z_current: torch.Tensor, delta_t: float = 1.0) -> torch.Tensor:
        return integrate_gradient_flow_ode(
            self.ode_func,
            z_current,
            float(delta_t),
            method=self.ode_method,
            n_steps=self.ode_n_steps,
        )


class DisruptVGAE(nn.Module):
    """
    DisruptNODE 本体。

    - エンコーダ: SharedVGAEEncoder（既存と同一）
    - テンポラル予測子: TrendAwareDualForcePredictor（新規）
    - デコーダ: 距離スコア + 共有 Φ 項（CoPE-VGAE と同型の consistent energy）

    unified_training.compute_loss_standardized と互換のインターフェース:
      encode / decode / predict_future / temporal_predictor.potential_net

    呼び出し側は各年の forward 前に setup_before_loss(z_t, data_t, device) を
    呼ぶことで topic 情報とトレンド信号をセットする。
    """

    def __init__(
        self,
        num_nodes: int,
        num_corps: int,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        initial_corp_vectors=None,
        w_pot_init: float = 0.05,
        link_score_mode: str = "distance",
        cosine_logit_scale: float = 5.0,
        sigma_init: float = 0.5,
        ode_method: str = "rk4",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_corps = int(num_corps)
        self.num_topics = int(num_nodes) - int(num_corps)
        self.latent_dim = int(latent_dim)
        self.temporal_history_len = 1

        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)

        self.temporal_predictor = TrendAwareDualForcePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_topics=self.num_topics,
            sigma_init=sigma_init,
            ode_method=ode_method,
            ode_n_steps=ode_n_steps,
        )

        self.r = nn.Parameter(torch.tensor(1.0))
        self.w_pot = nn.Parameter(torch.tensor(float(w_pot_init)))

        self._current_trend_plus: Optional[torch.Tensor] = None
        self._current_trend_minus: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup_before_loss(
        self,
        z_t: torch.Tensor,
        data_t: Data,
        device: torch.device,
    ) -> None:
        """
        encode 直後、compute_loss_standardized 呼び出し前にコール。
        topic 位置とトレンド信号を TrendModulatedPotentialNet にセットする。
        """
        z_topics = z_t[self.num_corps :].detach()

        tp = getattr(data_t, "topic_trend_plus", None)
        tm = getattr(data_t, "topic_trend_minus", None)
        if tp is None:
            tp = torch.zeros(self.num_topics, device=device)
            tm = torch.zeros(self.num_topics, device=device)
        else:
            tp = tp.to(device=device, dtype=torch.float32)
            tm = tm.to(device=device, dtype=torch.float32)

        self.temporal_predictor.set_topic_info(z_topics, tp, tm)
        self._current_trend_plus = tp
        self._current_trend_minus = tm

    # ------------------------------------------------------------------
    # VGAE interface
    # ------------------------------------------------------------------

    def get_node_features(self, x: torch.Tensor, node_indices=None) -> torch.Tensor:
        features = x.clone()
        if node_indices is None:
            node_indices = torch.arange(self.num_nodes, device=x.device)
        corp_idx = node_indices[node_indices < self.num_corps]
        if corp_idx.numel() > 0:
            features[corp_idx] = self.corp_embeddings(corp_idx)
        return features

    def encode(
        self, x: torch.Tensor, edge_index: torch.Tensor, node_indices=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_feat = self.get_node_features(x, node_indices)
        mu, logvar = self.encoder(x_feat, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        return mu

    def decode_logits(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            return self.cosine_logit_scale * (z_src * z_dst).sum(dim=1)
        dist_sq = torch.sum((z_src - z_dst) ** 2, dim=1)
        logits = self.r - dist_sq
        # consistent energy: Φ を共有
        pn = self.temporal_predictor.potential_net
        phi_src = pn(z_src).squeeze(-1)
        phi_dst = pn(z_dst).squeeze(-1)
        logits = logits + torch.tanh(self.w_pot) * (phi_src + phi_dst)
        return torch.clamp(logits, -10.0, 10.0)

    def decode(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(z, edge_index))

    def predict_future(
        self,
        z_history_list: List[torch.Tensor],
        year_calendar_start: Optional[int] = None,
    ) -> torch.Tensor:
        z_last = z_history_list[-1]
        z_topics = z_last[self.num_corps :].detach()
        tp = self._current_trend_plus
        tm = self._current_trend_minus
        if tp is None:
            tp = torch.zeros(self.num_topics, device=z_last.device)
            tm = torch.zeros(self.num_topics, device=z_last.device)
        self.temporal_predictor.set_topic_info(z_topics, tp, tm)
        return self.temporal_predictor(z_last)


# ------------------------------------------------------------------
# DisruptNODE 専用の 1 エポック学習
# ------------------------------------------------------------------

def train_one_epoch_disrupt(
    model: DisruptVGAE,
    graphs: Dict[int, Data],
    num_corps: int,
    hist_edges: Set[Tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kw: Dict,
) -> Dict[str, float]:
    """
    unified_training.train_one_epoch に準じているが、
    各年ペアで setup_before_loss() を挟む点が異なる。
    """
    from pnode_patent_runner.unified_training import compute_loss_standardized

    model.train()
    years = sorted(graphs.keys())
    pairs = list(zip(years[:-1], years[1:]))
    agg = {
        k: 0.0
        for k in (
            "total",
            "recon",
            "kl",
            "latent_pred",
            "future_link",
            "potential",
            "trajectory",
            "density_align",
            "attention_ground",
            "grad_phi_l2",
        )
    }
    n = 0
    for y0, y1 in pairs:
        data_t = graphs[y0].to(device)
        data_t1 = graphs[y1].to(device)

        z_t, mu_t, logvar_t = model.encode(data_t.x, data_t.edge_index)
        model.setup_before_loss(z_t, data_t, device)

        optimizer.zero_grad()
        loss, br = compute_loss_standardized(
            model,
            data_t,
            data_t1,
            num_corps,
            [z_t],
            hist_edges,
            precomputed_z_mu_logvar=(z_t, mu_t, logvar_t),
            y0=y0,
            y1=y1,
            **loss_kw,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k in agg:
            agg[k] += br.get(k, 0.0)
        n += 1

    return {k: v / n for k, v in agg.items()} if n > 0 else agg
