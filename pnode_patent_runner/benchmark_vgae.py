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
    AdaptiveGateNeuralODEPredictor,
    EvolveGCNOPredictor,
    GradientNeuralODEPredictor,
    GradientNeuralODEPredictorExplicitAttention,
    GradientODEFunc,
    GradientResidualNeuralODEPredictor,
    GravityNeuralODEPredictor,
    HarmonicPotentialPredictor,
    NeuralODEPredictor,
    NeuralODEPredictorTime,
    PopulationCoupledPotentialNet,
    RNNLatentPredictor,
    ROLANDPredictor,
    SharedVGAEEncoder,
    StaticLatentPredictor,
    integrate_gradient_flow_ode,
)
from pnode_patent_runner.time_dependent_potential import GradientNeuralODEPredictorTime

BENCHMARK_VARIANT_KEYS = (
    "static",
    "rnn",
    "neural_ode",
    "pnode",
    "pnode_explicit",
    "pnode_residual",
    "pnode_pc",
    "agnode",
    "harmonic",
    "gravity",
    "evolve_gcn",
    "roland",
)


class _PCPNodePredictor(nn.Module):
    """
    PC-PNODE temporal predictor。PopulationCoupledPotentialNet を
    既存の GradientODEFunc + integrate_gradient_flow_ode で駆動する薄いラッパー。
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        potential_feature_mode: str = "mlp",
        rff_learnable_basis: bool = False,
        w_rho_init: float = 0.1,
        w_delta_init: float = 0.1,
        log_bandwidth_init: float = 0.0,
        max_anchors: int = 512,
        ode_method: str = "dopri5",
        ode_n_steps: int = 4,
    ):
        super().__init__()
        self.potential_net = PopulationCoupledPotentialNet(
            latent_dim, hidden_dim,
            potential_feature_mode=potential_feature_mode,
            rff_learnable_basis=rff_learnable_basis,
            w_rho_init=w_rho_init,
            w_delta_init=w_delta_init,
            log_bandwidth_init=log_bandwidth_init,
            max_anchors=max_anchors,
        )
        self.ode_func = GradientODEFunc(self.potential_net)
        self.ode_method = str(ode_method or "dopri5").lower()
        self.ode_n_steps = int(ode_n_steps)

    def set_population(self, mu_t, mu_prev=None,
                       active_mask_t=None, active_mask_prev=None):
        self.potential_net.set_population(mu_t, mu_prev, active_mask_t, active_mask_prev)

    def forward(self, z_current, delta_t=1.0):
        return integrate_gradient_flow_ode(
            self.ode_func, z_current, float(delta_t),
            method=self.ode_method, n_steps=self.ode_n_steps,
        )


class PNodeHistoryFuseGRU(nn.Module):
    """
    直近 K 年分の潜在を (N, K*D) で受け取り、GRU で要約して ODE 初値 (N, D) を返す。
    """

    def __init__(self, latent_dim: int, history_len: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.history_len = int(history_len)
        self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=1, batch_first=True)

    def forward(self, zs_flat: torch.Tensor) -> torch.Tensor:
        n = zs_flat.size(0)
        x = zs_flat.view(n, self.history_len, self.latent_dim)
        _, h = self.gru(x)
        return h[-1]


class TrendAdapter(nn.Module):
    """
    Architecture A: 二重エンコーダの軽量版 = z 空間→ L_trend 空間への射影。

    入力: z_topics ∈ ℝ^(n_topics × D)   (encoder 出力, detach 済み)
    出力: z_trend  ∈ ℝ^(n_topics × D)   (Φ計算用に再写像)

    変換: z_trend_j = W·z_j + b + e_j
        W: D × D linear projection (回転・スケーリング)
        b: D bias
        e_j: per-topic bias (n_topics × D)

    用途:
        L_trend = MSE_or_ListNet( Φ(adapter(z.detach())), -g )
        Linear と bias は L_trend だけが更新する (encoder は更新しない)
        → encoder は L_future + L_recon に集中、adapter は L_trend に集中
    """
    def __init__(
        self,
        latent_dim: int,
        num_topics: int,
        use_proj: bool = True,
        init_bias_std: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.num_topics = int(num_topics)
        self.use_proj = bool(use_proj)
        if self.use_proj:
            self.proj = nn.Linear(latent_dim, latent_dim)
            # 初期は恒等写像に近い (proj weight = I, bias = 0)
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(latent_dim))
                self.proj.bias.zero_()
        self.bias = nn.Parameter(
            torch.randn(num_topics, latent_dim) * float(init_bias_std)
        )

    def forward(self, z_topics: torch.Tensor) -> torch.Tensor:
        # z_topics: (n_topics_now, D)  ← detach 済みを期待
        n_now = min(z_topics.size(0), self.num_topics)
        out = z_topics
        if self.use_proj:
            out = self.proj(out)
        out = out[:n_now] + self.bias[:n_now]
        if z_topics.size(0) > n_now:
            # padding (通常は発生しない)
            return torch.cat([out, z_topics[n_now:]], dim=0)
        return out


class BenchmarkTemporalVGAE(nn.Module):
    """
    - static: 潜在の時間シフトなし
    - rnn: LSTM による系列予測（`temporal_history_len` 年分の履歴）
    - neural_ode: 素の Neural ODE
    - pnode: ポテンシャル勾配流 ODE（デコーダには Φ を入れない = w_pot 相当を幾何項のみ）
    - pnode_residual: 同上だが ODE に学習可能な残差場 h(z) を加算（非保守成分の近似）
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
        pnode_history_len: int = 1,
        time_dependent_potential: bool = False,
        year_min: int = 2010,
        year_max: int = 2020,
        pnode_ode_method: str = "dopri5",
        pnode_ode_n_steps: int = 4,
        pnode_potential_feature: str = "mlp",
        pnode_rff_frozen_basis: bool = False,
        pnode_density_calibrated: bool = False,
        pnode_density_log_weight: float = 1.0,
        pnode_density_ema_momentum: float = 0.05,
        pnode_hist_fuse_mode: str = "gru",
        neural_ode_hist_fuse_mode: str = "linear",
        pc_w_rho_init: float = 0.1,
        pc_w_delta_init: float = 0.1,
        pc_log_bandwidth_init: float = 0.0,
        topic_position_embedding: bool = False,
        topic_pos_embed_init_std: float = 0.01,
    ):
        super().__init__()
        if variant not in BENCHMARK_VARIANT_KEYS:
            raise ValueError(f"variant must be one of {BENCHMARK_VARIANT_KEYS}")
        if variant == "pnode_residual" and time_dependent_potential:
            raise ValueError("pnode_residual は time_dependent_potential と併用未対応")
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        self.variant = variant
        self.time_dependent_potential = bool(
            time_dependent_potential and variant == "pnode"
        )
        self.time_dependent_neural_ode = bool(
            time_dependent_potential and variant == "neural_ode"
        )
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)

        if variant == "rnn":
            self.temporal_history_len = int(rnn_history_len)
        elif variant in (
            "pnode",
            "pnode_explicit",
            "pnode_residual",
            "pnode_pc",
            "agnode",
            "neural_ode",
        ):
            self.temporal_history_len = max(1, int(pnode_history_len))
        else:
            self.temporal_history_len = 1

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.r = nn.Parameter(torch.tensor(1.0))
        self.w_pot = nn.Parameter(torch.tensor(float(w_pot_init)))
        # Architecture A: TrendAdapter (二重エンコーダの軽量版)
        #   z は L_recon + L_future で最適化 (元の役割)
        #   adapter(z.detach()) は L_trend 専用空間に z_topic を再写像
        #   adapter = Linear(latent_dim, latent_dim) + per-topic bias
        #   このため L_trend は encoder ではなく adapter のみを更新できる
        self.use_trend_adapter = bool(topic_position_embedding)
        if self.use_trend_adapter:
            n_topics = num_nodes - num_corps
            self.trend_adapter = TrendAdapter(
                latent_dim=latent_dim,
                num_topics=n_topics,
                use_proj=True,
                init_bias_std=float(topic_pos_embed_init_std),
            )
        else:
            self.trend_adapter = None
        self.pnode_hist_fuse_mode = str(pnode_hist_fuse_mode).lower()
        if self.pnode_hist_fuse_mode not in ("linear", "gru"):
            raise ValueError("pnode_hist_fuse_mode must be 'linear' or 'gru'")
        self.pnode_potential_feature = str(pnode_potential_feature).lower()
        self.pnode_rff_frozen_basis = bool(pnode_rff_frozen_basis)
        self.pnode_hist_fuse: Optional[nn.Module]
        if variant in ("pnode", "pnode_explicit", "pnode_residual", "pnode_pc", "agnode") and self.temporal_history_len > 1:
            if self.pnode_hist_fuse_mode == "gru":
                self.pnode_hist_fuse = PNodeHistoryFuseGRU(
                    latent_dim, self.temporal_history_len
                )
            else:
                self.pnode_hist_fuse = nn.Linear(
                    latent_dim * self.temporal_history_len,
                    latent_dim,
                )
        else:
            self.pnode_hist_fuse = None

        self.neural_ode_hist_fuse_mode = str(neural_ode_hist_fuse_mode).lower()
        if self.neural_ode_hist_fuse_mode not in ("linear", "gru"):
            raise ValueError("neural_ode_hist_fuse_mode must be 'linear' or 'gru'")
        self.neural_ode_hist_fuse: Optional[nn.Module]
        if variant == "neural_ode" and self.temporal_history_len > 1:
            if self.neural_ode_hist_fuse_mode == "gru":
                self.neural_ode_hist_fuse = PNodeHistoryFuseGRU(
                    latent_dim, self.temporal_history_len
                )
            else:
                self.neural_ode_hist_fuse = nn.Linear(
                    latent_dim * self.temporal_history_len,
                    latent_dim,
                )
        else:
            self.neural_ode_hist_fuse = None

        if variant == "evolve_gcn":
            self.temporal_predictor = EvolveGCNOPredictor(latent_dim, hidden_dim)
            self.temporal_history_len = int(rnn_history_len)
        elif variant == "roland":
            self.temporal_predictor = ROLANDPredictor(latent_dim, hidden_dim)
        elif variant == "agnode":
            self.temporal_predictor = AdaptiveGateNeuralODEPredictor(
                latent_dim,
                hidden_dim,
                ode_method=pnode_ode_method,
                ode_n_steps=pnode_ode_n_steps,
                feature_mode=str(pnode_potential_feature),
                rff_learnable_basis=not bool(pnode_rff_frozen_basis),
            )
        elif variant == "gravity":
            num_topics = num_nodes - num_corps
            self.temporal_predictor = GravityNeuralODEPredictor(
                num_topics=num_topics,
                ode_method=pnode_ode_method,
                ode_n_steps=pnode_ode_n_steps,
            )
        elif variant == "harmonic":
            num_topics = num_nodes - num_corps
            self.temporal_predictor = HarmonicPotentialPredictor(
                num_corps=num_corps,
                num_topics=num_topics,
                latent_dim=latent_dim,
            )
        elif variant == "static":
            self.temporal_predictor = StaticLatentPredictor()
        elif variant == "rnn":
            self.temporal_predictor = RNNLatentPredictor(latent_dim, hidden_dim)
        elif variant == "neural_ode":
            if self.time_dependent_neural_ode:
                self.temporal_predictor = NeuralODEPredictorTime(
                    latent_dim, hidden_dim, self.year_min, self.year_max
                )
            else:
                self.temporal_predictor = NeuralODEPredictor(latent_dim, hidden_dim)
        elif variant == "pnode_pc":
            self.temporal_predictor = _PCPNodePredictor(
                latent_dim,
                hidden_dim,
                potential_feature_mode=self.pnode_potential_feature,
                rff_learnable_basis=not self.pnode_rff_frozen_basis,
                w_rho_init=float(pc_w_rho_init),
                w_delta_init=float(pc_w_delta_init),
                log_bandwidth_init=float(pc_log_bandwidth_init),
                ode_method=pnode_ode_method,
                ode_n_steps=pnode_ode_n_steps,
            )
        elif variant == "pnode_explicit":
            self.temporal_predictor = GradientNeuralODEPredictorExplicitAttention(
                latent_dim,
                hidden_dim,
                ode_method=pnode_ode_method,
                ode_n_steps=pnode_ode_n_steps,
                feature_mode=str(pnode_potential_feature),
                rff_learnable_basis=not bool(pnode_rff_frozen_basis),
            )
        else:
            if self.time_dependent_potential:
                self.temporal_predictor = GradientNeuralODEPredictorTime(
                    latent_dim,
                    hidden_dim,
                    self.year_min,
                    self.year_max,
                    ode_method=pnode_ode_method,
                    ode_n_steps=pnode_ode_n_steps,
                )
            elif variant == "pnode_residual":
                self.temporal_predictor = GradientResidualNeuralODEPredictor(
                    latent_dim,
                    hidden_dim,
                    density_calibrated=bool(pnode_density_calibrated),
                    density_log_weight=float(pnode_density_log_weight),
                    density_momentum=float(pnode_density_ema_momentum),
                    potential_feature_mode=self.pnode_potential_feature,
                    rff_learnable_basis=not self.pnode_rff_frozen_basis,
                    ode_method=pnode_ode_method,
                    ode_n_steps=pnode_ode_n_steps,
                )
            else:
                self.temporal_predictor = GradientNeuralODEPredictor(
                    latent_dim,
                    hidden_dim,
                    density_calibrated=bool(pnode_density_calibrated),
                    density_log_weight=float(pnode_density_log_weight),
                    density_momentum=float(pnode_density_ema_momentum),
                    potential_feature_mode=self.pnode_potential_feature,
                    rff_learnable_basis=not self.pnode_rff_frozen_basis,
                    ode_method=pnode_ode_method,
                    ode_n_steps=pnode_ode_n_steps,
                )

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
        # 注意: トピック位置埋め込みはここでは適用しない (L_future に汚染される)。
        # apply_trend_adapter() メソッドで L_trend 専用に detach+adapter を適用する。
        return z, mu, logvar

    def apply_trend_adapter(self, z_topics_raw: torch.Tensor) -> torch.Tensor:
        """
        L_trend 専用の z_topic 変換 (Architecture A)。
        - z_topics_raw を detach() してから adapter を適用することで、
          L_trend の勾配は encoder ではなく adapter (proj + bias) のみを更新する。
        - encoder は L_recon + L_future のみで最適化される。
        - adapter は L_trend のみで最適化されるため、
          z が L_future に支配されていても adapter が空間を再形成できる。
        """
        if not self.use_trend_adapter or self.trend_adapter is None:
            return z_topics_raw
        z_detached = z_topics_raw.detach()
        return self.trend_adapter(z_detached)

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
        # Dual-Force側の勾配漏れ修正（dual_force_vgae.py, 2026-08-22）と対称の修正。
        # ここで detach しないと、時間発展モジュール（static は恒等写像、つまりパラメータ
        # ゼロ）を経由した future_link_loss/latent_pred_loss の勾配がエンコーダまで直接
        # 伝播し、手法間の比較がエンコーダの表現力の差にすり替わる
        # （DUAL_FORCE_REDESIGN.md §6.7: rank_renorm が static と無差別だった一因）。
        z_history_list = [z.detach() for z in z_history_list]
        if self.variant == "gravity":
            z_all     = z_history_list[-1]
            z_authors = z_all[: self.num_corps]
            z_topics  = z_all[self.num_corps :]
            self.temporal_predictor.set_topic_positions(z_topics)
            z_authors_next = self.temporal_predictor(z_authors)
            return torch.cat([z_authors_next, z_topics], dim=0)
        if self.variant == "harmonic":
            return self.temporal_predictor(z_history_list[-1])
        if self.variant in ("rnn", "evolve_gcn"):
            k = self.temporal_history_len
            zs = list(z_history_list)
            while len(zs) < k:
                zs = [zs[0]] + zs
            return self.temporal_predictor(zs[-k:])
        z_last = z_history_list[-1]
        if self.variant in (
            "pnode",
            "pnode_explicit",
            "pnode_residual",
            "pnode_pc",
        ) and self.pnode_hist_fuse is not None:
            k = self.temporal_history_len
            zs = list(z_history_list)
            while len(zs) < k:
                zs = [zs[0]] + zs
            zs = zs[-k:]
            z_last = self.pnode_hist_fuse(torch.cat(zs, dim=1))
        elif self.variant == "neural_ode" and self.neural_ode_hist_fuse is not None:
            k = self.temporal_history_len
            zs = list(z_history_list)
            while len(zs) < k:
                zs = [zs[0]] + zs
            zs = zs[-k:]
            z_last = self.neural_ode_hist_fuse(torch.cat(zs, dim=1))
        if self.variant == "pnode" and self.time_dependent_potential:
            if year_calendar_start is None:
                raise ValueError(
                    "year_calendar_start is required for time-dependent P-NODE"
                )
            return self.temporal_predictor(z_last, int(year_calendar_start))
        if self.variant == "neural_ode" and self.time_dependent_neural_ode:
            if year_calendar_start is None:
                raise ValueError(
                    "year_calendar_start is required for time-dependent Neural ODE"
                )
            return self.temporal_predictor(z_last, int(year_calendar_start))
        return self.temporal_predictor(z_last)
