"""CoPE 型 VGAE で時間依存ポテンシャル Φ(z, year) を用いる変種（UnifiedVGAETD）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.models import SharedVGAEEncoder
from pnode_patent_runner.time_dependent_potential import GradientNeuralODEPredictorTime

METHOD_SHORT_NAME_TD = "CoPE-VGAE-TD"
METHOD_FULL_NAME_EN_TD = "CoPE-VGAE with time-dependent potential Phi(z, year)"


class UnifiedVGAETD(nn.Module):
    """
    ``decode`` / ``decode_logits`` に **calendar_year** が必要（その年の Φ を使用）。
    ``predict_future(z_list, year_start)`` は出発年の Φ で ODE。
    """

    def __init__(
        self,
        num_nodes: int,
        num_corps: int,
        input_dim: int,
        year_min: int,
        year_max: int,
        hidden_dim: int = 256,
        latent_dim: int = 2,
        initial_corp_vectors=None,
        w_pot_init=None,
        link_score_mode: str = "distance",
        cosine_logit_scale: float = 5.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        self.year_min = int(year_min)
        self.year_max = int(year_max)
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)
        self.temporal_history_len = 1

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.temporal_predictor = GradientNeuralODEPredictorTime(
            latent_dim, hidden_dim, self.year_min, self.year_max
        )

        self.r = nn.Parameter(torch.tensor(1.0))
        _wp = 0.0 if w_pot_init is None else float(w_pot_init)
        self.w_pot = nn.Parameter(torch.tensor(_wp))

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

    def decode_logits(self, z, edge_index, calendar_year: int):
        pn = self.temporal_predictor.potential_net
        yi = pn.year_tensor(calendar_year, z.size(0), z.device)
        phi = pn(z, yi)
        if phi.dim() > 1:
            phi = phi.squeeze(-1)
        pe = phi[edge_index[0]] + phi[edge_index[1]]
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            cos_theta = (z_src * z_dst).sum(dim=1)
            logits = self.cosine_logit_scale * cos_theta + self.w_pot * pe
        else:
            dist_sq = torch.sum((z_src - z_dst) ** 2, dim=1)
            logits = self.r - dist_sq + self.w_pot * pe
        return torch.clamp(logits, -10, 10)

    def decode(self, z, edge_index, calendar_year: int):
        return torch.sigmoid(self.decode_logits(z, edge_index, calendar_year))

    def predict_future(self, z_history_list, year_calendar_start: int):
        return self.temporal_predictor(z_history_list[-1], year_calendar_start)
