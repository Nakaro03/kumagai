"""UnifiedVGAE（表記上 CoPE-VGAE）。"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.models import GradientNeuralODEPredictor, SharedVGAEEncoder

METHOD_SHORT_NAME = "CoPE-VGAE"
METHOD_FULL_NAME_EN = "Consistent Potential-Energy Variational Graph Autoencoder"
METHOD_FULL_NAME_JA = "一貫ポテンシャルエネルギー変分グラフオートエンコーダ"


class UnifiedVGAE(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_corps,
        input_dim,
        hidden_dim=256,
        latent_dim=2,
        sequence_length=3,
        initial_corp_vectors=None,
        w_pot_init=None,
        link_score_mode="distance",
        cosine_logit_scale=5.0,
        density_calibrated_potential: bool = False,
        density_log_weight: float = 1.0,
        density_ema_momentum: float = 0.05,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_corps
        self.latent_dim = latent_dim
        if link_score_mode not in ("distance", "cosine"):
            raise ValueError("link_score_mode must be 'distance' or 'cosine'")
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)

        self.corp_embeddings = nn.Embedding(num_corps, input_dim)
        if initial_corp_vectors is not None:
            self.corp_embeddings.weight.data.copy_(initial_corp_vectors)
        else:
            nn.init.normal_(self.corp_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.temporal_predictor = GradientNeuralODEPredictor(
            latent_dim,
            hidden_dim,
            density_calibrated=density_calibrated_potential,
            density_log_weight=density_log_weight,
            density_momentum=density_ema_momentum,
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

    def decode_logits(self, z, edge_index):
        phi = self.temporal_predictor.potential_net(z)
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

    def decode(self, z, edge_index):
        return torch.sigmoid(self.decode_logits(z, edge_index))

    def predict_future(self, z_history_list, year_calendar_start=None):
        del year_calendar_start
        return self.temporal_predictor(z_history_list[-1])
