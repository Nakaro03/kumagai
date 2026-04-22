import torch
import torch.nn as nn
import torch.nn.functional as F
from pnode_patent_runner.models import SharedVGAEEncoder
from pnode_patent_runner.dual_force_models import DualForcePNODEPredictor

class DualForceVGAE(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_authors,
        input_dim,
        hidden_dim=256,
        latent_dim=2,
        initial_author_vectors=None,
        gamma=1.0,
        ode_method="dopri5",
        ode_n_steps=4,
        link_score_mode="cosine",
        cosine_logit_scale=5.0
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_corps = num_authors # num_corpsとして扱う（既存コードとの互換性）
        self.latent_dim = latent_dim
        self.link_score_mode = link_score_mode
        self.cosine_logit_scale = float(cosine_logit_scale)

        self.author_embeddings = nn.Embedding(num_authors, input_dim)
        if initial_author_vectors is not None:
            self.author_embeddings.weight.data.copy_(initial_author_vectors)
        else:
            nn.init.normal_(self.author_embeddings.weight, mean=0.0, std=0.05)

        self.encoder = SharedVGAEEncoder(input_dim, hidden_dim, latent_dim)
        self.temporal_predictor = DualForcePNODEPredictor(
            latent_dim, hidden_dim, gamma=gamma, 
            ode_method=ode_method, ode_n_steps=ode_n_steps
        )

    def get_node_features(self, x, node_indices=None):
        features = x.clone()
        if node_indices is None:
            node_indices = torch.arange(self.num_nodes, device=x.device)
        author_idx = node_indices[node_indices < self.num_corps]
        if author_idx.numel() > 0:
            features[author_idx] = self.author_embeddings(author_idx)
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
        # Dual-Force P-NODEではポテンシャル項をデコードに直接入れない設計とする（ベクトル場のみに影響）
        z_src = z[edge_index[0]]
        z_dst = z[edge_index[1]]
        if self.link_score_mode == "cosine":
            z_src = F.normalize(z_src, p=2, dim=1, eps=1e-8)
            z_dst = F.normalize(z_dst, p=2, dim=1, eps=1e-8)
            cos_theta = (z_src * z_dst).sum(dim=1)
            logits = self.cosine_logit_scale * cos_theta
        else:
            dist_sq = torch.sum((z_src - z_dst) ** 2, dim=1)
            logits = 1.0 - dist_sq # r=1.0固定
        return torch.clamp(logits, -10, 10)

    def decode(self, z, edge_index):
        return torch.sigmoid(self.decode_logits(z, edge_index))

    def predict_future(self, z_history_list, data_t=None):
        """
        data_t: 現在の年のDataオブジェクト。トピック情報を含む。
        """
        if data_t is not None:
            # トピック情報をODEベクトル場にセット
            num_authors = self.num_corps
            # トピックの埋め込み（ノード特徴の後半部分）
            # ここでは簡単のため、data_t.xから直接取得する
            P_j = data_t.x[num_authors:] 
            # もしエンコード後のトピック位置を使う場合は以下：
            # z_t, _, _ = self.encode(data_t.x, data_t.edge_index)
            # P_j = z_t[num_authors:]
            
            # トピックの勢い情報をセット
            self.temporal_predictor.ode_func.set_topic_info(
                P_j.to(z_history_list[-1].device),
                data_t.topic_trend_plus.unsqueeze(-1).to(z_history_list[-1].device),
                data_t.topic_trend_minus.unsqueeze(-1).to(z_history_list[-1].device)
            )
            
        return self.temporal_predictor(z_history_list[-1])
