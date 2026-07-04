import torch
import torch.nn as nn
import torch.nn.functional as F
from pnode_patent_runner.models import GradientODEFunc, integrate_gradient_flow_ode

class DualForcePotentialODEFunc(nn.Module):
    """
    Dual-Force P-NODE (Dual-Force Potential Neural ODE) のベクトル場定義。
    
    dz/dt = sum_j alpha_{ij}^+ (P_j - H_i(t)) - gamma * sum_j alpha_{ij}^- (P_j - H_i(t))
    """
    def __init__(self, latent_dim, hidden_dim, gamma=1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        
        # アテンション用の重み行列
        self.W_Q = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(latent_dim + 1, hidden_dim, bias=False) # [P_j || D_j]
        self.P_proj = None # Will be initialized on first forward
        
        # トピックの位置 P_j と 勢い D_j^+, D_j^- を保持するためのプレースホルダ
        # 実行時に set_topic_info で設定される
        self.register_buffer("P_j", torch.zeros(0, latent_dim))
        self.register_buffer("D_plus_j", torch.zeros(0, 1))
        self.register_buffer("D_minus_j", torch.zeros(0, 1))

    def set_topic_info(self, P_j, D_plus_j, D_minus_j):
        self.P_j = P_j
        self.D_plus_j = D_plus_j
        self.D_minus_j = D_minus_j

    def forward(self, t, h_i):
        """
        h_i: (num_authors, latent_dim)
        """
        if self.P_j.numel() == 0:
            return torch.zeros_like(h_i)
            
        num_authors = h_i.size(0)
        num_topics = self.P_j.size(0)
        
        # Query: 著者の現在地
        Q = self.W_Q(h_i) # (num_authors, hidden_dim)
        
        # Dynamic adjustment of P_proj if input_dim is different from latent_dim
        if self.P_proj is None or self.P_proj.in_features != self.P_j.size(-1):
            self.P_proj = nn.Linear(self.P_j.size(-1), self.latent_dim, bias=False).to(self.P_j.device)
        
        P_j_latent = self.P_proj(self.P_j)
        
        # Key+: トピック位置 + 成長の勢い
        K_plus_input = torch.cat([P_j_latent, self.D_plus_j], dim=-1) 
        K_plus = self.W_K(K_plus_input) # (num_topics, hidden_dim)
        
        # Key-: トピック位置 + 衰退の勢い
        K_minus_input = torch.cat([P_j_latent, self.D_minus_j], dim=-1)
        K_minus = self.W_K(K_minus_input) # (num_topics, hidden_dim)
        
        # Attention scores
        scores_plus = torch.matmul(Q, K_plus.t()) / (self.hidden_dim ** 0.5)
        alpha_plus = F.softmax(scores_plus, dim=-1)
        
        scores_minus = torch.matmul(Q, K_minus.t()) / (self.hidden_dim ** 0.5)
        alpha_minus = F.softmax(scores_minus, dim=-1)
        
        # Vector Field calculation
        # P_j - H_i(t): (num_authors, num_topics, latent_dim)
        diff = P_j_latent.unsqueeze(0) - h_i.unsqueeze(1)
        
        # 流入項: sum_j alpha_{ij}^+ (P_j - H_i(t))
        # (num_authors, num_topics, 1) * (num_authors, num_topics, latent_dim) -> (num_authors, latent_dim)
        v_in = (alpha_plus.unsqueeze(-1) * diff).sum(dim=1)
        
        # 流出項: gamma * sum_j alpha_{ij}^- (P_j - H_i(t))
        v_out = (alpha_minus.unsqueeze(-1) * diff).sum(dim=1)
        
        return v_in - torch.abs(self.gamma) * v_out

class DualForcePNODEPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, gamma=1.0, ode_method="dopri5", ode_n_steps=4):
        super().__init__()
        self.ode_func = DualForcePotentialODEFunc(latent_dim, hidden_dim, gamma)
        self.ode_method = ode_method
        self.ode_n_steps = ode_n_steps

    def forward(self, z_current: torch.Tensor, num_authors: int, delta_t: float = 1.0) -> torch.Tensor:
        """
        二部グラフ上の z は [著者; トピック]。ODE は**著者行のみ**積分し、トピック行は直前 z を維持。
        """
        n = int(num_authors)
        za = z_current[:n]
        z_new_a = integrate_gradient_flow_ode(
            self.ode_func,
            za,
            float(delta_t),
            method=self.ode_method,
            n_steps=self.ode_n_steps,
        )
        if z_current.size(0) <= n:
            return z_new_a
        out = z_current.clone()
        out[:n] = z_new_a
        return out
