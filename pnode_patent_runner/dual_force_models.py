import torch
import torch.nn as nn
import torch.nn.functional as F
from pnode_patent_runner.models import GradientODEFunc, integrate_gradient_flow_ode

class DualForcePotentialODEFunc(nn.Module):
    r"""
    Dual-Force P-NODE のベクトル場定義（v2: 単一Attention・潜在空間アンカー版）。

    P_j はトピックノードの**潜在ベクトル z_j**（エンコーダ出力、detach — TAP-NODEと同じ
    アンカーの取り方）。D_j はトピック j の**符号付き**トレンド（D_j = M_j(t) - M_j(t-1)）。

    Key はトピック位置に成長・衰退の勢いを足し込んだ単一系統:
        K_j = W_K (P_j + scale(D_j))
        Q_i = W_Q z_i
        alpha_ij = softmax_j( Q_i . K_j / sqrt(hidden_dim) )

    ベクトル場は、同じ alpha_ij を D_j の**符号**（scale前の生の値で判定）でマスクして
    引力・反発の2項に配分する:
        dz_i/dt = sum_{j: D_j>0} a_ij (P_j - z_i)  -  gamma * sum_{j: D_j<0} a_ij (P_j - z_i)

    2つのablation軸（未解決の設計論点への対処、2026-07-22追加）:

    - **d_scale_mode**（Keyに足し込む D_j の大きさをどう扱うか）:
        - "raw"（既定・現行版）: D_j をそのまま加算（生の件数差、P_j のスケールを圧倒しうる）
        - "learnable": D_j に学習可能スカラー exp(log_d_scale) を掛けてから加算
          （K_j = W_K(P_j + exp(log_d_scale)*D_j)、log_d_scale の初期値は log(1)=0）
        - "zscore": D_j をその年のトピック集団内でz-score標準化してから加算
          （符号の判定＝マスクには影響しない。標準化は Key への入力にのみ使う）
        - "rank": D_j を分位点（ランク）変換してから加算（quantile normalization）。
          D_hat_j = Phi^-1((rank(D_j)-0.5)/J)。z-scoreは外れ値がそのまま桁違いの値として残り
          Key/Attentionを支配しうる（2026-07-24 実測: 82,561社中86.8%が同じ1トピックに
          Attention集中）。ランク変換は成長・衰退の順序関係を保ちながら、最大値の大きさを
          有界にする（統計学の定番の外れ値対策）。
    - **renorm_masked_attention**（マスク後にグループ内で再正規化するか）:
        - False（既定・現行版）: alpha_ij にマスクを掛けるだけ（横ばいトピックへの確率質量は捨てられる）
        - True: マスク後に成長側・衰退側それぞれの中で合計1になるよう再正規化する
          （a_ij = alpha_ij*mask_j / sum_k alpha_ik*mask_k、横ばいトピックが多くても駆動力は薄まらない）
    """
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        gamma=1.0,
        d_scale_mode: str = "raw",
        renorm_masked_attention: bool = False,
    ):
        super().__init__()
        assert d_scale_mode in ("raw", "learnable", "zscore", "rank")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.d_scale_mode = d_scale_mode
        self.renorm_masked_attention = renorm_masked_attention

        # アテンション用の重み行列（単一系統）
        self.W_Q = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(latent_dim, hidden_dim, bias=False)  # 入力は P_j + scale(D_j)

        if d_scale_mode == "learnable":
            # exp(log_d_scale) が D_j に掛かる倍率。初期値 log(1)=0 で v1(raw)と同じ挙動から学習開始
            self.log_d_scale = nn.Parameter(torch.tensor(0.0))

        # トピックの潜在位置 P_j と符号付きトレンド D_j を保持するためのプレースホルダ
        # 実行時に set_topic_info で設定される
        self.register_buffer("P_j", torch.zeros(0, latent_dim))
        self.register_buffer("D_j", torch.zeros(0, 1))  # 符号判定用（常に生の値）

    def set_topic_info(self, P_j, D_j):
        self.P_j = P_j.detach()
        self.D_j = D_j.detach()  # マスクの符号判定は常にこの生の値を使う

    def _d_for_key(self) -> torch.Tensor:
        if self.d_scale_mode == "zscore":
            mu = self.D_j.mean()
            sd = self.D_j.std(unbiased=False).clamp_min(1e-6)
            return (self.D_j - mu) / sd
        if self.d_scale_mode == "rank":
            return self._rank_gaussian(self.D_j)
        if self.d_scale_mode == "learnable":
            return self.D_j * torch.exp(self.log_d_scale)
        return self.D_j  # "raw"

    @staticmethod
    def _rank_gaussian(x: torch.Tensor) -> torch.Tensor:
        """quantile normalization: D_hat_j = Phi^-1((rank(D_j)-0.5)/J)。
        同値（tie）は平均順位で扱う。外れ値の順序は保つが大きさは有界にする。"""
        flat = x.squeeze(-1)
        n = flat.numel()
        order = torch.argsort(flat)
        ranks = torch.empty(n, dtype=flat.dtype, device=flat.device)
        ranks[order] = torch.arange(1, n + 1, dtype=flat.dtype, device=flat.device)
        # 同値の tie は平均順位に補正
        _, inverse, counts = torch.unique(flat, return_inverse=True, return_counts=True)
        sum_ranks = torch.zeros(counts.shape[0], dtype=flat.dtype, device=flat.device)
        sum_ranks.scatter_add_(0, inverse, ranks)
        avg_rank_per_group = sum_ranks / counts.to(flat.dtype)
        ranks = avg_rank_per_group[inverse]
        p = (ranks - 0.5) / n
        p = p.clamp(1.0 / (4 * n), 1 - 1.0 / (4 * n))
        q = torch.distributions.Normal(0.0, 1.0).icdf(p)
        return q.unsqueeze(-1)

    def forward(self, t, h_i):
        """
        h_i: (num_authors, latent_dim)
        """
        if self.P_j.numel() == 0:
            return torch.zeros_like(h_i)

        # Query: 著者の現在地
        Q = self.W_Q(h_i)  # (num_authors, hidden_dim)

        # Key: トピック位置 + （scaleされた）トレンド
        K_input = self.P_j + self._d_for_key()  # (num_topics, latent_dim)
        K = self.W_K(K_input)  # (num_topics, hidden_dim)

        # Attention（単一系統）
        scores = torch.matmul(Q, K.t()) / (self.hidden_dim ** 0.5)
        alpha = F.softmax(scores, dim=-1)  # (num_authors, num_topics)

        # 成長トピック(D_j>0)・衰退トピック(D_j<0)へのマスク（符号判定は常に生のD_j）
        mask_pos = (self.D_j.squeeze(-1) > 0).to(alpha.dtype)  # (num_topics,)
        mask_neg = (self.D_j.squeeze(-1) < 0).to(alpha.dtype)

        a_pos = alpha * mask_pos.unsqueeze(0)  # (num_authors, num_topics)
        a_neg = alpha * mask_neg.unsqueeze(0)

        if self.renorm_masked_attention:
            # clamp_min を極小値(例: 1e-8)にすると、成長側/衰退側の質量がごく小さい研究者で
            # a_pos/a_neg が桁違いに増幅され、ODE積分中にNaNへ発散する事例を実測
            # （learnable_renorm, construction, seed=7）。0.05 を下限にして増幅率を
            # 最大20倍に抑え、質量がほぼ無い場合は再正規化しない挙動に近づける。
            a_pos = a_pos / a_pos.sum(dim=1, keepdim=True).clamp_min(0.05)
            a_neg = a_neg / a_neg.sum(dim=1, keepdim=True).clamp_min(0.05)

        # sum_j a_ij (P_j - h_i) = a_ij@P_j - h_i * sum_j(a_ij)（分配法則）を行列積で計算し、
        # (num_authors, num_topics, latent_dim) の3次元テンソルを一切materializeしない。
        # トピック数が多いドメイン（例: construction は CPC 4,940 種類）で、
        # 素朴なbroadcast-then-sum実装は GPU メモリを容易に使い切るため必須の最適化。
        v_in = a_pos @ self.P_j - h_i * a_pos.sum(dim=1, keepdim=True)
        v_out = a_neg @ self.P_j - h_i * a_neg.sum(dim=1, keepdim=True)

        return v_in - torch.abs(self.gamma) * v_out

class DualForcePNODEPredictor(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_dim,
        gamma=1.0,
        ode_method="dopri5",
        ode_n_steps=4,
        d_scale_mode: str = "raw",
        renorm_masked_attention: bool = False,
    ):
        super().__init__()
        self.ode_func = DualForcePotentialODEFunc(
            latent_dim,
            hidden_dim,
            gamma,
            d_scale_mode=d_scale_mode,
            renorm_masked_attention=renorm_masked_attention,
        )
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
