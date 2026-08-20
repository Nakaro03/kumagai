import torch
import torch.nn as nn
import torch.nn.functional as F

from pnode_patent_runner.dual_force_vgae import DualForceVGAE
from pnode_patent_runner.models import integrate_gradient_flow_ode


class TrendAnchoredPotentialODEFunc(nn.Module):
    r"""
    TAP-NODE (Trend-Anchored Potential Neural ODE) のベクトル場定義。

    スカラーポテンシャル:
        Φ(z, t) = -logsumexp_j [ log w_j(t) - ||z - P_j||^2 / (2 h^2) ]
        log w_j(t) = κ * log1p(M_j(t)) + b * D̃_j(t) + c * D̃_j(t) * burst_j(t)
        D̃_j(t) = log1p(D_j^+(t)) - log1p(D_j^-(t))   (符号付き対数トレンド)
        burst_j(t) = gem_skill.py と同一定義（対数モメンタムの年内80パーセンタイル指標）

    勾配流は解析形:
        dz/dt = -α ∇Φ(z) = α ( Σ_j s_j(z) P_j - z ) / h^2
        s_j(z) = softmax_j( log w_j - ||z - P_j||^2 / (2 h^2) )

    学習パラメータは κ, b, log_h, log_scale, c のスカラー 5 個のみ
    （c は既定 0 初期化・burst 未設定時は寄与ゼロで元の TAP-NODE と同一に退化する、
    `docs/DUAL_FORCE_REDESIGN.md` §3 P4 の burst 条件付き拡張）。
    アンカー P_j は当年エンコーダ出力のトピック潜在（detach、データ扱い）。
    衰退は D̃_j < 0 が井戸を浅くする形で単一スカラー場内に表現される。
    """

    def __init__(
        self,
        kappa_init: float = 1.0,
        b_init: float = 0.5,
        log_h_init: float = 0.0,
        log_scale_init: float = -2.0,
        c_init: float = 0.0,
    ):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(float(kappa_init)))
        self.b = nn.Parameter(torch.tensor(float(b_init)))
        self.log_h = nn.Parameter(torch.tensor(float(log_h_init)))
        # GradientODEFunc と同じ初速: softplus(-2) ≈ 0.127
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale_init)))
        self.c = nn.Parameter(torch.tensor(float(c_init)))

        self.register_buffer("P_j", torch.zeros(0, 2))
        self.register_buffer("log1p_mass", torch.zeros(0))
        self.register_buffer("d_tilde", torch.zeros(0))
        self.register_buffer("burst", torch.zeros(0))

    def set_topic_info(
        self,
        P_j: torch.Tensor,
        mass: torch.Tensor,
        trend_plus: torch.Tensor,
        trend_minus: torch.Tensor,
        burst: torch.Tensor = None,
    ) -> None:
        self.P_j = P_j.detach()
        self.log1p_mass = torch.log1p(mass.detach().clamp(min=0.0))
        self.d_tilde = torch.log1p(trend_plus.detach().clamp(min=0.0)) - torch.log1p(
            trend_minus.detach().clamp(min=0.0)
        )
        self.burst = (
            burst.detach() if burst is not None else torch.zeros_like(self.d_tilde)
        )

    def _log_w(self) -> torch.Tensor:
        return (
            self.kappa * self.log1p_mass
            + self.b * self.d_tilde
            + self.c * self.d_tilde * self.burst
        )

    def _logits(self, z: torch.Tensor) -> torch.Tensor:
        h2 = torch.exp(2.0 * self.log_h)
        sq = torch.cdist(z, self.P_j, p=2) ** 2  # (N, J)
        return self._log_w().unsqueeze(0) - sq / (2.0 * h2)

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        """可視化・検証用のスカラーポテンシャル Φ(z)。"""
        if self.P_j.numel() == 0:
            return torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        return -torch.logsumexp(self._logits(z), dim=1)

    def forward(self, t, z: torch.Tensor) -> torch.Tensor:
        if self.P_j.numel() == 0:
            return torch.zeros_like(z)
        h2 = torch.exp(2.0 * self.log_h)
        s = F.softmax(self._logits(z), dim=1)  # (N, J)
        alpha = F.softplus(self.log_scale)
        return alpha * (s @ self.P_j - z) / h2


class TAPNODEPredictor(nn.Module):
    def __init__(self, ode_method: str = "dopri5", ode_n_steps: int = 4, **func_kwargs):
        super().__init__()
        self.ode_func = TrendAnchoredPotentialODEFunc(**func_kwargs)
        self.ode_method = ode_method
        self.ode_n_steps = ode_n_steps

    def forward(self, z_current: torch.Tensor, num_authors: int, delta_t: float = 1.0) -> torch.Tensor:
        """著者行のみ積分し、トピック行は直前 z を維持（DualForcePNODEPredictor と同型）。"""
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


class TAPVGAE(DualForceVGAE):
    """
    エンコーダ・幾何デコーダは DualForceVGAE と共有し、時間発展のみ
    TAP-NODE（トレンド・アンカー型ポテンシャルの純勾配流）に差し替える。
    """

    def __init__(self, *args, **kwargs):
        ode_method = kwargs.get("ode_method", "dopri5")
        ode_n_steps = kwargs.get("ode_n_steps", 4)
        super().__init__(*args, **kwargs)
        self.temporal_predictor = TAPNODEPredictor(
            ode_method=ode_method, ode_n_steps=ode_n_steps
        )
        self.variant = "tap_node"

    def predict_future(self, z_history_list, data_t=None, year_calendar_start=None):
        """
        アンカー P_j は当年潜在 z のトピック行（detach）。質量・トレンド・burst は data_t から。
        `topic_burst` が無いデータ（従来の author_topic バンドル等）では burst=0 に自動退化し、
        元の TAP-NODE と同一の挙動になる。
        """
        z = z_history_list[-1]
        if data_t is not None:
            dtp = data_t.to(z.device, non_blocking=True)
            burst = getattr(dtp, "topic_burst", None)
            self.temporal_predictor.ode_func.set_topic_info(
                z[self.num_corps :],
                dtp.topic_mass.to(z.device),
                dtp.topic_trend_plus.to(z.device),
                dtp.topic_trend_minus.to(z.device),
                burst.to(z.device) if burst is not None else None,
            )
        return self.temporal_predictor(z, self.num_corps, delta_t=1.0)
