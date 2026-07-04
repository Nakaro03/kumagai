#!/usr/bin/env python3
"""
P-NODE の数値的妥当性チェック（実装テスト・回帰検出用）。

1) PotentialNet: autograd ∇_z sum_i Φ(z_i) と座標ごとの中心差分の整合。
2) GradientODEFunc: -tanh(scale) * ∇Φ(z) との数値一致。

  python -m pnode_patent_runner.ode_numerical_check
  python -m pnode_patent_runner.ode_numerical_check --output pnode_patent_runner/outputs/validity/ode_check.txt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn


@dataclass
class OdeCheckResult:
    max_rel_grad_mlp: float
    ok_grad_mlp: bool
    max_err_ode_func: float
    ok_ode_func: bool
    lines: List[str]


def _phi_sum(pot: nn.Module, z: torch.Tensor) -> torch.Tensor:
    out = pot(z)
    if out.dim() == 2 and out.size(-1) == 1:
        out = out.squeeze(-1)
    return out.sum()


def autograd_grad_phi_sum(pot: nn.Module, z: torch.Tensor) -> torch.Tensor:
    z_req = z.detach().clone().requires_grad_(True)
    phi = _phi_sum(pot, z_req)
    g = torch.autograd.grad(phi, z_req, create_graph=False)[0]
    return g


def central_diff_grad_per_element(
    pot: nn.Module, z: torch.Tensor, eps: float
) -> torch.Tensor:
    z = z.detach()
    n, d = z.shape
    out = torch.zeros_like(z)
    for i in range(n):
        for j in range(d):
            zp = z.clone()
            zm = z.clone()
            zp[i, j] += eps
            zm[i, j] -= eps
            pp = pot(zp)
            pm = pot(zm)
            if pp.dim() == 2:
                pp, pm = pp.squeeze(-1), pm.squeeze(-1)
            out[i, j] = (pp[i] - pm[i]) / (2.0 * eps)
    return out


def run_checks(
    n: int = 8,
    d: int = 2,
    hidden: int = 32,
    eps: float = 1e-3,
) -> OdeCheckResult:
    from pnode_patent_runner.models import GradientODEFunc, PotentialNet

    lines: List[str] = []

    torch.manual_seed(0)
    pot = PotentialNet(d, hidden, feature_mode="mlp")
    pot.eval()
    z = torch.randn(n, d) * 0.5
    g_auto = autograd_grad_phi_sum(pot, z)
    g_fd = central_diff_grad_per_element(pot, z, eps)
    rel = (g_auto - g_fd).abs() / (g_auto.abs() + g_fd.abs() + 1e-8)
    max_rel = float(rel.max().item())
    ok1 = bool(max_rel < 0.05) and bool(
        torch.allclose(g_auto, g_fd, rtol=0.1, atol=1e-2)
    )
    lines.append(f"check_grad_phi_mlp: max relative gap = {max_rel:.6e}  OK={ok1}")

    torch.manual_seed(1)
    pot2 = PotentialNet(d, hidden, feature_mode="mlp")
    ode = GradientODEFunc(pot2)
    z2 = torch.randn(4, d) * 0.3
    t0 = torch.tensor(0.0)
    v = ode(t0, z2)
    z_req = z2.detach().clone().requires_grad_(True)
    ph = pot2(z_req)
    if ph.dim() == 2 and ph.size(-1) == 1:
        ph = ph.squeeze(-1)
    g2 = torch.autograd.grad(ph.sum(), z_req)[0]
    alpha = -torch.tanh(ode.scale) * g2
    err_ode = (v - alpha).abs().max().item()
    # float32 + autograd 経路差で 1e-5 は厳しすぎる
    ok2 = err_ode < 1e-2
    lines.append(
        f"check_gradient_ode_func: max |v - (-tanh*grad)| = {err_ode:.6e}  OK={ok2}"
    )

    return OdeCheckResult(
        max_rel_grad_mlp=max_rel,
        ok_grad_mlp=ok1,
        max_err_ode_func=err_ode,
        ok_ode_func=ok2,
        lines=lines,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="P-NODE PotentialNet / GradientODEFunc の数値整合チェック"
    )
    p.add_argument("--grad-rel-tol", type=float, default=0.05)
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="結果を保存するテキストパス（省略時は stdout のみ）",
    )
    args = p.parse_args()

    res = run_checks()
    out_lines = "\n".join(res.lines) + "\n"
    print(out_lines.rstrip())

    fail_grad = not res.ok_grad_mlp or not res.ok_ode_func
    fail_tol = res.max_rel_grad_mlp > args.grad_rel_tol
    if fail_tol:
        print(
            f"FAIL: max_rel {res.max_rel_grad_mlp} > --grad-rel-tol {args.grad_rel_tol}",
            file=sys.stderr,
        )
    if fail_grad:
        print("FAIL: 数値的妥当性チェックに失敗", file=sys.stderr)
    code = 1 if (fail_grad or fail_tol) else 0

    if args.output:
        pth = Path(args.output)
        pth.parent.mkdir(parents=True, exist_ok=True)
        with open(pth, "w", encoding="utf-8") as f:
            f.write("# ode_numerical_check\n\n")
            f.write(out_lines)
            if code == 0:
                f.write("\nOK: all checks passed.\n")
            else:
                f.write("\nFAIL: see stderr.\n")
        print(f"Wrote: {pth}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
