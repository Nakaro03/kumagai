"""test_benchmark_vgae_encoder_detach.py — regression test for the shared-baseline
(static/rnn/neural_ode/pnode) encoder gradient leak, the symmetric fix to
test_dual_force_encoder_detach.py.

`BenchmarkTemporalVGAE.predict_future()` — the single method that implements ALL of
static/rnn/neural_ode/pnode (and gravity/harmonic/evolve_gcn/roland/pnode_explicit/
pnode_residual/pnode_pc) via a `variant` string — used to hand the encoder's raw latent
output straight to `self.temporal_predictor` without detaching it. For "static" in
particular (an identity map, zero parameters) this meant `z_t1_pred IS z_t`: any loss on
the "prediction" was mathematically identical to a loss on the encoder's own output, so
the encoder could trivially "solve" the future-link task by itself regardless of which
temporal module sat downstream. This was the asymmetric half of the fix requested
alongside the Dual-Force-only encoder-detach fix (dual_force_vgae.py, commit 3ca6760) —
without this, a "fair" Dual-Force vs static/RNN/NeuralODE/PNODE comparison isn't apples
to apples (only Dual-Force would be blocked from cheating).

Run: python pnode_patent_runner/test_benchmark_vgae_encoder_detach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pnode_patent_runner.benchmark_vgae import BenchmarkTemporalVGAE

VARIANTS = ["static", "rnn", "neural_ode", "pnode"]


def _build_model_and_data(variant: str):
    torch.manual_seed(0)
    num_corps, num_topics = 6, 4
    num_nodes = num_corps + num_topics
    input_dim, hidden_dim, latent_dim = 8, 16, 2

    model = BenchmarkTemporalVGAE(
        num_nodes=num_nodes, num_corps=num_corps, input_dim=input_dim,
        hidden_dim=hidden_dim, latent_dim=latent_dim, variant=variant,
    )
    model.train()

    x = torch.randn(num_nodes, input_dim)
    src = torch.randint(0, num_corps, (20,))
    dst = torch.randint(num_corps, num_nodes, (20,))
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    eval_edge_index = torch.stack([src[:5], dst[:5]], dim=0)
    return model, x, edge_index, eval_edge_index


def test_future_prediction_loss_does_not_reach_encoder() -> None:
    for variant in VARIANTS:
        model, x, edge_index, eval_edge_index = _build_model_and_data(variant)

        z_t, _, _ = model.encode(x, edge_index)
        z_t1_pred = model.predict_future([z_t])
        future_like_loss = model.decode(z_t1_pred, eval_edge_index).sum() + z_t1_pred.pow(2).sum()

        model.zero_grad()
        if future_like_loss.requires_grad:
            # "static" is a zero-parameter identity map: once z_t1_pred is detached,
            # nothing downstream requires grad at all, so there's nothing to backward()
            # through — that itself is the strongest possible proof of no leak.
            future_like_loss.backward()

        leaked = [n for n, p in model.encoder.named_parameters()
                  if p.grad is not None and torch.any(p.grad != 0)]
        assert not leaked, f"[{variant}] encoder params still leak gradient: {leaked}"


def test_reconstruction_loss_still_trains_encoder() -> None:
    for variant in VARIANTS:
        model, x, edge_index, eval_edge_index = _build_model_and_data(variant)

        z_t, _, _ = model.encode(x, edge_index)
        recon_like_loss = model.decode(z_t, eval_edge_index).sum()

        model.zero_grad()
        recon_like_loss.backward()

        trained = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.encoder.parameters()
        )
        assert trained, f"[{variant}] encoder should still receive gradient from its own reconstruction loss"


if __name__ == "__main__":
    test_future_prediction_loss_does_not_reach_encoder()
    print(f"PASSED: test_future_prediction_loss_does_not_reach_encoder ({', '.join(VARIANTS)})")
    test_reconstruction_loss_still_trains_encoder()
    print(f"PASSED: test_reconstruction_loss_still_trains_encoder ({', '.join(VARIANTS)})")
