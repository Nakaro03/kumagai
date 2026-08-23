"""test_dual_force_encoder_detach.py — regression test for the Dual-Force encoder gradient leak.

`DualForceVGAE.predict_future()` used to pass the encoder's raw output straight into the
temporal-dynamics module (`DualForcePNODEPredictor`) without detaching it. Any loss computed
from the predicted future latent (`latent_pred_loss`, `future_link_loss` in
`compute_dual_force_loss`) therefore back-propagated through the ODE AND all the way into the
GAT encoder — so the encoder itself could "absorb the answer" without the Dual-Force attention
mechanism doing any work. This was the root cause behind SS6.7 of the (now-frozen)
DUAL_FORCE_REDESIGN.md: rank_renorm showed no significant edge over static/RNN/NeuralODE/PNODE
once all five shared one encoder trained end-to-end.

Fix: `predict_future()` now detaches the latest latent snapshot before handing it to the
temporal-dynamics module. The encoder still trains normally via reconstruction/KL loss, which
use `encode()`'s raw (non-detached) output directly — this test checks both properties.

Run: python pnode_patent_runner/test_dual_force_encoder_detach.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pnode_patent_runner.dual_force_vgae import DualForceVGAE


def _build_model_and_data():
    torch.manual_seed(0)
    num_authors, num_topics = 6, 4
    num_nodes = num_authors + num_topics
    input_dim, hidden_dim, latent_dim = 8, 16, 2

    model = DualForceVGAE(
        num_nodes=num_nodes, num_authors=num_authors, input_dim=input_dim,
        hidden_dim=hidden_dim, latent_dim=latent_dim,
    )
    model.train()

    x = torch.randn(num_nodes, input_dim)
    src = torch.randint(0, num_authors, (20,))
    dst = torch.randint(num_authors, num_nodes, (20,))
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    class _Data:
        pass

    data_t = _Data()
    data_t.topic_trend_plus = torch.rand(num_topics)
    data_t.topic_trend_minus = torch.rand(num_topics)
    data_t.to = lambda device, non_blocking=False: data_t  # matches Data.to(...) usage

    eval_edge_index = torch.stack([src[:5], dst[:5]], dim=0)
    return model, x, edge_index, data_t, eval_edge_index


def test_future_prediction_loss_does_not_reach_encoder() -> None:
    model, x, edge_index, data_t, eval_edge_index = _build_model_and_data()

    z_t, _, _ = model.encode(x, edge_index)
    z_t1_pred = model.predict_future([z_t], data_t)
    future_like_loss = model.decode(z_t1_pred, eval_edge_index).sum() + z_t1_pred.pow(2).sum()

    model.zero_grad()
    future_like_loss.backward()

    leaked = [n for n, p in model.encoder.named_parameters()
              if p.grad is not None and torch.any(p.grad != 0)]
    assert not leaked, f"encoder params still receive gradient from the future-prediction loss: {leaked}"

    temporal_grad = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.temporal_predictor.parameters()
    )
    assert temporal_grad, "temporal_predictor should still receive gradient (fix should not kill training signal)"


def test_reconstruction_loss_still_trains_encoder() -> None:
    model, x, edge_index, data_t, eval_edge_index = _build_model_and_data()

    z_t, _, _ = model.encode(x, edge_index)
    recon_like_loss = model.decode(z_t, eval_edge_index).sum()

    model.zero_grad()
    recon_like_loss.backward()

    trained = any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in model.encoder.parameters()
    )
    assert trained, "encoder should still receive gradient from its own (reconstruction) loss"


if __name__ == "__main__":
    test_future_prediction_loss_does_not_reach_encoder()
    print("PASSED: test_future_prediction_loss_does_not_reach_encoder")
    test_reconstruction_loss_still_trains_encoder()
    print("PASSED: test_reconstruction_loss_still_trains_encoder")
