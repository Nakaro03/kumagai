"""test_eval_negative_sampling.py — regression test for the future-link eval negative sampler.

`eval_pairs()` (predictability_popularity_auc.py) shares its negative-sampling logic
verbatim with `future_link_auc_scores()` (unified_training.py) and
`future_link_auc_scores_dual_force()` (dual_force_eval.py): all three build the
negative-rejection set from the positive edges, then subsample `max_pos` positives
for scoring. Before 2026-08-21 the rejection set was built AFTER subsampling, so on
any domain/year with more true positive links than `max_pos` (e.g. construction 2021:
70,884 true links vs max_pos=1500), a true future link outside the subsample could be
drawn as a "negative" — 178/200 (89%) mislabeled in an adversarial synthetic check,
~0.67% on real construction data. The fix builds the rejection set from the full
positive edge set before subsampling (see git history around this commit).

Run: python pnode_patent_runner/test_eval_negative_sampling.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from pnode_patent_runner.predictability_popularity_auc import eval_pairs


def test_no_true_positive_leaks_into_negatives() -> None:
    """More positives than max_pos, moderate density so enough valid negatives exist."""
    torch.manual_seed(0)
    num_corps, num_topics = 300, 200
    max_pos, neg_ratio = 1500, 1

    c = torch.randint(0, num_corps, (12000,))
    p = torch.randint(num_corps, num_corps + num_topics, (12000,))
    edge_index = torch.unique(torch.stack([c, p], dim=0), dim=1)
    assert edge_index.size(1) > max_pos, "test setup needs true positives > max_pos"

    data_next = SimpleNamespace(edge_index=edge_index)
    graphs = {2020: SimpleNamespace(edge_index=edge_index), 2021: data_next}

    pos_ei, neg_ei = eval_pairs(
        graphs, num_corps, year_prev=2020, year_next=2021, max_pos=max_pos, neg_ratio=neg_ratio
    )

    assert pos_ei.size(1) == max_pos, f"expected {max_pos} scored positives, got {pos_ei.size(1)}"
    assert neg_ei.size(1) == max_pos * neg_ratio, (
        f"negative sampler fell short of the requested count: {neg_ei.size(1)} < {max_pos * neg_ratio} "
        "(retry budget exhausted — check candidate density before trusting this test's other asserts)"
    )

    full_pos_set = {tuple(edge_index[:, i].tolist()) for i in range(edge_index.size(1))}
    neg_pairs = [tuple(neg_ei[:, i].tolist()) for i in range(neg_ei.size(1))]
    leaked = [pair for pair in neg_pairs if pair in full_pos_set]
    assert not leaked, f"true positive(s) mislabeled as negative: {leaked[:5]}"
    assert len(neg_pairs) == len(set(neg_pairs)), "duplicate negative pairs sampled"

    # determinism: same inputs must reproduce identical positives and negatives
    pos_ei2, neg_ei2 = eval_pairs(
        graphs, num_corps, year_prev=2020, year_next=2021, max_pos=max_pos, neg_ratio=neg_ratio
    )
    assert torch.equal(pos_ei, pos_ei2), "positive sampling is not deterministic across calls"
    assert torch.equal(neg_ei, neg_ei2), "negative sampling is not deterministic across calls"


if __name__ == "__main__":
    test_no_true_positive_leaks_into_negatives()
    print("PASSED: test_no_true_positive_leaks_into_negatives")
