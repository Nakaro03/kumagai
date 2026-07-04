"""
技術トレンド予測の評価指標モジュール。

問題設定: 「将来リンク予測 AUC」ではなく「技術トレンド景観の推定」を評価する。

主要指標:
  - Entry-AUC  : 学習期間に存在しなかった新規リンクのみの ROC-AUC
                 「著者が初めて参入する技術」を正しく予測できるか
  - Exit-AUC   : 学習期間に存在したが消滅したリンクの ROC-AUC
                 「著者が離れる技術」を正しく予測できるか
  - Spearman(Φ,g): Φランキング vs 実際の成長率の Spearman 相関
                   PC-PNODE だけが出力できる指標
  - NDCG@K     : 上位K成長トピックの予測精度

比較戦略:
  - Static/NeuralODE/PNODE はΦを持たないため Spearman は N/A
  - Entry-AUC で「動的モデルの優位」を測る
  - Spearman で「PC-PNODE のトレンド景観の意味」を測る
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _build_edge_set(edge_index: torch.Tensor, num_corps: int) -> Set[Tuple[int, int]]:
    """二部グラフのエッジを (著者idx, トピックidx) の set で返す。"""
    ei = edge_index
    mask = (ei[0] < num_corps) & (ei[1] >= num_corps)
    ei = ei[:, mask]
    return {(int(ei[0, i]), int(ei[1, i])) for i in range(ei.size(1))}


def _encode_and_predict(
    model: torch.nn.Module,
    graph,
    edges: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """モデルで z を得てエッジスコアを返す。"""
    with torch.no_grad():
        z, _, _ = model.encode(graph.x, graph.edge_index)
        scores = model.decode(z, edges).squeeze(-1)
        if hasattr(scores, "sigmoid"):
            pass
        scores = torch.sigmoid(scores) if scores.min() < 0 or scores.max() > 1 else scores
    return scores.cpu().numpy()


def _sample_neg_edges(
    pos_edges: Set[Tuple[int, int]],
    exclude_edges: Set[Tuple[int, int]],
    author_ids: np.ndarray,
    topic_ids: np.ndarray,
    n_neg: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """pos_edges / exclude_edges に含まれない負例エッジをサンプリング。"""
    neg = []
    max_tries = n_neg * 20
    for _ in range(max_tries):
        if len(neg) >= n_neg:
            break
        a = int(rng.choice(author_ids))
        t = int(rng.choice(topic_ids))
        if (a, t) not in pos_edges and (a, t) not in exclude_edges:
            neg.append((a, t))
    return neg


# ─────────────────────────────────────────────────────────────────────────────
# Entry-AUC: 新規参入リンクのみ
# ─────────────────────────────────────────────────────────────────────────────

def compute_entry_auc(
    model: torch.nn.Module,
    graphs: Dict[int, object],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
    max_pos: int = 800,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Entry-AUC: 学習期間に一度も現れなかった新規リンクのみで評価。

    「著者が初めて参入する技術を正しく予測できるか」
    Static VGAE は "同じリンクが続く" ことを予測するため不利になる。
    """
    train_years = [y for y in sorted(graphs.keys()) if y <= year_prev]
    if year_next not in graphs or not train_years:
        return {"entry_auc": float("nan"), "entry_ap": float("nan"), "n_entry": 0}

    # 学習期間の全エッジ集合
    hist_edges: Set[Tuple[int, int]] = set()
    for y in train_years:
        hist_edges |= _build_edge_set(graphs[y].edge_index, num_corps)

    # ホールドアウト年のエッジ
    ho_edges = _build_edge_set(graphs[year_next].edge_index, num_corps)

    # 新規参入: ホールドアウトに出現 かつ 学習期間に未出現
    entry_edges = ho_edges - hist_edges
    if len(entry_edges) < 5:
        return {"entry_auc": float("nan"), "entry_ap": float("nan"), "n_entry": len(entry_edges)}

    rng = np.random.default_rng(seed)
    entry_list = list(entry_edges)
    if len(entry_list) > max_pos:
        idxs = rng.choice(len(entry_list), max_pos, replace=False)
        entry_list = [entry_list[i] for i in idxs]

    author_ids = np.array(sorted({a for a, _ in hist_edges | ho_edges}))
    topic_ids  = np.array(sorted({t for _, t in hist_edges | ho_edges}))

    neg_list = _sample_neg_edges(
        set(entry_list), hist_edges | ho_edges,
        author_ids, topic_ids, len(entry_list), rng,
    )
    if len(neg_list) < 5:
        return {"entry_auc": float("nan"), "entry_ap": float("nan"), "n_entry": len(entry_list)}

    all_edges = entry_list + neg_list
    labels = np.array([1] * len(entry_list) + [0] * len(neg_list))
    ei = torch.tensor(all_edges, dtype=torch.long).T.to(device)

    data_prev = graphs[year_prev].to(device)
    scores = _encode_and_predict(model, data_prev, ei, device)

    try:
        auc = roc_auc_score(labels, scores)
        ap  = average_precision_score(labels, scores)
    except Exception:
        auc, ap = float("nan"), float("nan")

    return {"entry_auc": auc, "entry_ap": ap, "n_entry": len(entry_list)}


# ─────────────────────────────────────────────────────────────────────────────
# Exit-AUC: 離脱リンクのみ
# ─────────────────────────────────────────────────────────────────────────────

def compute_exit_auc(
    model: torch.nn.Module,
    graphs: Dict[int, object],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
    max_pos: int = 800,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Exit-AUC: 直前年に存在したが翌年に消えたリンクのみで評価。

    低スコア(モデルが「次年はない」と予測)が正解。
    AUC を「消滅予測」として反転して報告する。
    """
    if year_prev not in graphs or year_next not in graphs:
        return {"exit_auc": float("nan"), "n_exit": 0}

    prev_edges = _build_edge_set(graphs[year_prev].edge_index, num_corps)
    next_edges = _build_edge_set(graphs[year_next].edge_index, num_corps)

    exit_edges = prev_edges - next_edges   # 消えたリンク
    stay_edges = prev_edges & next_edges   # 残ったリンク

    if len(exit_edges) < 5 or len(stay_edges) < 5:
        return {"exit_auc": float("nan"), "n_exit": len(exit_edges)}

    rng = np.random.default_rng(seed)
    exit_list = list(exit_edges)
    stay_list = list(stay_edges)

    n = min(len(exit_list), len(stay_list), max_pos)
    exit_list = [exit_list[i] for i in rng.choice(len(exit_list), n, replace=False)]
    stay_list = [stay_list[i] for i in rng.choice(len(stay_list), n, replace=False)]

    all_edges = exit_list + stay_list
    # 残留(1) vs 離脱(0) → モデルは残留を高スコアにするはずなので離脱を検出できる
    labels = np.array([0] * len(exit_list) + [1] * len(stay_list))
    ei = torch.tensor(all_edges, dtype=torch.long).T.to(device)

    data_prev = graphs[year_prev].to(device)
    scores = _encode_and_predict(model, data_prev, ei, device)

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = float("nan")

    return {"exit_auc": auc, "n_exit": len(exit_list)}


# ─────────────────────────────────────────────────────────────────────────────
# Spearman(Φ, g): Φランキング vs 成長率  [PC-PNODE 専用]
# ─────────────────────────────────────────────────────────────────────────────

def compute_spearman_phi_growth(
    model: torch.nn.Module,
    graphs: Dict[int, object],
    num_corps: int,
    device: torch.device,
    topic_growth_by_year: Dict[int, torch.Tensor],
    year: int,
) -> Dict[str, float]:
    """
    Spearman(Φ(z_topic), g_topic): PC-PNODE のトレンド景観の品質を測る。

    - r < 0: 低Φ(谷) = 成長トピック  [期待動作]
    - r ≈ 0: Static/PNODE と同等 (景観が無意味)
    - 比較手法は Φ を持たないので N/A

    Returns
    -------
    dict with keys: spearman_r, spearman_p, kendall_tau, n_topics, phi_span
    """
    nan_result = {
        "spearman_r": float("nan"), "spearman_p": float("nan"),
        "kendall_tau": float("nan"), "n_topics": 0, "phi_span": float("nan"),
    }

    if year not in graphs or year not in topic_growth_by_year:
        return nan_result

    pot = getattr(getattr(model, "temporal_predictor", None), "potential_net", None)
    if pot is None:
        return nan_result

    data_y = graphs[year].to(device)
    with torch.no_grad():
        z, _, _ = model.encode(data_y.x, data_y.edge_index)
        z_topics = z[num_corps:]
        if hasattr(pot, "set_population"):
            pot.set_population(z.detach())
        phi = pot(z_topics).squeeze(-1)

    g = topic_growth_by_year[year]
    n = min(len(phi), len(g))
    phi_np = phi[:n].cpu().numpy()
    g_np   = g[:n].numpy()

    mask = np.isfinite(phi_np) & np.isfinite(g_np)
    if mask.sum() < 5:
        return nan_result

    r, p = stats.spearmanr(phi_np[mask], g_np[mask])
    tau, _ = stats.kendalltau(phi_np[mask], g_np[mask])

    return {
        "spearman_r":   float(r),
        "spearman_p":   float(p),
        "kendall_tau":  float(tau),
        "n_topics":     int(mask.sum()),
        "phi_span":     float(phi_np[mask].max() - phi_np[mask].min()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# NDCG@K: 上位K成長トピックの予測精度  [PC-PNODE 専用]
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndcg_phi_growth(
    model: torch.nn.Module,
    graphs: Dict[int, object],
    num_corps: int,
    device: torch.device,
    topic_growth_by_year: Dict[int, torch.Tensor],
    year: int,
    k: int = 10,
) -> Dict[str, float]:
    """
    NDCG@K: Φ が低い(谷)トップKトピックに実際の成長率が集中しているか。

    relevance = max(0, g_j)  (成長率を正値化)
    予測ランキング = Φ 昇順 (低Φ = 注目予測)
    """
    nan_result = {"ndcg": float("nan"), "precision_at_k": float("nan")}

    pot = getattr(getattr(model, "temporal_predictor", None), "potential_net", None)
    if pot is None or year not in graphs or year not in topic_growth_by_year:
        return nan_result

    data_y = graphs[year].to(device)
    with torch.no_grad():
        z, _, _ = model.encode(data_y.x, data_y.edge_index)
        z_topics = z[num_corps:]
        if hasattr(pot, "set_population"):
            pot.set_population(z.detach())
        phi = pot(z_topics).squeeze(-1).cpu().numpy()

    g = topic_growth_by_year[year].numpy()
    n = min(len(phi), len(g))
    phi, g = phi[:n], g[:n]

    # Φ 昇順ランキング (低Φ = 注目予測 = 先頭)
    phi_rank = np.argsort(phi)
    g_rel    = np.maximum(g, 0.0)  # 負値は 0 に

    if g_rel.sum() < 1e-8:
        return nan_result

    # DCG@K
    top_k = phi_rank[:k]
    dcg = sum(g_rel[i] / np.log2(rank + 2) for rank, i in enumerate(top_k))

    # IDCG@K
    ideal_rank = np.argsort(-g_rel)
    idcg = sum(g_rel[ideal_rank[rank]] / np.log2(rank + 2) for rank in range(min(k, n)))

    ndcg = dcg / idcg if idcg > 0 else float("nan")

    # Precision@K: 上位K中で実際に成長(g>0)のトピック数
    growing = (g[top_k] > 0).sum()
    prec_k  = growing / k

    return {"ndcg": float(ndcg), "precision_at_k": float(prec_k)}


# ─────────────────────────────────────────────────────────────────────────────
# 全指標をまとめて計算
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_trend_metrics(
    model: torch.nn.Module,
    graphs: Dict[int, object],
    num_corps: int,
    device: torch.device,
    year_prev: int,
    year_next: int,
    topic_growth_by_year: Optional[Dict[int, torch.Tensor]] = None,
    ndcg_k: int = 10,
) -> Dict[str, float]:
    """
    全トレンド評価指標を一括計算して dict で返す。

    Keys:
        entry_auc, entry_ap, n_entry
        exit_auc, n_exit
        spearman_r, spearman_p, phi_span   (Φ なし手法は nan)
        ndcg, precision_at_k               (Φ なし手法は nan)
    """
    result: Dict[str, float] = {}

    # Entry / Exit AUC
    result.update(compute_entry_auc(model, graphs, num_corps, device, year_prev, year_next))
    result.update(compute_exit_auc(model, graphs, num_corps, device, year_prev, year_next))

    # Spearman / NDCG (Φ 保有モデルのみ有効)
    if topic_growth_by_year is not None:
        sp = compute_spearman_phi_growth(
            model, graphs, num_corps, device, topic_growth_by_year, year_prev,
        )
        result["spearman_r"]  = sp["spearman_r"]
        result["spearman_p"]  = sp["spearman_p"]
        result["phi_span"]    = sp["phi_span"]

        nd = compute_ndcg_phi_growth(
            model, graphs, num_corps, device, topic_growth_by_year, year_prev, k=ndcg_k,
        )
        result["ndcg"]           = nd["ndcg"]
        result["precision_at_k"] = nd["precision_at_k"]
    else:
        for k in ("spearman_r", "spearman_p", "phi_span", "ndcg", "precision_at_k"):
            result[k] = float("nan")

    return result
