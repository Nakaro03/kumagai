# PI-SDE の役割 — 最終決定 (Dual-Track)

**Date**: 2026-05-20
**Status**: 確定 (X3-clean は descriptive、X5 は predictive にロック)

---

## 1. 結論 (一文)

> **X3-clean は "descriptive landscape" として VIS/TVCG/Scientometrics に投稿し、 X5 は "predictive trajectory" として NeurIPS/ICLR に投稿する。両者は同じ Φ-based 数理基盤を共有するが、評価指標・主張・ターゲット venue が異なる別系統の研究成果とする。**

これにより X3 で得られた成果を捨てず、かつ X5 の予測主張を separate な貢献として正当化できる。

---

## 2. 決定の根拠

### 2.1 実証データ ([aggregate_x3_clean_validity.py](../aggregate_x3_clean_validity.py))

| 評価方式 | Spearman ρ | 結論 |
|---|---|---|
| **alltime** (Φ fit on training timepoints) | 0.69 – 0.98 | descriptive 用途では強い |
| **leave-one-out** (paper t=3 holdout) | **−0.35** ± 0.08 | predictive 用途では破綻 |

→ "X3-clean は予測モデルではない" は確定事実。

### 2.2 TSFM ベースライン比較 (前セッション)

3 domain × 8 method の sweep で:
- Chronos-2, Moirai-2, TimesFM が **Spearman ρ で軒並み負相関**
- naive (persistence, mean) も中立程度
- **PI-SDE 単独の予測性能は欠如**と確定

→ 「Spearman ρ on growth」という評価軸自体が時系列予測の本質と乖離していた可能性が高い。

### 2.3 ストーリー分離の必然性

X3 で「予測できる」と主張すると leave-one-out 結果と矛盾する。逆に X3 を捨てると 4 ドメイン × 5 seed の作業が無駄になる。
→ 両者を **異なる主張のもとに同時提出する** のが合理的。

---

## 3. 二系統の責任分離

| 系統 | X3-clean (Descriptive) | X5 (Predictive) |
|---|---|---|
| **主張** | 観測データの interpretable retrospective landscape | held-out 時刻の population trajectory 予測 |
| **訓練 protocol** | All timepoints fit (memorization 許容) | **LOTO** (Leave-One-Timepoint-Out) で訓練中に held-out を作る |
| **損失** | Sinkhorn marginal + Φ-anchor (g_norm 制約) | 4-term composite: predict + phys + geom + smooth |
| **主指標** | Anchor consistency Pearson, alltime Φ-rank ρ, NDCG (alltime) | **W1_marginal, Hits@10, MRR, AP, NDCG@10 (held-out)** |
| **比較対象** | PCA + 成長率カラー (descriptive baseline), interactive HTML | **PRESCIENT / MIOFlow / scNODE / Chronos-2 / Moirai-2** |
| **ターゲット venue** | IEEE VIS, EuroVis, TVCG, Scientometrics | NeurIPS, ICML, ICLR (+ TGB-Seq benchmark) |
| **Limitations 明記** | leave-one-out で破綻、2D 可視化は分散 4% のみ | T=10-12 が小スケール、Chronos-2 と W1 で同等以上を要求 |

---

## 4. リソース配分 (今後 2 ヶ月)

| 期間 | X3-clean (Descriptive) | X5 (Predictive) |
|---|---|---|
| Week 1-2 | 既存 16 landscape figure の最終調整 | **x5/ パッケージ実装 (model/loss/train/eval)** |
| Week 3-4 | Task analysis + Design rationale | smoke test + ablation A0-A6 sweep |
| Week 5-6 | User study pilot (n=3) | PRESCIENT/MIOFlow baseline 導入 |
| Week 7-8 | User study main (n=10) | 3 domain × 5 seed full sweep + 集約 |
| Week 9-10 | VIS draft 執筆 | NeurIPS draft 執筆 |

両者は **独立に走らせる** ので、片方が遅れても他方は止まらない。

---

## 5. 評価指標の正式切替 — 統一定義

X5 の指標は **既存 baseline 評価 ([baseline_all.py](../baseline_all.py)) にも同じ実装を適用** し、表で公平比較できるよう統一する。実装場所: [x5/eval.py](../x5/eval.py)。

### 5.1 Primary metrics (論文 main table)

| 指標 | 意味 | 計算 (実装) |
|---|---|---|
| **W1_marginal** | held-out 時刻の population 分布距離 | `geomloss.SamplesLoss("sinkhorn", p=1, blur=0.05)` |
| **MMD_RBF** | population 分布距離 (補完) | RBF カーネル MMD on samples, σ = median heuristic |
| **Hits@10** | top-10 topic 識別 | rank topics by −Φ value, count overlap with true top-10 by g_norm |
| **MRR** | first-hit rank の reciprocal | mean of 1/(rank of true top-1 in predicted ranking) |
| **AP** | binary high-growth (g_norm > median) の precision-recall area | `sklearn.average_precision_score` |
| **NDCG@10** | ranking quality | 標準 NDCG with g_raw を relevance とする |

### 5.2 Secondary metrics (補助・appendix)

| 指標 | 用途 |
|---|---|
| **Spearman ρ** | 旧主指標、参考用に残す |
| **Anchor Pearson r** | X3-clean descriptive 主張用 |
| **W1_centroid** | topic centroid の予測距離 (補完) |
| **MSE_norm**, **MAE_norm** | 既存 baseline との互換 |

### 5.3 削除する指標

- `prec_at_10` (= Hits@10 と同義、二重カウント回避)
- alltime g_pred Spearman (memorization 由来、誤解を招く)

---

## 6. 主張のスコープ — 公式テンプレ

### X3-clean の主張 (VIS/Scientometrics 向け)

> *"We propose X3-clean, an interpretable energy-based landscape over a bipartite scholarly graph. Φ-rank achieves Spearman ρ ≈ 0.6 on observed timepoints across 3/4 domains; the learned Φ surface visualizes topic density and growth direction in a unified figure. **We explicitly do not claim future prediction**: leave-one-out evaluation confirms the model does not generalize to held-out timepoints. The contribution is a descriptive visualization tool for research strategists."*

### X5 の主張 (NeurIPS 向け)

> *"We propose X5, a Φ-driven predictive Neural SDE that adopts LOTO (Leave-One-Timepoint-Out) training to recover forecasting ability lost in prior descriptive variants. A 4-term composite loss (predict + physical anchor + geodesic + temporal smoothness) enables population-level trajectory prediction on held-out timepoints. X5 achieves W1_marginal improvements of ≥20% over Chronos-2, MIOFlow, and PRESCIENT on 3 scholarly domains; Hits@10 reaches 0.30+ where descriptive baselines collapse to chance. The work bridges cell-trajectory inference (系統 A) and bipartite dynamic graphs (系統 B)."*

---

## 7. 撤回項目 (誤って既に主張していたら明示的に retract)

| 項目 | 撤回理由 |
|---|---|
| 「PI-SDE X3 で未来トピック成長を予測できる」 | leave-one-out で破綻 (Section 2.1) |
| 「Φ = −log p を経験的に証明」 | Pearson r = 0.24 のみ、証明とは呼べない |
| 「単一手法で descriptive と predictive を両立」 | 訓練 protocol が異なる必要があると判明 |

これらは投稿前のドラフトから削除する。

---

## 8. 次のアクション (実装レベル)

1. ✅ 本文書を確定 (`docs/PISDE_ROLE_DECISION.md`)
2. → x5/ パッケージ実装 (本セッションで scaffolding 着手)
3. → 評価モジュール (`x5/eval.py`) を baseline_all.py からも呼べる形で実装
4. → smoke test (1 domain × 30 epoch) で NaN 出ないか確認
5. → 既存 X3-clean checkpoint に新評価指標を適用、descriptive 主張の strength を再測定
