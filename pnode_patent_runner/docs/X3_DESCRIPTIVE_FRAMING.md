# X3-clean — Descriptive Framing (Pivot from Predictive to Retrospective)

本書は **X3-clean の論文化方針**を固定する。実証データに基づき、**何を主張すべきか・してはいけないか**を明文化する。元の "predictive trend modeling" 主張は leave-one-out 実験で**否定された**ため、descriptive analysis tool として再 framing する。

## 1. 経緯と決定

### 1.1 元の主張 (撤回)
> "PI-SDE X3 で**未来のトピック成長を予測**できる。Φ = -log p というエネルギー基底定式化により、単一ハイパラで multi-domain 横断の強い予測性能を達成。"

### 1.2 実証で起きたこと

[aggregate_x3_clean_validity.py](../aggregate_x3_clean_validity.py) (4 domain × 5 seed + paper leave-one-out × 5 seed = 25 ラン):

| 評価方式 | 結果 |
|---------|------|
| **alltime** (学習と評価で同じ t を使用) | g_pred ρ = 0.69–0.98, Φ-rank ρ = -0.26 ~ -0.69 |
| **leave-one-out** (paper t=3 holdout × 5 seed) | **g_pred ρ = -0.35 ± 0.08 (負相関！)**, Φ-rank 0/5 有意 |

→ 「alltime での好成績は **memorization の産物**」が確認された。これは X3 だけでなく、g_n 入力リーク (X3-baseline) を取り除いた X3-clean でも同様。

### 1.3 新しい主張 (確定)
> "PI-SDE X3-clean は、観測済み topic 集合に対する **interpretable retrospective landscape** を構成する。学習された Φ は (1) 観測データの構造を高精度で fit し (alltime Φ-rank ρ≈0.6 across domains)、(2) 物理的意味 (low Φ = 高密度 = 既存集積) を持つ heatmap として可視化可能で、(3) growth anchor 制約により直接成長率の解釈と対応する。**未来予測のためのモデルではない**。"

## 2. 主張のスコープ

### 2.1 言ってよいこと ✅

| 主張 | 根拠 |
|------|------|
| 観測済みデータに対する **descriptive ranking** が高い | alltime Φ-rank ρ≈-0.67 (paper), -0.60 (patent), -0.48 (jp). 5/5 seed 有意 |
| 古典 / 現代 TS ベースライン (alltime 評価) より強い | PatchTST 0.22 → X3-clean Φ-rank 0.69 (paper) |
| **Multi-seed 安定** | std ≤ 0.09 across domains |
| **Multi-domain 汎用性** | 3/4 domain で有意 (arxiv のみ弱い) |
| **単一ハイパラ** (λ_growth) で動作 | engineering robust |
| **解釈可能な可視化** | landscape + anchor + interactive 3 種類提供 |
| Growth anchor 制約が成立 | Anchor proof Pearson r=0.80 (集約), 0.94 (paper t=3) |

### 2.2 言ってはいけないこと ❌

| ❌ 主張 | 反証 |
|--------|------|
| 「未来時点の topic growth を予測できる」 | leave-one-out g_spearman = **-0.35** (負相関) |
| 「Φ = -log p を経験的に証明」 | Panel E Pearson r = 0.24 のみ |
| 「外挿 (unseen t) に強い」 | 同上 leave-one-out 破綻 |
| 「X3 baseline (g_n 入力あり) が予測能力を持つ」 | リーク経由の identity copy |
| 「実時間で次年度のホットトピックを推薦できる」 | 同上 leave-one-out 破綻 |

### 2.3 グレーゾーン ⚠

| ⚠ 主張 | 注意 |
|--------|------|
| 「ベクトル場 -∇Φ は score function」 | 理論的にはそう、実証は interactive HTML の PCA-2D で部分的にしか見えない (4% explained variance) |
| 「retrospective に成長要因を発見」 | case study レベルでは可能、定量主張は要 user study |
| 「他 PI-SDE 変種 (X1, X2) より優れる」 | Φ-rank は X1 が強い (0.80) ので、「X1 と同等 + interpretable」と controlled に表現 |

## 3. 評価方法の変更

### 3.1 削除する評価

- ❌ leave-one-out future prediction MSE
- ❌ leave-one-out g_pred Spearman
- ❌ X3 baseline g_pred の主張全般 (リーク経由)

### 3.2 追加する評価

| 新指標 | 何を測るか | 実装 |
|--------|-----------|------|
| **Historical reconstruction fidelity** | 「観測データを Φ surface が説明できているか」 | alltime Φ-rank ρ, NDCG@10 (既存) |
| **Anchor consistency** | 「学習目的の anchor 制約が成立しているか」 | Pearson r(Φ(c_j,t), -g̃_j(t)) (既存) |
| **Cross-domain robustness** | 「同じ手法・HP で多 domain を fit できるか」 | 4-domain summary (既存) |
| **Case-study quality** | 「具体トピックの軌跡が解釈可能か」 | [plot_pisde_x3_case_study.py](../plot_pisde_x3_case_study.py) |
| **User study (将来)** | 「expert がこの landscape から trend を読めるか」 | task analysis + interview |

### 3.3 比較対象の変更

| 元の比較 | 新しい比較 |
|---------|-----------|
| ARIMA / LSTM / DLinear / PatchTST (予測タスク) | PCA + 成長率カラー (descriptive baseline) |
| TGN / EvolveGCN (graph temporal SOTA) | 不要 (predictive 比較ではないため) |
| X1 vs X2 vs X3 (predictive accuracy) | X1 vs X2 vs X3 (**landscape interpretability**) |

## 4. ターゲット venue

| Venue | 適合度 | 必要な準備 |
|-------|-------|-----------|
| **IEEE VIS / EuroVis (本命)** | ◎ | task analysis + design rationale + user study (10 人 × 2 task) + case studies |
| **TVCG ジャーナル** | ○ | 上記 + より丁寧な技術詳細 |
| **KDD / WWW industry track** | ○ | 大規模実データ + business 解釈 + tool としての位置づけ |
| **Scientometrics / J. of Informetrics** | ◎ | 既に研究領域分析として完成度高い |
| NeurIPS / ICML / ICLR | ❌ | 撤退 (predictive 主張がないため理論貢献として弱い) |

## 5. アウトプット一覧 (X3-clean)

### 5.1 学習結果
- [RESULTS_X3_ABLATION/](../../RESULTS_X3_ABLATION/) — 25 ラン全 checkpoint + evaluation JSON
- [aggregate_x3_clean_validity.py](../aggregate_x3_clean_validity.py) — 集約スクリプト

### 5.2 可視化 (4 domain × seed=42 × 4 種類 = 16 ファイル)

#### 5.2.1 ファイル種別
- `landscape_x3clean_t{T}.png` — 6 パネル静止画 (A:obs B:Φ-heat C:centroid D:rank E:EBM F:anchor)
- `anchor_x3clean_all_t.png` — anchor proof per-t + 集約
- `landscape_x3clean_interactive.html` — t スライダ付き Plotly
- `case_study_x3clean.png` — 注目トピックの軌跡

#### 5.2.2 フルパス一覧 (16 ファイル)

すべて以下のパターン下:
`/home/nakamuraroi/kumagai/RESULTS_X3_ABLATION/{DATA_NAME}/mask/x3abl_mask_g0.5/seed_42/alltime/`

| Domain (DATA_NAME) | trained timepoints | landscape file | リンク |
|--------------------|-------------------|-----------------|--------|
| `PNode_Paper_X1` | 4 (2022-2025) | `landscape_x3clean_t3.png` | [PNG](../../RESULTS_X3_ABLATION/PNode_Paper_X1/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_t3.png) [anchor](../../RESULTS_X3_ABLATION/PNode_Paper_X1/mask/x3abl_mask_g0.5/seed_42/alltime/anchor_x3clean_all_t.png) [case](../../RESULTS_X3_ABLATION/PNode_Paper_X1/mask/x3abl_mask_g0.5/seed_42/alltime/case_study_x3clean.png) [HTML](../../RESULTS_X3_ABLATION/PNode_Paper_X1/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_interactive.html) |
| `PNode_Patent_Energy_X1_top50` | 12 (2010-2021) | `landscape_x3clean_t11.png` | [PNG](../../RESULTS_X3_ABLATION/PNode_Patent_Energy_X1_top50/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_t11.png) [anchor](../../RESULTS_X3_ABLATION/PNode_Patent_Energy_X1_top50/mask/x3abl_mask_g0.5/seed_42/alltime/anchor_x3clean_all_t.png) [case](../../RESULTS_X3_ABLATION/PNode_Patent_Energy_X1_top50/mask/x3abl_mask_g0.5/seed_42/alltime/case_study_x3clean.png) [HTML](../../RESULTS_X3_ABLATION/PNode_Patent_Energy_X1_top50/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_interactive.html) |
| `PNode_ArXiv_Construction_X1_v2` | 11 (2014-2024) | `landscape_x3clean_t10.png` | [PNG](../../RESULTS_X3_ABLATION/PNode_ArXiv_Construction_X1_v2/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_t10.png) [anchor](../../RESULTS_X3_ABLATION/PNode_ArXiv_Construction_X1_v2/mask/x3abl_mask_g0.5/seed_42/alltime/anchor_x3clean_all_t.png) [case](../../RESULTS_X3_ABLATION/PNode_ArXiv_Construction_X1_v2/mask/x3abl_mask_g0.5/seed_42/alltime/case_study_x3clean.png) [HTML](../../RESULTS_X3_ABLATION/PNode_ArXiv_Construction_X1_v2/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_interactive.html) |
| `PNode_JP_Construction_X1` | 11 (2014-2024) | `landscape_x3clean_t10.png` | [PNG](../../RESULTS_X3_ABLATION/PNode_JP_Construction_X1/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_t10.png) [anchor](../../RESULTS_X3_ABLATION/PNode_JP_Construction_X1/mask/x3abl_mask_g0.5/seed_42/alltime/anchor_x3clean_all_t.png) [case](../../RESULTS_X3_ABLATION/PNode_JP_Construction_X1/mask/x3abl_mask_g0.5/seed_42/alltime/case_study_x3clean.png) [HTML](../../RESULTS_X3_ABLATION/PNode_JP_Construction_X1/mask/x3abl_mask_g0.5/seed_42/alltime/landscape_x3clean_interactive.html) |

#### 5.2.3 配下に同居するファイル (1 domain あたり)
- `config.pt` — 学習 config
- `train.best.pt`, `train.epoch_000200.pt` — モデル + predictor の checkpoint
- `evaluation_x3_ablation_mask.json` — 全 t の Spearman/NDCG/MSE 評価値
- `train.log` — epoch ごとの損失推移

### 5.3 スクリプト
- [run_pisde_x3_ablation.py](../run_pisde_x3_ablation.py) — X3-clean training (MODE=mask)
- [plot_pisde_x3_landscape.py](../plot_pisde_x3_landscape.py) — 6 パネル
- [plot_pisde_x3_anchor.py](../plot_pisde_x3_anchor.py) — anchor
- [run_interactive_landscape_pisde_x3.py](../run_interactive_landscape_pisde_x3.py) — HTML
- [plot_pisde_x3_case_study.py](../plot_pisde_x3_case_study.py) — case study (NEW)

## 6. 次のステップ

### 6.1 必須 (vis venue 提出に向けて)
1. **Task analysis**: 想定ユーザ (research strategist, patent analyst) の task をリストアップ
2. **Design rationale**: なぜ Φ-anchor を可視化に使うか、なぜ UMAP / PCA か
3. **User study プロトコル設計**: 10 人 × 2 タスク (trend identification, anomaly spotting)

### 6.2 推奨
4. **Descriptive baseline 比較**: PCA + g カラー vs X3-clean Φ で「trend visibility」を比較
5. **X1, X2 も同条件で landscape を生成** — 「Φ-rank は X1 が強いが visualization 解釈は X3-clean が容易」を示せれば差別化

### 6.3 任意
6. PI-SDE family 全体の leave-one-out チェック (X1, X2) — 「memorization は family-wide な特性」と明示するため

## 7. リスクと limitation の明記

論文の Limitations セクションには以下を**自発的に明記**する:

1. **本手法は未来予測ではない**: 全 metric は観測データに対する fit。Leave-one-out で破綻することは検証済 (Appendix X)
2. **Memorization-based**: 学習データに対する高 ρ は memorization を含む。これは "descriptive" claim の範囲では問題ないが、誤読を避けるため明記する
3. **2D 可視化の物理的限界**: PCA-2D は分散の 4-15% のみ説明。Score field 矢印は近似
4. **Topic 数依存**: 30 トピック程度の小スケールで動作。1000+ トピックでの挙動は未検証
5. **Domain dependence**: arxiv_construction で Φ-rank が弱い (1/5 seed 有意)。一般化のためには追加実験が必要

これらを冒頭から書くことで「**honest descriptive contribution**」として査読者に信頼される位置を取る。
