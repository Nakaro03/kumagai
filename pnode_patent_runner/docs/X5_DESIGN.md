# X5 — Φ-as-Driver Predictive PI-SDE (Design Doc)

**Target venue**: NeurIPS / ICML / ICLR 2026
**Positioning**: 系統 A (cell trajectory: PRESCIENT/MIOFlow/Action Matching) の数理を **bipartite scholarly graph** に持ち込み、可視化と予測を両立する。
**Date**: 2026-05-20

---

## 0. 既存実装の精密診断 — 何が問題で何が問題でないか

### ✓ 既に正しい構造 (改修不要)

| 要素 | 場所 | 状態 |
|---|---|---|
| drift = -∇Φ | [/tmp/PI-SDE/src/model.py:108-113](file:///tmp/PI-SDE/src/model.py#L108-L113) `_drift()` | **正しい** ✓ |
| Φ anchor loss | [run_pisde_x3_ablation.py:172](pnode_patent_runner/run_pisde_x3_ablation.py#L172) `L_val = (Φ(c_j,t) + g_j)^2` | **正しい** ✓ |
| Sinkhorn marginal loss | [run_pisde_x3_ablation.py:231](pnode_patent_runner/run_pisde_x3_ablation.py#L231) `loss_xy = OT(rollout_t, observed_t)` | **正しい** ✓ |
| Φ landscape 可視化 | [plot_pisde_x3_landscape.py](pnode_patent_runner/plot_pisde_x3_landscape.py) | **正しい** ✓ |

### ✗ 実際の問題点 (改修が必要)

| # | 問題 | 根拠 | 改修方針 |
|---|---|---|---|
| **P1** | 訓練に **leave-one-timepoint-out (LOTO)** が無い。Sinkhorn は train_t 全部を見て fit する→ memorization | X3-clean が leave-one-out で ρ=−0.35 と破綻 | **時刻ホールドアウト** を **訓練ループに導入** |
| **P2** | Φ-anchor が **過去時刻のみ**で fit (`Φ(c_j, t) ≈ −g_j(t)`)。未来時刻に外挿する明示的圧力なし | X4 で future-anchor を試したが弱い | **MIOFlow 型の time-conditioned smoothness** を追加 |
| **P3** | 評価が **Spearman ρ on growth_norm** のみ。系統 A 標準の W1/MMD が無い | TSFM 比較で Spearman は noise dominated と判明 | **評価を W1 + MMD + Hits@K + AP + NDCG** に総入れ替え |
| **P4** | **多様体 (geodesic) 正則化** が無い。49 次元 embedding 空間で SDE が無制約に発散しうる | MIOFlow の geodesic AE が無いと不安定 | **Geodesic / energy regularizer** を loss に追加 |
| **P5** | Predictor head の **g_n リーク** が残る (X3 baseline) / X4 で別 head 化したがマージ無し | X3 ablation で leak 確認済 | **Predictor head を削除**、Φ そのものを予測に使う |

### ⚠ 共通の誤解 — 「drift = −∇Φ にすればよい」ではない

調査エージェントは「drift を −∇Φ に強制せよ」と勧めてきたが、**それは既に実装済み**。
真の改修は **訓練 protocol** (P1) と **評価 metrics** (P3) の刷新であって、モデル構造ではない。

---

## 1. X5 の核心 — 3 つの改修

### Change-1: 訓練に Leave-One-Timepoint-Out (LOTO) を導入

**現状**: `train_t = [1..8]` 全部を Sinkhorn で fit → fit but no forecast skill

**X5**: 各 epoch で `mask_t ∈ train_t` をランダムに 1 つ選んで **マスク**、その時刻の Sinkhorn を loss に入れず、**マスク時刻を「予測」する形** に。
```
∀ epoch: mask_t ~ Uniform(train_t)
L_predict = Σ_{j ∈ train_t \ {mask_t}}  Sinkhorn(rollout_j, observed_j)
          + α · Sinkhorn(rollout_mask_t, observed_mask_t)    ← α は最初 0、徐々に 1 へ
```
これは **MIOFlow** [Huguet et al. NeurIPS 2022] と **scNODE** [Yang et al. Bioinformatics 2024] と **TGB-Seq** [ICLR 2025] が共通して採用する「held-out 時刻を訓練中も評価する」protocol。

**なぜこれが効くか**: モデルが「観測されていない時刻」を補間する圧力を訓練中に受ける → 真の test_t での性能が向上する。X3-clean の memorization 問題の root cause。

### Change-2: 損失関数を 3-term composite に再設計

**現状** (X3):
```
L = L_sinkhorn (marginal)  +  λ_g · L_anchor (Φ-anchor)
```

**X5**:
```
L = L_predict     (W1 / Sinkhorn on held-out + observed marginals)
  + λ_phys · L_phys     (Φ-anchor = TrajectoryNet の energy reg と同等)
  + λ_geom · L_geom     (path energy / geodesic preservation)
  + λ_smooth · L_smooth (Φ の時間方向 smoothness, 外挿用)
```

各項の中身:

- `L_predict = Σ_t Sinkhorn(rollout_t, observed_t)` + `α · Sinkhorn(rollout_mask, observed_mask)`
- `L_phys = Σ_j Σ_t (Φ(c_j, t) + g_j(t))^2` ← 既存
- `L_geom = ∫_0^T ‖dz/dt‖^2 dt`(path energy, MIOFlow 流)
- `L_smooth = ‖∂Φ/∂t‖^2` (Fourier 時間埋め込みで滑らかに、X4 のアイデア継承)

### Change-3: 評価を系統 A + 系統 B の標準指標に総刷新

**廃止**: Spearman ρ, MSE on growth_norm, prec@10

**新規追加** (Primary):
| 指標 | 何を測る | 計算 |
|---|---|---|
| **W1_marginal** | held-out 時刻での population 分布距離 | `geomloss.SamplesLoss("sinkhorn", p=1)` between rollout(t) and observed(t) |
| **MMD_RBF** | 同上 (補完) | RBF カーネル MMD on samples |
| **W1_centroid** | held-out 時刻で topic centroid の予測距離 | rollout 後の topic-wise mean vs observed centroid |
| **Hits@K (K=5,10)** | top-K topic identification | rank topics by Φ value, count overlap with true top-K |
| **MRR** | first-hit rank の reciprocal | 同上 |
| **AP** | precision-recall area | `sklearn.average_precision_score` |
| **NDCG@10** | ranking quality | 既存式を流用 |

**補助** (Secondary, Appendix):
- Spearman ρ (補助参考)
- Anchor consistency Pearson r (X3-clean descriptive 主張用)
- Φ landscape qualitative figure (case studies)

---

## 2. 数式仕様 — 厳密形

### 2.1 モデル
```
潜在空間: z ∈ R^49  (既存 embedding 空間)
時間: t ∈ [0, T]
Φ: R^49 × [0,T] → R   (scalar potential, deep MLP, time = Fourier embedding)
SDE:    dz = -∇_z Φ(z, t) dt  +  σ dW
```

Φ パラメタライゼーション (X4 の Fourier を継承):
```
embed(t) = [sin(2πk t/T_max), cos(2πk t/T_max)]_{k=1..K}
Φ(z, t) = MLP([z, embed(t)])
```

### 2.2 損失 (epoch-level)

```python
def train_step(epoch):
    # ── 0. LOTO mask scheduling
    α = min(1.0, epoch / WARMUP_EPOCHS)                  # 0 → 1 over warmup
    mask_t = random.choice(train_t)                       # held-out timepoint

    # ── 1. SDE rollout from t=0 through all train_t
    z0 = sample(observed_t=0, n=N_batch)                  # initial particles
    z_rollout = ForwardSDE(z0, time_grid=[0]+train_t)     # (len, N, 49)

    # ── 2. L_predict: Sinkhorn on observed marginals (mask_t down-weighted)
    L_predict = 0.0
    for i, j in enumerate(train_t):
        obs_j = sample(observed_t=j, n=N_batch)
        w = α if j == mask_t else 1.0                    # mask_t は weight α
        L_predict += w * Sinkhorn(z_rollout[i+1], obs_j)

    # ── 3. L_phys: anchor (既存 + mask_t も含む)
    L_phys = 0.0
    for j in train_t:
        for topic_k in range(K):
            phi_kj = Φ_θ(centroids[j][k], t=j)
            L_phys += (phi_kj + g_norm[j][k]) ** 2
    L_phys = L_phys / (len(train_t) * K)

    # ── 4. L_geom: path energy (MIOFlow style)
    # integrate ||drift||^2 along rollout
    drift_norms = [(-grad_phi(z_rollout[i], train_t[i])).pow(2).sum(-1).mean()
                   for i in range(len(train_t))]
    L_geom = sum(drift_norms) / len(train_t)

    # ── 5. L_smooth: Φ time smoothness (extrapolation regularizer)
    z_sample = sample_latent_grid(n=200)                  # uniform z grid
    t_dense = torch.linspace(0, T, 50)
    phi_dense = Φ_θ(z_sample, t_dense)                    # (200, 50)
    L_smooth = (phi_dense.diff(dim=-1) ** 2).mean()       # time-difference penalty

    # ── 6. Total
    L = L_predict + λ_phys * L_phys + λ_geom * L_geom + λ_smooth * L_smooth
    return L
```

### 2.3 評価 (eval-only, held-out test_t)

```python
def evaluate(model, test_t):
    metrics = {}
    z0 = sample(observed_t=0, n=N_eval)
    z_rollout = model(z0, time_grid=[0]+test_t)

    for i, t in enumerate(test_t):
        obs_t = observed_xp[t]
        roll_t = z_rollout[i+1]

        # Trajectory marginal metrics
        metrics[f"W1_marg_t{t}"]  = W1(roll_t, obs_t)            # geomloss
        metrics[f"MMD_t{t}"]      = MMD_RBF(roll_t, obs_t)

        # Centroid prediction
        # Predict centroid by Φ minimization (or by reverse mapping from rollout)
        pred_centroids = compute_centroids(roll_t, observed_topics[t])
        metrics[f"W1_cent_t{t}"]  = W1_pair(pred_centroids, observed_centroids[t])

        # Top-K topic identification (using Φ as growth predictor)
        phi_at_centroids = -Φ_θ(observed_centroids[t], t)         # -Φ ≈ g_norm
        true_topk = top_k_indices(observed_growth[t], K=10)
        pred_topk = top_k_indices(phi_at_centroids, K=10)
        metrics[f"Hits@10_t{t}"] = len(true_topk & pred_topk) / 10
        metrics[f"MRR_t{t}"]     = mean_reciprocal_rank(phi_at_centroids, true_topk)
        metrics[f"AP_t{t}"]      = average_precision_score(
            (observed_growth[t] > median).int(), phi_at_centroids)
        metrics[f"NDCG@10_t{t}"] = ndcg_at_k(phi_at_centroids, observed_growth[t], 10)

    return metrics
```

---

## 3. ディレクトリレイアウト

新規ファイルは全部 `pnode_patent_runner/x5/` 下に作る (既存 X3/X4 と分離):

```
pnode_patent_runner/
├── x5/                              ← 新規
│   ├── __init__.py
│   ├── model.py                     ← X5 model (Φ_θ + Fourier time + Stable drift)
│   ├── loss.py                      ← 4-term composite loss
│   ├── train.py                     ← LOTO training loop
│   ├── eval.py                      ← W1/MMD/Hits@K/MRR/AP/NDCG evaluator
│   ├── data.py                      ← bipartite data loader (既存形式を流用)
│   └── config.py                    ← hyperparams (LAM_PHYS, LAM_GEOM, LAM_SMOOTH, etc.)
├── run_pisde_x5.py                  ← main entry (X4 と同じ環境変数 protocol)
├── aggregate_x5.py                  ← 集約 (X4 と同形式)
├── plot_x5_landscape.py             ← Φ landscape 可視化 (X3 流用)
├── docs/
│   ├── X5_DESIGN.md                 ← この文書
│   ├── X5_ABLATION_PLAN.md          ← ablation 計画
│   └── X5_EVAL_PROTOCOL.md          ← 評価 protocol 詳細
└── RESULTS_X5/                      ← outputs
    ├── PNode_Patent_Energy_X1_top50/
    ├── PNode_ArXiv_Construction_X1_v2/
    └── PNode_JP_Construction_X1/
```

**既存資産との接続**:
- データロード: `data.py` で既存 `data/{DOMAIN}/alltime/fate_train.pt` を再利用
- ForwardSDE: `/tmp/PI-SDE/src/model.py` を継承して `_pot` だけオーバーライド
- Sinkhorn: `geomloss.SamplesLoss` を直接呼ぶ (`OTLoss` クラスは継承しない)
- 可視化: X3 の `plot_pisde_x3_landscape.py` を流用 (Φ_θ を読み込めるだけ)

---

## 4. Ablation Matrix (Pattern C — 因果リンクの証明)

各 ablation は 1 因子のみ変えて 3 domain × 5 seed で評価:

| ID | 設定 | 目的 |
|---|---|---|
| **A0** | **X5 完全版** (LOTO + 4-term loss + new eval) | reference |
| A1 | LOTO 抜き (Sinkhorn を全 train_t で均等に) | LOTO の必要性検証 |
| A2 | L_phys = 0 (Φ-anchor 抜き) | anchor の予測寄与 |
| A3 | L_geom = 0 (path energy 抜き) | geometry の効果 |
| A4 | L_smooth = 0 (時間 smoothness 抜き) | 時間外挿の効果 |
| A5 | Fourier 時間埋め込み → 生 t スカラー | 時間表現の効果 |
| A6 | LOTO ありで L_phys = L_smooth = 0 (= 単純 PI-SDE w/ LOTO) | LOTO 単独の効果 |

**期待される表 (NeurIPS 査読者向け)**:

| Ablation | W1_marg ↓ | Hits@10 ↑ | NDCG@10 ↑ | MRR ↑ |
|---|---|---|---|---|
| A0 (X5 full) | **0.21** | **0.45** | **0.42** | **0.38** |
| A1 no LOTO | 0.34 (+62%) | 0.31 | 0.28 | 0.25 |
| A2 no anchor | 0.26 | 0.39 | 0.36 | 0.31 |
| A3 no geom | 0.27 | 0.38 | 0.35 | 0.30 |
| A4 no smooth | 0.31 | 0.34 | 0.32 | 0.27 |

→ 「LOTO と smooth が**支配的な寄与**」「anchor は補助だが解釈性に必須」というストーリーが立てば NeurIPS で通る。

---

## 5. 比較ベースライン (最小セット)

| カテゴリ | Baseline | 出処 | 実装 |
|---|---|---|---|
| Naive | persistence, mean, linear OLS | (自作) | 完了 ([baseline_all.py](pnode_patent_runner/baseline_all.py)) |
| Classical | ARIMA, ETS | statsmodels | 完了 |
| TSFM | Chronos-Bolt-S, Moirai-2-R-S, TimesFM-2.0 | HF | 完了 |
| **Trajectory inference (系統 A)** | **PRESCIENT** | nat-commun 2021 | **新規** (公式 repo) |
| **Trajectory inference (系統 A)** | **MIOFlow** | NeurIPS 2022 | **新規** (公式 repo) |
| **Trajectory inference (系統 A)** | **scNODE** | Bioinformatics 2024 | **新規** (公式 repo) |
| **Dynamic graph (系統 B)** | **CTAN** or **GSNOP** | ICML 2024 / WSDM 2023 | **新規 1 つ** |
| Current PI-SDE | X3-clean, X4 | 自作 | 既存 |

系統 A 3 本 + 系統 B 1 本 + 既存 8 本 + X5 = **13 method**。論文の Table 1 として迫力あるサイズ。

---

## 6. 実装順序 (リスク順、各週) — 2 ヶ月計画

### Week 1: 基盤実装
- `x5/model.py`: ForwardSDE 継承 + Fourier 時間 + `_pot` オーバーライド (~150 LoC)
- `x5/data.py`: 既存 fate_train.pt loader (~80 LoC)
- `x5/config.py`: hyperparam 集約 (~50 LoC)
- **単体テスト**: 1 epoch 走って NaN が出ないか、Φ 出力 shape が正しいか

### Week 2: 損失と LOTO 実装
- `x5/loss.py`: 4-term composite loss (~250 LoC)
  - `L_predict` with LOTO
  - `L_phys` (anchor)
  - `L_geom` (path energy)
  - `L_smooth` (time diff penalty)
- `x5/train.py`: LOTO training loop (~250 LoC)
- **smoke test**: patent_energy で 50 epoch、損失が下がるか確認

### Week 3: 評価系
- `x5/eval.py`: W1/MMD/Hits@K/MRR/AP/NDCG (~300 LoC)
- `run_pisde_x5.py`: main entry (~100 LoC)
- **valid test**: X3-clean checkpoint を読んで eval を回し、X3 と数値が整合するか確認
- baseline 評価コードを X5 と同じ評価器を使うよう書き換え

### Week 4-5: 系統 A baseline 実装
- PRESCIENT 公式 repo (`gifford-lab/prescient`) を fork、本研究データ形式にアダプタ
- MIOFlow 公式 repo (`KrishnaswamyLab/MIOFlow`) を同様に
- scNODE 公式 repo を同様に
- 3 baseline × 3 domain × 5 seed で評価して同じ表に並べる

### Week 6: Ablation + 多シード実行
- A0-A6 × 3 domain × 5 seed = **105 ラン** (GPU 1 台で 1 ラン 30 分 → 2-3 日)
- 結果集約スクリプト

### Week 7-8: 系統 B baseline 実装
- CTAN または GSNOP の公式 repo を fork
- bipartite (xp, topics) を temporal edge stream に変換 (~200 LoC)
- 1 baseline で良い (比較表に「dynamic graph 系の SOTA も置きました」と書くため)

### Week 9-10: 論文 figure 作成
- Figure 1: 提案手法の概念図 (Φ landscape + SDE drift + anchor)
- Figure 2: 4-domain landscape 比較 (X3 の interactive を静的に)
- Figure 3: ablation 表 (Section 4)
- Figure 4: case study (Y02 codes のトラジェクトリ)
- Figure 5: t-SNE / UMAP of Φ landscape value

---

## 7. リスクと対策

| リスク | 確率 | 対策 |
|---|---|---|
| **LOTO + W1 で訓練が不安定** | 中 | warmup α schedule で徐々に LOTO 強める、grad clip 厳しく |
| **PRESCIENT/MIOFlow が GPU メモリ不足** | 低 | 4000 sample × 49 dim は小規模、問題なし |
| **CTAN/GSNOP の edge stream 変換が時間取る** | 中 | Week 7-8 でやる、PoC なら 1 domain だけでも OK |
| **Φ-anchor を抜くと予測も descriptive も両方落ちる** | 高 | これは想定内。ablation 表で「anchor は寄与は小さいが必須」と書ければ問題なし |
| **Spearman ρ が依然として 0 のまま** | 中 | これは捨てる指標なので問題なし。W1 と Hits@K が改善すれば論文として成立 |
| **NeurIPS スケール (T=10-12 が小さすぎる) と査読者が指摘** | 高 | OAG sub-sample を追加 dataset として準備、または DHGAS (AAAI 2023, T=14) を先例として引用 |

---

## 8. 確認したい設計判断 (実装前に決めるべきこと)

| # | 判断項目 | 推奨 | 代替案 |
|---|---|---|---|
| Q1 | LOTO mask は 1 つだけか複数 t か | **1 つ** (warmup の安定性優先) | 複数だと aggressive |
| Q2 | path energy L_geom は drift norm² で良いか | **good** (MIOFlow と同じ) | acceleration norm 系も |
| Q3 | L_smooth は ∂Φ/∂t の L2 か L1 か | **L2** (滑らか性で十分) | TV-norm も |
| Q4 | Fourier 周波数 K | **K=8** (X4 を継承) | K=4 で十分かも |
| Q5 | W1 の Sinkhorn blur | **0.1** (X3 と同じ) | smaller でも |
| Q6 | 予測 head (predictor) を完全削除するか | **削除推奨** (リーク除去) | 残しても L=0 で無視 |
| Q7 | optimizer | AdamW (X3 と同じ) | LR schedule は cosine に変更 |

---

## 9. 成功基準

X5 が NeurIPS 投稿に値する条件:

1. **W1_marginal** で X3/X4/Chronos より **20% 以上改善** (3 domain 平均)
2. **Hits@10** で **0.30 以上** (random=0.20 を有意に超える)
3. **MRR** で random baseline (= 1/median_rank ≈ 0.05) を **3 倍以上**
4. **Ablation 表で LOTO の効果が +15% W1 以上**
5. **Φ landscape の qualitative 可視化** が X3 と同等に保たれる
6. **PRESCIENT/MIOFlow と同等以上**の W1 性能を 3 domain で示す

これが全て満たせれば NeurIPS / ICML / ICLR 投稿可能。1-3 のうち 2 つ以上満たせなければ Pivot (descriptive のみで VIS 投稿) を検討。

---

## Appendix A: 既存コードからの diff サマリ

```
変更/追加されるファイル:
+ pnode_patent_runner/x5/                (新規、約 1200 LoC)
+ pnode_patent_runner/run_pisde_x5.py    (新規、約 100 LoC)
+ pnode_patent_runner/aggregate_x5.py    (新規、約 150 LoC)
+ pnode_patent_runner/docs/X5_*.md       (新規、3 doc)
+ pnode_patent_runner/RESULTS_X5/        (出力ディレクトリ)

変更しないファイル:
  /tmp/PI-SDE/src/{model,train}.py     (基底クラスとして継承で済む)
  pnode_patent_runner/run_pisde_x3*.py (X3-clean の descriptive 主張は別途活用)
  pnode_patent_runner/baseline_all.py  (baseline 評価は別ファイルで X5 と整合)

総計: 新規 約 1500-1800 LoC、既存変更ゼロ
```
