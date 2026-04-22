# 仮説検証計画: 「PNODEは長期トレンド予測において既存手法より優れる」

元仮説は曖昧であり、そのままでは科学的に検証不可能。以下で 5 つの検証可能な下位仮説に分解し、各々について実験設定・棄却条件を固定する。

一次実装: [`unified_training.py`](../unified_training.py)（`rollout_z_pred_multistep`, `evaluate_future_link_metrics_for_horizon_gaps`, `evaluate_latent_rollout_metrics_for_horizon_gaps`）。統計プロトコル: [STATS_PREREGISTRATION.md](STATS_PREREGISTRATION.md)、[LONG_HORIZON_PREREGISTRATION.md](LONG_HORIZON_PREREGISTRATION.md)。

---

## 0. 用語の操作的定義（曖昧さの排除）

| 用語 | 操作的定義 | 実装上の対応 |
|------|-----------|-------------|
| **長期** | `sorted(graphs.keys())` 上のインデックス差 k >= 2（暦年差ではない） | `--eval-horizon-gaps 1,2,3` |
| **トレンド予測** | 将来リンク（future-link）の予測精度。「あるトピックに著者が関与するか」の二値分類 | `evaluate_future_link_metrics_for_horizon_gaps` の ROC-AUC |
| **優れる** | 同一シード・同一データで、ペア差の片側 Wilcoxon 検定で p < 0.05（Holm 補正後） | `aggregate_benchmark_seeds --paired-pnode-vs` |
| **PNODE** | `BenchmarkTemporalVGAE(variant="pnode")` with B+D: Phi=mlp, K=4, fuse=gru | `--methods pnode --pnode-potential-feature mlp --pnode-history-len 4 --pnode-hist-fuse-mode gru` |
| **既存手法** | static, rnn, neural_ode（同一エンコーダ・同一損失枠・同一HPO予算） | `--methods static,rnn,neural_ode,pnode` |

---

## 1. 仮説の分解（5 本の検証可能な下位仮説）

### H1: 長期 future-link AUC における PNODE vs Neural ODE（主仮説）

**定式化:**

ドメイン d、シード s、インデックス差 k において:

```
D_{d,s,k} = AUC_{pnode,d,s,k} - AUC_{neural_ode,d,s,k}
```

H1_0 (帰無): median(D_{d,s,k}) <= 0 （PNODEはNeuralODE以下）
H1_1 (対立): median(D_{d,s,k}) > 0  （PNODEがNeuralODEを上回る）

**検定**: k=2 に固定し、シード S={42,43,44,45,46} での D_{d,s,2} に対する片側 Wilcoxon 符号付順位検定。

**主指標**: `final_metrics_by_horizon_gap["2"]["auc"]`（ホールドアウト分割）

**棄却条件**: p >= 0.05 または 5 シード中 D > 0 が 3 本以下

---

### H2: 長期での劣化率が PNODE < Neural ODE（相対的ロバスト性）

**定式化:**

劣化率を以下で定義:

```
R_{m,d,s} = AUC_{m,d,s,k=1} - AUC_{m,d,s,k=2}
```

（k=1 から k=2 への AUC の落ち幅。R が大きいほど劣化が激しい）

H2_0: median(R_{pnode} - R_{neural_ode}) >= 0 （PNODEの劣化がNeuralODE以上）
H2_1: median(R_{pnode} - R_{neural_ode}) < 0  （PNODEの劣化がNeuralODEより小さい）

**意味**: AUCの絶対値が同じでも、k を伸ばしたときの「落ち方」が緩やかなら長期に強いと言える。

**検定**: ペア差 (R_{pnode,s} - R_{neural_ode,s}) に対する片側 Wilcoxon。

**棄却条件**: p >= 0.05 または効果量 Cohen's d < 0.2

---

### H3: 潜在ロールアウトの精度（MSE / 方向一致率）

**定式化:**

```
MSE_{m,d,s,k} = mean_i || z_rollout_{i} - mu_encoder_{i} ||^2   (active nodes)
DirAcc_{m,d,s,k} = sign一致率（潜在ノルム変化の方向）
```

H3a: MSE_{pnode,k=2} < MSE_{neural_ode,k=2}  （潜在予測がより正確）
H3b: DirAcc_{pnode,k=2} > DirAcc_{neural_ode,k=2}  （トレンド方向がより正確）

**主指標**: `final_latent_metrics_by_horizon_gap["2"]["mse"]` および `["2"]["direction_agreement"]`

**棄却条件**: 
- H3a: 5 シード中 MSE_{pnode} < MSE_{neural_ode} が 3 本以下
- H3b: 5 シード中 DirAcc_{pnode} > DirAcc_{neural_ode} が 3 本以下

---

### H4: PNODE vs RNN+VGAE（短期で劣位でも長期で逆転するか）

**定式化:**

```
Gap_{s,k} = AUC_{pnode,s,k} - AUC_{rnn,s,k}
```

H4_0: Gap_{s,k=2} <= Gap_{s,k=1}  （k が伸びてもPNODEはRNNとの差を縮めない）
H4_1: Gap_{s,k=2} > Gap_{s,k=1}   （k が伸びるとPNODEが相対的に改善する）

**検定**: ペア差 (Gap_{s,k=2} - Gap_{s,k=1}) に対する片側 Wilcoxon。

**意味**: RNNに絶対値で勝てなくても、「長期になるほどPNODEの相対的立場が良くなる」ことを示す。

**棄却条件**: p >= 0.05 またはペア差の中央値 <= 0

---

### H5: Staticベースラインとの差が k とともに拡大する

**定式化:**

```
Margin_{s,k} = AUC_{pnode,s,k} - AUC_{static,s,k}
```

H5_0: Margin_{s,k=2} <= Margin_{s,k=1}  （時間モデリングの付加価値が長期で減少）
H5_1: Margin_{s,k=2} > Margin_{s,k=1}   （時間モデリングの付加価値が長期で増大）

**意味**: 最もナイーブなベースラインとの差が k で広がれば、「時間ダイナミクスの明示的モデリング」の価値が長期ほど大きいことの根拠になる。

**棄却条件**: 5 シード中 Margin が拡大するのが 3 本以下

---

## 2. 実験設定

### 2.1 データ

| ドメイン | CSV | year-range | holdout | 役割 |
|----------|-----|-----------|---------|------|
| author_topic (主) | `data/processed/arxiv_cs_embedded_2020-2026_full.csv` | 2022-2025 | 2025 | 主結果 |
| patent (副) | `notebooks/work/dataset/topic_info3.csv` | 2015-2020 | 2020 | 汎化確認 |

### 2.2 手法と固定パラメータ

| 手法 | key | 固定パラメータ |
|------|-----|---------------|
| Static | `static` | — |
| RNN+VGAE | `rnn` | `--rnn-history-len 4` |
| Neural ODE | `neural_ode` | `--pnode-history-len 4` (hist fuse有効時) |
| **P-NODE (B+D)** | `pnode` | `--pnode-potential-feature mlp --pnode-history-len 4 --pnode-hist-fuse-mode gru` |

### 2.3 共通設定

```
--epochs 20
--latent-dim 2
--hidden-dim 128
--min-patents 5
--cope-link-score distance
--eval-horizon-gaps 1,2,3
--loss-aux-warmup-epochs 0
```

### 2.4 シード

S = {42, 43, 44, 45, 46}

---

## 3. 対照群（baseline）の設計原理

| 対照 | 何を制御しているか | PNODEとの差分 |
|------|-------------------|---------------|
| `static` | 時間モデリングの**完全な不在** | ODE + Phi + 履歴融合（全部） |
| `neural_ode` | **ポテンシャル構造の不在**（任意ベクトル場） | Phi の勾配流制約のみ |
| `rnn` | **ODE連続時間の不在**（離散系列予測） | 連続ダイナミクス vs 離散LSTM |

`neural_ode` が最も「公平な」対照: 同一エンコーダ・同一デコーダ・同一損失枠で、**ベクトル場の構造制約だけが異なる**。

---

## 4. 有意差検定の方法

### 4.1 主検定

**片側 Wilcoxon 符号付順位検定**（non-parametric, paired）

理由:
- サンプルサイズが 5（正規性の仮定は危険）
- 同一シードのペアデータ（独立ではない）
- 方向性の事前仮説あり（PNODEが「上回る」）

### 4.2 多重比較の補正

H1-H5 のうち**本文の主張に含める仮説の数 m** に応じて Holm-Bonferroni を適用。

推奨: **主張は H1 の 1 本に絞る**（Holm 不要）。H2-H5 は「補助的分析」として本文では p 値を報告するが、「有意」とは書かない。

全 5 本を同時に主張する場合: alpha_holm = 0.05 / 5, 0.05 / 4, ... の順で適用。

### 4.3 効果量

ペア差 D_{s} のブートストラップ 95% CI（10000回リサンプリング）を付録に報告。

---

## 5. 仮説が棄却される条件（具体的な数値基準）

### H1 が棄却される場合（主仮説の失敗）

以下の**いずれか**が成立:

1. Wilcoxon p >= 0.05
2. 5シード中、D_{s,k=2} > 0 が 3本以下（多数決で負け）
3. ペア差の中央値が 0.01 未満（統計的に有意でも実用上無意味）

### H1 が棄却されたときの対処戦略

| 結果パターン | 論文での記述 |
|-------------|-------------|
| k=1 では PNODE > Neural ODE だが k=2 では差なし | 「PNODEの優位は短期に限られる。勾配流制約はロールアウトの誤差累積に対してロバストではない」 |
| k=1 でも k=2 でも差なし | 「ポテンシャル構造の帰納バイアスはリンク予測精度に寄与しない。解釈性（Phiマップ）が主要な付加価値」 |
| k=2 で PNODE < Neural ODE | 「勾配流の保守性制約がロールアウトの自由度を制限し、長期で逆効果。非保守成分（residual）の必要性を示唆」 |
| 全 baseline に負ける | 「PNODEの価値は予測精度ではなく可視化・解釈性にある」と明示。主張を「精度」から「解釈性とのトレードオフ」に切り替え |

---

## 6. 再現コマンド

### 6.1 ベンチマーク実行（5 シード × 2 ドメイン）

```bash
for SEED in 42 43 44 45 46; do
  python -m pnode_patent_runner.run_benchmark_comparison \
    --data-domain author_topic \
    --data data/processed/arxiv_cs_embedded_2020-2026_full.csv \
    --year-range 2022 2025 \
    --holdout-test-year 2025 \
    --min-patents 5 \
    --epochs 20 --seed $SEED \
    --methods static,rnn,neural_ode,pnode \
    --pnode-potential-feature mlp \
    --pnode-history-len 4 \
    --pnode-hist-fuse-mode gru \
    --eval-horizon-gaps 1,2,3 \
    --output-json pnode_patent_runner/outputs/hypothesis_test/benchmark_author_topic_seed${SEED}.json
done
```

### 6.2 集約・検定

```bash
# H1: PNODE vs Neural ODE, k=2, ホールドアウト AUC
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "pnode_patent_runner/outputs/hypothesis_test/benchmark_author_topic_seed*.json" \
  --horizon-gap 2 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode
```

### 6.3 H2（劣化率）の計算

```bash
# k=1 と k=2 の両方を集約し、差分を手動で算出
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "pnode_patent_runner/outputs/hypothesis_test/benchmark_author_topic_seed*.json" \
  --horizon-gap 1 --horizon-field auc --horizon-split final --markdown

python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "pnode_patent_runner/outputs/hypothesis_test/benchmark_author_topic_seed*.json" \
  --horizon-gap 2 --horizon-field auc --horizon-split final --markdown
```

劣化率 R = AUC(k=1) - AUC(k=2) を各シード・各手法で算出し、ペア差の Wilcoxon を計算。

---

## 7. 判定フローチャート

```
H1 検定 (PNODE vs Neural ODE, k=2)
  │
  ├── p < 0.05, median(D) > 0.01
  │     → 「PNODEは長期future-linkでNeuralODEを上回る」と主張
  │     → H2-H5 を補助分析として報告
  │
  ├── p < 0.05, median(D) <= 0.01
  │     → 「統計的有意だが実用的差は小さい。解釈性が主な付加価値」
  │
  └── p >= 0.05
        │
        ├── H2 が成立（劣化率でPNODEが優位）
        │     → 「絶対AUCではなく劣化のロバスト性でPNODEが優れる」に主張を修正
        │
        ├── H4 が成立（RNNとの差がk増加で縮小）
        │     → 「RNNの優位は短期に集中。長期ではPNODEが追いつく」
        │
        └── いずれも不成立
              → 「ポテンシャル勾配流の帰納バイアスは長期予測に寄与しない」を否定結果として報告
              → 論文の主張を「精度」から「解釈可能なエネルギーランドスケープの提供」に転換
```

---

## 関連ドキュメント

- [STATS_PREREGISTRATION.md](STATS_PREREGISTRATION.md) — 統計プロトコルの詳細
- [LONG_HORIZON_PREREGISTRATION.md](LONG_HORIZON_PREREGISTRATION.md) — H_long の事前登録テンプレート
- [TREND_PREDICTION_EXPERIMENT.md](TREND_PREDICTION_EXPERIMENT.md) — タスクの操作化定義
- [PNODE_PAPER_FRAMING.md](PNODE_PAPER_FRAMING.md) — 論文全体のフレーミング
- [PNODE_BOTTLENECK_AND_ABLATIONS.md](PNODE_BOTTLENECK_AND_ABLATIONS.md) — アブレーション設計
