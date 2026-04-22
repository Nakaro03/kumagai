# 長期 future-link 優位（H_long）の事前登録テンプレート

曖昧な主張「PNODE は長期トレンド予測で既存手法より優れる」を、**棄却可能な 1 本の主張**に落とすための固定表。数式・統計手順の解説は [STATS_PREREGISTRATION.md](STATS_PREREGISTRATION.md)、タスク操作化は [TREND_PREDICTION_EXPERIMENT.md](TREND_PREDICTION_EXPERIMENT.md)。

**原則**: 本文の主表は **`--holdout-test-year`** ありの JSON の **`final_metrics_by_horizon_gap`**（`--horizon-split final`）を用いる。短期 \(k=1\) の主張と混ぜない。

---

## 1. 事前登録表（論文 Methods / Supplementary にそのまま貼る）

| 項目 | 記号・内容 | 本リポジトリのテンプレ固定（投稿前に実データに合わせて 1 行だけ書き換え） |
|------|------------|-----------------------------------------------------------------------------|
| ホライズン集合 | \(\mathcal{K}\) | \(\{2, 3\}\)（**主張を 1 本に絞る**なら \(\{2\}\) のみにし、Holm 対象を減らす） |
| 「長期」の定義 | \(k\) | `sorted(graphs.keys())` の **インデックス差**（暦年差ではない） |
| 主指標（1 のみ） | \(A_{d,s,m,k}\) | `final_metrics_by_horizon_gap[str(k)]["auc"]`（ROC-AUC） |
| 副指標（探索・補正なし可） | — | 同辞書の `ap` / `ece`（本文では「参考」と明記） |
| 主対照 | \(b^\star\) | `neural_ode`（任意 NODE 対 勾配流の境界） |
| 副次対照（Holm に入れるか事前決定） | \(b\) | `static`, `rnn`（\(\mathcal{K}\) と掛け算すると比較数が増える） |
| ドメイン | \(d\) | `patent`（主結果 1 ドメインに限定するのが推奨） |
| シード集合 | \(S\) | \(\{42,43,44,45,46\}\)（5 本） |
| データ・分割 | — | CSV パス相当、`--data-domain`、`--year-range`、`--holdout-test-year Y` を JSON と一致させて記載 |
| HPO | — | 各手法 **同一 trial 数**・`--optuna-best-json-map` 等で対称化した旨 |
| **Holm–Bonferroni の対象** | — | **事前に列挙**。例 A: 主張 1 本のみ → \((k=2, b^\star=\texttt{neural\_ode})\) の **Wilcoxon 1 本だけ**（補正不要）。例 B: baseline 3 本 × \(k\in\{2,3\}\) の **6 検定**を同時主張に含める → 6 本すべて Holm の順位付け対象。 |
| 有意水準 | \(\alpha\) | 0.05 |
| 検定 | — | ペア差 \(D_{d,s,k}(b)=A_{d,s,\mathrm{pnode},k}-A_{d,s,b,k}\) の **片側 Wilcoxon**（PNODE が大きい方向） |
| 棄却規準 | — | 補正後 \(p\ge\alpha\)、またはペア差の 95% CI が 0 を含む、等 → [STATS_PREREGISTRATION.md](STATS_PREREGISTRATION.md) の棄却節 |

**H_long（記述例・1 本に絞った最小主張）**  
固定した \((d, k, b^\star)=(\texttt{patent}, 2, \texttt{neural\_ode})\) について、シード \(s\in S\) にわたる \(D_{d,s,2}(b^\star)\) に対し片側 Wilcoxon で \(p<\alpha\) かつ効果量の CI が 0 を超える。

---

## 2. 集約・検定コマンド（`aggregate_benchmark_seeds`）

`--eval-horizon-gaps` 付きでベンチを回した各シード JSON に対し、ホライズン \(k\) の AUC を集約し Wilcoxon まで一発で出す。

```bash
# ホールドアウト側・インデックス差 k=2 の AUC を集約し、pnode vs neural_ode
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed*.json" \
  --horizon-gap 2 \
  --horizon-field auc \
  --horizon-split final \
  --markdown \
  --paired-pnode-vs neural_ode
```

`k=3` や `train` 分割（感度）も同様に `--horizon-gap` / `--horizon-split` を変える。

---

## 3. 実装キー対照（対照群）

| 実装 `key` | 役割 |
|------------|------|
| `pnode` | 提案 |
| `neural_ode` | 主対照（推奨 \(b^\star\)） |
| `static` / `rnn` | 副次（Holm 対象に入れる場合のみ主張に含める） |

外部実装ベースラインを主張に含める場合は [EXTERNAL_BASELINE_PLAN.md](EXTERNAL_BASELINE_PLAN.md) を別途満たすこと。
