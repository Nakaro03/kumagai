# 主実験の統計・事前登録（査読用）

本リポジトリの主表は **複数シード**の `run_benchmark_comparison` の JSON を集約し、**平均 ± 標準誤差（SE）**を報告する。集約には [`aggregate_benchmark_seeds.py`](../aggregate_benchmark_seeds.py) を用いる。

**長期（多ホライズン）優位の事前登録テンプレ**（\(\mathcal{K}\)、\(b^\star\)、Holm 対象の 1 表）: [LONG_HORIZON_PREREGISTRATION.md](LONG_HORIZON_PREREGISTRATION.md)。

---

## 事前登録（論文 Methods にそのまま貼れる）

1. **シード**: `S = \{42, 43, 44, 45, 46\}` の **5 本**（変更する場合は投稿前に固定し、付録に変更履歴を書かない）。  
2. **主指標**: ホールドアウト時は **`final_val_auc`** を **主指標**とする（副指標: `final_val_ap`、`final_val_ece`）。多ホライズンがある場合は **`final_metrics_by_horizon_gap["k"]["auc"]`** を別表で同様に集約（`aggregate_benchmark_seeds --horizon-gap K --horizon-split final`）。  
3. **対照**: 各シードで **同一データ・同一 CLI（HPO は `--optuna-best-json-map` で手法対称）** を保証する。  
4. **主要比較**: **ペア差** \(d_s = \mathrm{AUC}_{\mathrm{pnode},s}-\mathrm{AUC}_{\mathrm{base},s}\)（`base` ∈ `static`, `neural_ode`, `rnn`）について **片側 Wilcoxon 符号付順位検定**（PNODE が大きい方向）。  
5. **多重比較**: baseline が $m$ 個なら **Holm–Bonferroni** で有意水準を補正（$\alpha=0.05$）。多ホライズン $k$ 本を同時に主張する場合は **$k$ も補正対象に含める**か、**主指標を $k=1$ のみ**に事前限定する。  
6. **効果量**: Cohen’s $d$ または **ブートストラップ 95% CI**（ペア差）を付録に 1 表。

---

## 棄却規準（例）

- 補正後いずれの baseline に対しても **$p \ge 0.05$** かつ CI が 0 を跨ぐ → **H2（NODE 優位）主張は採用しない**。  
- シード間で符号がばらつく → **感度分析表**に落とし、強い主張を避ける。

---

## 実装コマンド

```bash
# 例: 特許・ホールドアウト・5 シードの JSON を集約
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed42.json \
  pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed43.json \
  --markdown
```

`--glob` で複数 JSON を指定可能。PNODE と baseline の **同一シード・同一ファイル順**のペアに対する片側 Wilcoxon は例えば次のとおり。

```bash
python -m pnode_patent_runner.aggregate_benchmark_seeds \
  --glob "pnode_patent_runner/outputs/cope_benchmark/benchmark_patent_seed*.json" \
  --markdown \
  --paired-pnode-vs neural_ode
```

Holm 補正は baseline 本数に応じて表計算または別スクリプトで行う（本スクリプトは raw p のみ）。
