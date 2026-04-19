# 現状課題を踏まえた新手法案（実装・検証用メモ）

本書はリポジトリのベンチマーク整理と整合する形で、**案1〜案5**の要点・リスク・**ユーザー確認事項**・**論文スコープの推奨**・**消融と公平比較の手順**をまとめたものです（プラン文書の実体化）。実装コードは含みません。

---

## 1. 現状の課題（ベンチマーク JSON 整理との対応）

| 課題 | 含意 |
|------|------|
| **RNN+VGAE が安定的に強い** | 勾配流・単一 \(\Phi\) だけでは、リンク予測で必ずしも上回れない。主張は「SOTA」ではなく条件付きに。 |
| **ドメイン依存** | 特許・ArXiv・著者–トピックで順位が変わる。主表はドメイン別または脚注で条件を揃える。 |
| **学習・HPO と最終指標のギャップ** | Optuna の `best_value` と `run_benchmark_comparison` の `final_val_*` が一致しない場合がある。Setup で手続きを明記。 |
| **学習曲線の不安定さ** | エポック途中で AUC が落ちる例がある。early stopping・複数シード・平均分散の報告を検討。 |
| **比較の公平性** | CoPE のみ HPO だと主張が弱い。可能なら全手法で同一 `n-trials` の対称 HPO。 |

---

## 2. 新手法案の一覧（概要）

### 案1：人気・密度で校準したポテンシャル

観測履歴から推定した密度 \(p_{\mathrm{hist}}(z)\)（またはトピック別の重み付き密度）に対し、

\[
\Phi(z) \approx -\log p_{\mathrm{hist}}(z) + \text{学習可能な補正項}
\]

のように **エネルギーをデータ事前でアンカー**する。勾配流は歴史的に「踏まれやすい」盆地へ向かう解釈が付きうる。

**実装上の接点**：[`models.py`](../models.py) の `PotentialNet` の置換または併用、`UnifiedVGAE` の `decode_logits` および ODE 側の \(\Phi\) 共有の維持。

### 案2：二層ポテンシャル（遅い成分＋速い成分）

\(\Phi(z,t)=\Phi_{\mathrm{slow}}(z)+\Phi_{\mathrm{fast}}(z,t)\) 等で、トレンドと年次変動を分離。RNN が捉える高周波を理論的に残す。

### 案3：ハイブリッド時間ヘッド（勾配流 ODE ＋ 残差 RNN/GRU）

\(z_{t+1} = \mathrm{ODE}(z_t) + \epsilon\,g_{\mathrm{rnn}}(z_{t-k:t})\) のように、解釈は ODE に残しつつ系列残差を別モジュールへ。

### 案4：将来リンク向けの対比学習／ランキング損失

時刻整合の正負例に対する InfoNCE 型や margin ランキングを [`unified_training.py`](../unified_training.py) に追加し、評価指標とのギャップを縮める。

### 案5：条件付きポテンシャル \(\Phi(z \mid c)\)

トピック／分野 \(c\) ごとに盆地が異なる条件付きエネルギー。案1の密度を \(c\) 別に定義しやすい。

---

## 3. To-do：案1「密度」の定義 — **ユーザー確認シート**

**以下を埋めてください（複数選択可）。** これが決まらないと数式・実装規模が確定しません。

| ID | 選択肢 | あなたの想定（はい／いいえ／メモ） |
|----|--------|-----------------------------------|
| D1 | **潜在空間上の経験分布**（各年の \(\mu\) または \(z\) の KDE、正規化流、深層エネルギーモデルなど） | | 
| D2 | **観測グラフ由来**（年次エッジ数・次数・共起から特徴を作り、それを \(z\) 空間に写像した上での「密度」や重み） | | はい | 
| D3 | **外部のトピック人気度**（引用数・出願件数・ダウンロード等の時系列をトピックに結合） | |

**追加の確認（任意だが推奨）**

- 「人気＝低 \(\Phi\)」を **常に仮定**するか、**飽和・下降局面**（ブーム終焉）を別項でモデル化するか。
- 密度は **エポックごとに更新**するか、**事前に一度**推定して固定するか。

---

## 4. To-do：優先ドメインと公平 HPO — **推奨（リポジトリ状況ベース）**

| 項目 | 推奨 | 理由 |
|------|------|------|
| **主表の主ドメイン** | 論文のストーリー次第だが、**公平比較の実績がある `patent`（`topic_info3` 系）を軸**にし、ArXiv は **ホールドアウト**付きで補助、著者–トピックは **ラベル付き**で案5と相性が良い、が定番の組み合わせ。 | `benchmark_patent_symmetric_*_seed42.json` に **全手法・対称 HPO** の列がある。著者–トピックの既存 JSON は **CoPE のみ Optuna** のため、主表の唯一の根拠にはしにくい。 |
| **公平 HPO** | 主表は **`--optuna-best-json-map`** による **各手法同一 `n-trials`** を推奨（[`PAPER_WORKFLOW.md`](../PAPER_WORKFLOW.md) セクション5、[`scripts/run_symmetric_hpo_benchmark.example.sh`](../scripts/run_symmetric_hpo_benchmark.example.sh)）。 | 「CoPE だけ探索」は査読で不利になりやすい。 |
| **ホールドアウト** | ドメイン横断で主表を並べるなら **3 ドメインとも同一の `--holdout-test-year` 方針**（[`PAPER_WORKFLOW.md`](../PAPER_WORKFLOW.md) の推奨）。 | `final_val_*` の意味が揃う。 |

**ユーザーが上書きすべき点**：主貢献が「特許ネットワーク」か「学術コミュニティ」かで、主ドメインと図表の順序が変わる。

---

## 5. To-do：消融・ベンチライン手順（PAPER_WORKFLOW 準拠）

新手法を案1〜案5のいずれか（または併用）で実装した**後**に回す手順の雛形です。

### 5.1 公平な主実験（全ベースライン＋提案）

- エントリポイント：[`run_benchmark_comparison.py`](../run_benchmark_comparison.py)
- 主表用：**対称 HPO** なら `--optuna-best-json-map` に、`cope` / `static` / `rnn` / `neural_ode` / `pnode` 各々の `best_params_*.json` を渡す（[`PAPER_WORKFLOW.md`](../PAPER_WORKFLOW.md)）。
- **必ず論文に書く CLI 情報**：データパス、`--data-domain`、`--year-range` / `--arxiv-year-*`、`--min-patents`、`--epochs`、`--seed`、`--cope-link-score`、使用時は `--holdout-test-year` と **trial 数・探索空間**。

### 5.2 案別の消融（最小セット）

| 提案案 | 消融の例 | 備考 |
|--------|----------|------|
| **案1** | (a) 現行 `PotentialNet` のみ (b) 密度項のみ固定 (c) \(-\log p_{\mathrm{hist}}+\) 学習補正のフル | 「人気=低 \(\Phi\)」の仮定の有効性を切り分け |
| **案2** | \(\Phi_{\mathrm{fast}}=0\) / \(\Phi_{\mathrm{slow}}=0\) / 両方 | 時間スケールの寄与 |
| **案3** | \(\epsilon=0\)（純 ODE）/ ODE なし（純 RNN）/ フル | ハイブリッドの必要性 |
| **案4** | 対比損失の重み 0 / 従来 6 成分のみ / 併用 | 指標とのギャップの縮小を検証 |
| **案5** | 条件なし \(\Phi(z)\) / 条件付き \(\Phi(z\mid c)\) / \(c\) 埋め込みのみ | ラベル情報の使い方 |

### 5.3 既存スクリプトとの役割分担

| 目的 | スクリプト |
|------|------------|
| CoPE の補助損失のオンオフ（特許パイプライン） | [`run_cope_effectiveness.py`](../run_cope_effectiveness.py) |
| 単一手法の HPO | [`run_optuna_unified_vgae.py`](../run_optuna_unified_vgae.py) |
| 多シード・多ドメインの雛形 | [`scripts/paper_benchmark_suite.example.sh`](../scripts/paper_benchmark_suite.example.sh) |
| 対称 HPO → ベンチ一括の雛形 | [`scripts/run_symmetric_hpo_benchmark.example.sh`](../scripts/run_symmetric_hpo_benchmark.example.sh) |

### 5.4 報告上の注意（再掲）

- 指標は **ROC-AUC と AP**、正負例は **サブサンプル手続き**付きで宣言（[`PAPER_WORKFLOW.md`](../PAPER_WORKFLOW.md) セクション2）。
- 2D \(\Phi\) 可視化を使う場合は **解釈用**であり、主表の `latent_dim`・シードと揃えた別実験である旨を明記。

---

## 6. 案1の実装状況（リポジトリ）

**採用した定式化（D1 の簡易版）**: 各エポックのエンコーダ平均 \(\mu\) を **active ノード上で EMA** し、**対角ガウス** \(p_{\mathrm{hist}}(z)\) の \(\log p(z)\) を計算。合成ポテンシャルは

\[
\Phi(z) = \phi_{\mathrm{nn}}(z) - w \log p_{\mathrm{hist}}(z)
\]

（\(w\) は `log_density_weight` として **学習可能**）。ODE とデコーダは従来どおり `potential_net` 経由で **同一 \(\Phi\)** を共有。

| モジュール | パス |
|------------|------|
| `HistoricalDiagonalLogProb` / `CalibratedPotentialNet` | [`models.py`](../models.py) |
| `UnifiedVGAE(..., density_calibrated_potential=...)` | [`unified_vgae.py`](../unified_vgae.py) |
| 学習中の \(\mu\) 更新 | [`unified_training.py`](../unified_training.py) の `train_one_epoch` |

**CLI（`cope` のみ有効なエントリで `--cope-density-calibrated`）**

- [`run_benchmark_comparison.py`](../run_benchmark_comparison.py): `--cope-density-calibrated` / `--cope-density-log-weight` / `--cope-density-ema-momentum`
- [`run_optuna_unified_vgae.py`](../run_optuna_unified_vgae.py): 同上（`--method cope` 時）
- [`run_train_unified_vgae_checkpoint.py`](../run_train_unified_vgae_checkpoint.py)、[`run_cope_effectiveness.py`](../run_cope_effectiveness.py)、可視化 CLI 2 本: 同上

ベンチマーク JSON に `cope_density_*` キーが追記される。

---

## 7. 次のアクション

1. **セクション3の確認シート（D1–D3）**で、本実装（対角ガウス EMA）で足りるか、KDE やグラフ統計が必要かを判断する。
2. **セクション4の主ドメイン**を研究目的に合わせて確定する。
3. 消融（`--cope-density-calibrated` のオンオフ、\(w\) 固定など）を主表に載せる。
