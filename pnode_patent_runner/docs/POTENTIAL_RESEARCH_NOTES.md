# ポテンシャル・リンク尤度・速度場（研究メモ）

実装プラン（診断・密度校準・文献整理・案C 設計）に基づくメモ。主表の数値そのものではなく、**定式化の対応**と**拡張時のインターフェース**を固定する。

---

## 1. ボルツマン型と現行ロジットの対応

### 現状（CoPE デコーダ）

エッジ \((i,j)\) に対しスカラー **logit** を作り **Bernoulli** とみなす（`sigmoid(logit)`）。幾何項＋`w_pot * (Φ_i + Φ_j)` は **エネルギー差分に相当するバイアス**として加算可能だが、**配分関数 Z** は明示していない。

### ボルツマン（エネルギーモデル）

ペアエネルギー \(E_{ij}\) に対し \(P(\text{edge}=1 \mid i,j) \propto \exp(-E_{ij}/T)\)。**二部グラフ全体**で正規化する場合、\(Z\) は全ペアにわたる和となり計算コストが大きい。

### 実務的な中間案

| 手法 | 内容 |
|------|------|
| **温度付きロジット** | 既存の `cosine_logit_scale` や `r`、学習可能スケールを **\(1/T\)** と解釈。実装変更は最小。 |
| **ノイズ対照推定 (NCE)** | 負例分布からのサンプルと正例の対数オッズを推定。現在の **hard negative サンプリング**は NCE 風の近似に近い。 |
| **InfoNCE / 対照学習** | 正例1＋負例 K の softmax。リンク予測では **future/recon の負例本数**を K として解釈できる。 |

**文献・キーワード**: Energy-based models (LeCun et al. tutorial), Noise Contrastive Estimation (Gutmann & Hyvärinen), contrastive link prediction in graphs.

---

## 2. 案 C: 非保守速度場 \(v = -\nabla\Phi + h\)

### 動機

[`ACCURACY_POTENTIAL_VIZ_DESIGN.md`](ACCURACY_POTENTIAL_VIZ_DESIGN.md) の通り、**純粋な勾配流**では系列の残差を表現しきれない場合がある。\(h\) で **回転成分・履歴依存**を吸収し、**Φ は解釈用（保守場）**に残す設計が可能。

### 提案インターフェース（設計のみ）

| コンポーネント | 役割 |
|----------------|------|
| **Φ** | 既存 `PotentialNet` / `TimeDependentPotentialNet`。可視化・`−∇Φ` 矢印の主役。 |
| **h** | `h(z, z_hist_summary)` の小型 MLP。入力は **現在潜在**と **過去 L ステップの要約**（平均プールや GRU 1 ステップ）。 |
| **ODE** | `dz/dτ = -α tanh(∇Φ) + β h`（スケールは学習可能）。 |

### 損失

- **リンク尤度**は現行どおりデコーダ経由（Φ を共有するか、デコーダ専用 Φ_dec を分けるかは消融で決める）。
- **潜在整合**: 既存の `latent_pred_loss` を **\(z_{t+1}^{\text{pred}}\)**（新 ODE）と **\(\mu_{t+1}\)** の MSE に。
- **軌道**: `h` が入ると **\(v\)** と教師変位のコサイン整合を **合成速度**で取る（`trajectory_loss` の一般化）。

### 可視化の分岐

- **ヒートマップ・等高線**: **Φ のみ**（保守場ポテンシャル）を表示するモードを既定にし、キャプションで「矢印は **−∇Φ** 成分のみ」または「**v 全体**」を明示するフラグを追加する想定。

---

## 3. 診断ツール

- **`last_epoch_train_breakdown`**: `run_benchmark_comparison` の各 `results[]` に、最終エポック平均の **recon / kl / latent_pred / future_link / potential / trajectory**（重み付き後）を記録。
- **`python -m pnode_patent_runner.diagnose_loss_breakdown`**: JSON の glob 集計、または `--smoke-train --smoke-td-compare` で短い CoPE TD 比較。

---

## 4. 密度校準（`CalibratedPotentialNet`）

**Φ ≈ φ_nn − w·log p_hist(z)** は [`models.CalibratedPotentialNet`](models.py) に実装済み。CLI は `--cope-density-calibrated`（TD との同時利用は README の制約に従うこと）。

### スモーク実測（同一条件・2 epoch・arxiv・ホールドアウト 2026）

同一 CSV・seed 42・`--min-patents` 既定の比較例（環境によりグラフ規模は変わりうる）:

| 設定 | HO AUC (2025→2026) | 備考 |
|------|-------------------|------|
| ベース CoPE | 0.5843 | `diagnose_cope_baseline_ho2026_seed42.json` |
| `--cope-density-calibrated` | 0.5881 | `diagnose_cope_density_calibrated_ho2026_seed42.json` |

epoch 数が少ないため主表用ではない。**`last_epoch_train_breakdown`** で potential / future の比率を JSON から追跡可能。
