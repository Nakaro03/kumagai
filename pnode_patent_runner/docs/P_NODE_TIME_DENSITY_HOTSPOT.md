# P-NODE 改善と「時間つき密度」によるホットスポット／衰退の可視化案

## 前提（ユーザー状況）

- **企業–特許**ドメインで、**他手法との比較および対称 HPO は既に実施済み**とする。
- 以降の焦点は **P-NODE（勾配流 ODE）単体の改善**と、**ポテンシャルを学習 MLP（`PotentialNet`）中心ではなく、時間を考慮した密度で定義する**設計案である。
- ベンチ上 **RNN+VGAE が最強**であることは前提とし、P-NODE が追う目標は「RNN に勝つ」ことだけでなく、**解釈可能性（盆地・流れ・時間変化）と数値の両立**として位置づけると論旨がぶれにくい。

---

## 1. RNN が強い中で P-NODE の精度を上げるには何をするか

勾配流 \(\dot z = -\nabla \Phi(z)\) は **保守場**に制限される一方、RNN は **非保守的な系列写像**を学習しうる。よって「同じ VGAE バックボーン」のまま数値だけ追う場合の現実的なレバーは次のとおり。

| レバー | 内容 |
|--------|------|
| **損失の再配分** | 既に CoPE では補助損失が効くことが確認されている。P-NODE でも **future-link / 軌道整合**の係数を Optuna 対象に含め、**リンク予測に直結する項**を相対的に強める。 |
| **潜在次元・ODE 解像度** | `latent_dim`、ODE の `hidden_dim`、`rtol/atol`、積分ステップの安定性。勾配流は悪条件になりやすい。 |
| **デコーダとの整合** | P-NODE はデコーダに \(\Phi\) を入れない。**幾何項（distance/cosine）と時間発展のスケール**が噛み合っているかを HPO で確認。 |
| **ハイブリッド（任意）** | 厳密な P-NODE を捨てず、\(\dot z = -\nabla\Phi(z) + \epsilon\,h(z)\) の **小さな補正**で系列寄与を足す（論文では定義を明記）。 |

**期待値**: RNN を単体で超えることは難しい場合があり得る。そのときは **「同じパラメータ予算・同じ HPO でどこまで近づくか」**と **解釈図の付加価値**を主張に含める。

---

## 2. 方針: \(\Phi\) を学習 MLP 主ではなく「時間つき密度」で定義する

### 2.1 考え方

各年 \(t\)（または窓）について、エンコーダ出力 \(\mu\)（または \(z\)）の集合から **潜在空間上の密度** \(\hat p_t(z)\) を推定する。

- **スカラーポテンシャル**を **密度から**定義する例:
  \[
  \Phi_t(z) = -\log \hat p_t(z) + \text{（任意の平滑化・下限クリップ）}
  \]
- 勾配流は **\(-\nabla_z \Phi_t(z)\)** とし、**同じ \(\Phi_t\)** を ODE と（必要なら）可視化に使う。

学習可能パラメータを **ゼロ**にするわけではなく、**\(\hat p_t\) の推定器**（帯域幅、混合数、簡易ニューラル密度など）に残る。従来の `PotentialNet` は **\(\Phi_t\) の補正項** \(\psi(z,t)\) として残す選択肢もある:
\[
\Phi_t(z) = -\log \hat p_t(z) + \psi_\theta(z,t).
\]

### 2.2 時間の入れ方（実装しやすい順）

1. **年ごとの KDE / ガウス混合**  
   その年のグラフで得た \(\mu\) だけで \(\hat p_t\) を構築。過去を混ぜない **スナップショット密度**。
2. **累積（因果）密度**  
   \(\hat p_{\le t} =\) 年 \(\le t\) の \(\mu\) 全部で KDE。興味の「累積的な人気」に近い。
3. **指数減衰 EMA**  
   \(\hat p_t \leftarrow \lambda \hat p_{t-1} + (1-\lambda)\,\hat p_{\mathrm{year}\,t}\)。滑らかな時間変化。

---

## 3. ホットスポットと「衰退スポット」の可視化定義

**ホットスポット（注目・流入）**  
- **高 \(\hat p_t(z)\)**、すなわち **低 \(\Phi_t(z) = -\log \hat p_t\)** の山（局所最大）。  
- 地図では **\(\Phi_t\)** のヒートマップで **谷（低エネルギー）**がホットスポットに対応（既存 `valley_red_peak_blue` 系と整合）。

**衰退スポット（かつて盛ん・今は薄い）**  
単年の \(\hat p_t\) だけでは「衰退」は見えにくい。**時間差**を入れる。

| 定義案 | 式のイメージ | 図での見え方 |
|--------|----------------|----------------|
| **密度の減少率** | \(D_t(z) = \log \hat p_{t}(z) - \log \hat p_{t-\Delta}(z)\) | \(D_t < 0\) が大きい所＝衰退スポット候補 |
| **相対順位の下落** | \(\hat p_t(z) / \hat p_{t-\Delta}(z)\) の小さい領域 | 正規化したヒートの第2パネル |
| **「過去の峰」との差分** | \(\hat p_{t-\Delta}\) でトップだった格子が \(\hat p_t\) で下がった格子 | セル単位の変化マップ |

可視化は **2 枚（または 2 レイヤ）**が分かりやすい:  
- **パネル A**: \(\Phi_t\) または \(\log \hat p_t\)（現在のホットスポット）  
- **パネル B**: \(D_t(z)\) または \(\partial_t \log \hat p_t\) の近似（衰退・成長）

---

## 4. P-NODE 実装との接続（リポジトリ観点）

- 現状の P-NODE は [`GradientNeuralODEPredictor`](models.py)＋`PotentialNet`。  
- **密度主体**にする場合は **`forward(z)` が \(\Phi_t(z)\)** を返すモジュールに差し替え、**\(t\)（年インデックス）をバッファまたは引数で渡す**必要がある（[`ACCURACY_POTENTIAL_VIZ_DESIGN.md`](ACCURACY_POTENTIAL_VIZ_DESIGN.md) の案 B と同型）。
- **学習ループ**では、各ステップの \(\mu\) で \(\hat p_t\) を更新し、\(-\log \hat p_t(z)\) を \(\Phi_t\) に使う。`torch.autograd` は **\(z\)** に対して通す（密度のパラメータは **\(z\) について微分可能**な推定器にするか、固定 KDE なら \(\nabla_z \log \hat p_t\) を明示式で書く）。

---

## 5. 検証の順序（短く）

1. **オフライン**: 学習済みエンコード \(\mu_{i,t}\) だけで \(\hat p_t\) を推定し、**パネル A/B のプロトタイプ図**（ホットスポット／衰退）が意味を持つか確認。  
2. **ODE に組み込み**: \(\Phi_t\) を固定または微分可能 KDE にして P-NODE の **1 年先 AUC** が改善するか。  
3. **本文**: 対称 HPO 済みの表はそのまま、**新規は P-NODE 行の改善と解釈図**を追加する形にすると比較が明確。

### 5.1 オフライン可視化（リポジトリ）

[`run_offline_mu_density_maps.py`](../run_offline_mu_density_maps.py) が、**eval 時の encode（＝μ）**から特許ノードの点を取り、**同一グリッド上**で `sklearn` のガウス KDE により \(\log \hat p_t(z)\) と \(D_t(z)=\log \hat p_t-\log \hat p_{\mathrm{ref}}\) を **PNG** 出力する（`ref` は `--delta-years` に合わせて `year-range` 内の過去年から自動選択）。

```bash
cd kumagai   # リポジトリルート
python -m pnode_patent_runner.run_offline_mu_density_maps \
  --data notebooks/work/dataset/topic_info3.csv \
  --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt \
  --year-range 2010 2020 \
  --delta-years 3 \
  --output-dir pnode_patent_runner/outputs/offline_density_mu
```

`--load-checkpoint` は **実在する `.pt` のパス**に置き換える（ドキュメントの `path/to/...` は例ではない）。まだ無い場合は [`run_train_unified_vgae_checkpoint.py`](run_train_unified_vgae_checkpoint.py) で学習し、既定どおり `pnode_patent_runner/outputs/cope_landscape/unified_vgae.pt` を生成する。

`--cope-density-calibrated` 等は学習時と同じアーキなら `run_interactive_landscape_cope_vector_field.py` と同様に付与する。

**`map_cope_alt_dark.html` と同じ UI**で、Φ だけ **密度由来（\(\Phi=-\log\hat p\)、特許 μ の KDE）**に差し替える場合は、[`run_interactive_landscape_cope_vector_field.py`](../run_interactive_landscape_cope_vector_field.py) に **`--phi-source density_kde`** を付ける。ショートカットとして [`run_interactive_landscape_mu_kde_html.py`](../run_interactive_landscape_mu_kde_html.py) は同じモジュールに `--phi-source density_kde` を前置する。ヒート・等高線・**−∇Φ 矢印**（数値勾配）は PotentialNet 版と同型。

---

## 6. リスク（論文に一言）

- 「人気＝高密度」は **規範的仮定**。衰退は **本当の不人気**か **分散の縮小**かを区別しにくい → **定義を Method に固定**し、複数の \(\Delta\) でロバスト性を付録に載せるとよい。
