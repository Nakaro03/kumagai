# ポテンシャル定式の2系（`pnode` / `pnode_explicit`）

実装の主な分岐は [benchmark_vgae.py](../benchmark_vgae.py) の `variant` と [models.py](../models.py) の `potential_net` クラスに対応する。

## 1. 記号の固定（本稿の主系を1つに揃えるときの参照表）

| キー | \(\Phi(z)\) の定義 | 値域の目安 | 時間予測 |
|------|-------------------|------------|----------|
| **`pnode`** | `PotentialNet` → **Tanh 出力**（RFF または MLP 特徴のあと MLP ヘッド） | 約 \((-1,1)\) スカラー/ノード | [GradientNeuralODEPredictor](pnode_patent_runner/models.py) + `CalibratedPotentialNet` 可 |
| **`pnode_explicit`** | [ExplicitAttentionPotentialNet](../models.py): \(\Phi=\mathrm{softplus}(g_\theta(\cdot))\) | \(\mathbb{R}_+\) | [GradientNeuralODEPredictorExplicitAttention](../models.py) |
| **導出量（explicit のみ定義）** | \(A(z)=\exp(-\Phi(z))\in(0,1]\) | 補助 `L_attn` で代理目標（次数等）に接地可 | 同上 |

**同じ** [GradientODEFunc](../models.py) により
\(\frac{dz}{d\tau}=-\tanh(s)\nabla_z\Phi\) を積分する。  
`pnode` の TanhΦ と `pnode_explicit` の非負Φは**可換な物理量ではない**ため、ベンチ表や論文図は**キーごとに**記号表を併記する。

## 2. `pnode_explicit` の RFF オプション

`pnode` と同様、`--pnode-potential-feature {mlp,rff}` を **`pnode_explicit` の \(g_\theta\)** 入力特徴にも流用する（`--pnode-rff-frozen-basis` で RFF 行列 \(B\) の学習可否）。  
RFF: \(p=zB\), \(\tilde x=[\sin p,\cos p]\) のあと、**Tanh なし**のヘッドでスカラー \(g\)、最後に \(\Phi=\mathrm{softplus}(g)\)。

## 3. 損失の拡張（`unified_training.compute_loss_standardized`）

- **`potential_reg_mode`**: `l2`（既定: `0.01·mean(Φ²)`）/ `log1p_sq`（`0.01·mean(log(1+Φ²))`）/ `centered_l2`（\(\Phi\) のバッチ平均を引いてから二乗; 全ゼロ吸引を和らげる消融用）
- **`trajectory_delta_source`**: `z`（再パラメ化 \(z_t\) 基準）/ `μ`（エンコーダ平均 \(\mu_t\) 基準、VAE ノイズ除く）
- **`trajectory_loss_type`**: `cosine` / `smooth1mcos`（`smooth_l1(1-cos)`）/ `huber_vec`（正規化ベクトル差の smooth_l1）
- **`trajectory_grad_floor` + `trajectory_grad_floor_weight`**: \(\|\nabla\Phi\|\) が閾値未満のペナ（平坦場の抑制）
- **ログ**: `grad_phi_l2` = ミニバッチ中アクティブノード上の **mean \(\|\nabla\Phi\|\)**（重み外 diagnostique）

## 4. 消融表テンプレ（掲載用の推奨グリッド）

| 研究項目 | 推奨スイープ | 目的 |
|----------|--------------|------|
| ポテンシャル正則 | `potential_reg_mode ∈ {l2, log1p_sq, centered_l2}` × `λ_pot` | 平坦Φ・A潰れの有無 |
| 軌道 | `trajectory_delta_source ∈ {z, μ}` × `trajectory_loss_type` | コサイン消失・ノイズ汚染 |
| ODE | `pnode_ode_method ∈ {dopri5, rk4}` × `n_steps` | 壁時計 vs AUC、併せて `grad_phi_l2` |

## 5. ワンライン ODE 診断（同一 seed）

```bash
# dopri5（既定）と rk4+分割数を変え、保存 JSON / `history.train_components.grad_phi_l2` を比較
python -m pnode_patent_runner.run_benchmark_comparison \
  --methods pnode_explicit --epochs 5 --seed 42 \
  --pnode-ode-method dopri5

python -m pnode_patent_runner.run_benchmark_comparison \
  --methods pnode_explicit --epochs 5 --seed 42 \
  --pnode-ode-method rk4 --pnode-ode-n-steps 8
```

連続実行例: [`scripts/run_pnode_ode_diagnostics_sweep.sh`](../scripts/run_pnode_ode_diagnostics_sweep.sh)（`DATA` / `DATA_DOMAIN` を設定）。

同様に `pnode`（TanhΦ）でも可。`plot_training_curves` で `grad_phi_l2` が列にあれば曲線化可能。

関連: [PNODE_BOTTLENECK_AND_ABLATIONS.md](PNODE_BOTTLENECK_AND_ABLATIONS.md)、[ARCHITECTURE.md](ARCHITECTURE.md)。
