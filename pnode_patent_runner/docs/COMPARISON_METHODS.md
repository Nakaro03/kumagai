# 比較手法（Comparison Methods / Baselines）

本節は、P-NODE 系の実験で用いる**全比較手法**を、(1) リポジトリ内の**実装通り**に記述し、(2) それぞれが由来する**先行研究へ帰属**させ、(3) 実装と正準手法の**差分を明記**する。書誌は Web 照合済み（[References](#references)、DOI/arXiv 付き）。評価は全手法で共通の future-link プロトコル（[PNODE_PAPER_FRAMING.md](PNODE_PAPER_FRAMING.md) §4、[EXPERIMENTAL_SETTINGS.md](EXPERIMENTAL_SETTINGS.md)）に載せる。

> **命名規約**（既存 doc と一致）: 実装キー `static / rnn / neural_ode / pnode / cope`、表示名 "Static / RNN / Neural ODE / P-NODE / CoPE-VGAE"、追加対照 VGRNN / StandardGODE / EvolveGCN-O / ROLAND。

---

## 0. 共有構成要素（全手法で同一）

全手法は**同一のエンコーダと同一の幾何デコーダ**を共有し、**潜在ベクトル場（時間発展則）だけ**が異なる。したがって比較は時間発展則の対照になっている。

- **エンコーダ = 4層 GAT-VGAE**（`SharedVGAEEncoder`, [models.py:15-47](../models.py#L15-L47)）。グラフ注意（GAT; Veličković et al. 2018）でメッセージパッシングし、変分グラフ自己符号化（VGAE; Kipf & Welling 2016）で潜在 \(z=\mu+\sigma\odot\varepsilon\) を得る。深い GAT の over-smoothing 対策として LayerNorm（Ba et al. 2016）と残差接続（He et al. 2016）を各層に挿入。
- **デコーダ = 幾何スコア**（`decode_logits`, [dual_force_vgae.py:64-76](../dual_force_vgae.py#L64-L76) / [benchmark_vgae.py:431-442](../benchmark_vgae.py#L431-L442)）。距離版 `logit = r - ||z_i-z_j||²`、またはコサイン版 `logit = s·cos(z_i,z_j)`。これは VGAE の内積デコーダ（Kipf & Welling 2016）の距離／角度版であり、**ポテンシャル \(\Phi\) をデコーダに入れない**（`cope` を除く）。

**English (draft).** *All methods share a 4-layer GAT-VGAE encoder (Veličković et al., 2018; Kipf & Welling, 2016) and a geometric (distance/cosine) link decoder, and differ only in the latent temporal-evolution rule, so the comparison isolates the dynamics model.*

### 一覧表

| キー | 実装（クラス, file:line） | 中核更新則 | 由来（帰属先） |
|---|---|---|---|
| `static` | `StaticLatentPredictor` [models.py:820](../models.py#L820) | \(z_{t+1}=z_t\)（persistence） | 記憶／持続ベースライン: Poursafaei et al. 2022 (EdgeBank) |
| `rnn` | `RNNLatentPredictor` [models.py:1087](../models.py#L1087) | \(z_{t+1}=W\,\mathrm{LSTM}(z_{t-K+1:t})_{[-1]}\) | VGRNN: Hajiramezanali et al. 2019; LSTM: Hochreiter & Schmidhuber 1997; GCRN: Seo et al. 2018 |
| `neural_ode` | `NeuralODEPredictor`/`StandardODEFunc` [models.py:827](../models.py#L827) | \(dz/dt=\tanh(s)\,\mathrm{MLP}(z)\), dopri5 | Neural ODE: Chen et al. 2018; Latent ODE: Rubanova et al. 2019; LG-ODE: Huang et al. 2020 |
| `pnode` | `GradientNeuralODEPredictor`/`GradientODEFunc` [models.py:394](../models.py#L394) | \(dz/dt=-\alpha\nabla\Phi(z)\) | Chen 2018 ＋ 構造化力場 HNN(Greydanus 2019)/LNN(Cranmer 2020); ポテンシャル地形 Wang 2008; spectral norm Miyato 2018; RFF Rahimi & Recht 2007 |
| `dual_force` | `DualForcePotentialODEFunc` [dual_force_models.py:6-79](../dual_force_models.py#L6-L79) | \(dz/dt=\sum_j\alpha^+_{ij}(P_j-H_i)-\gamma\sum_j\alpha^-_{ij}(P_j-H_i)\) | ポテンシャル+フラックス分解: Wang et al. 2008, Ao 2004; attention: Vaswani et al. 2017 |
| `tap_node` | `TrendAnchoredPotentialODEFunc` [tap_node_models.py:9-78](../tap_node_models.py#L9-L78) | \(dz/dt=\alpha(\sum_j s_j P_j-z)/h^2\) | 準ポテンシャル: Zhou et al. 2012; Waddington: Wang et al. 2011; mean-shift: Comaniciu & Meer 2002 |
| `evolvegcn` | `EvolveGCNOPredictor` [models.py:1205](../models.py#L1205) | \(W_t=\mathrm{GRU}(\bar z_{t-1},W_{t-1}),\ z_{t+1}=z_t W_t^\top+b\) | EvolveGCN(-O): Pareja et al. **2020 (AAAI)** |
| `roland` | `ROLANDPredictor` [models.py:1237](../models.py#L1237) | \(z_{t+1}=(1-\gamma)z_t+\gamma\,\mathrm{MLP}(z_t)\) | ROLAND: You et al. 2022 (KDD) |
| `cope` | `UnifiedVGAE`（**自著拡張**） | pnode ＋ デコーダに \(w_{\mathrm{pot}}(\Phi_i+\Phi_j)\) | 本研究の拡張。CIKM'21 の同名 CoPE (Zhang et al. 2021) とは**別物** |

---

## 1. Static（`static`）

- **実装**: `StaticLatentPredictor`（[models.py:820](../models.py#L820)）。`forward` は `return z_current`。時間発展を持たず、当年潜在をそのまま次年予測に用いる（\(z_{t+1}=z_t\)）。
- **帰属と差分**: 動的リンク予測における**持続／記憶（persistence / memorization）ベースライン**。近年の標準対照は EdgeBank（Poursafaei et al. 2022）で、「過去に観測されたエッジは再出現しやすい」ことを突く純記憶ベースライン。本実装は潜在の恒等写像＋幾何デコードなので EdgeBank の記憶則そのものではないが、**時間発展なしの下限対照**という役割は同一。この位置づけを明記する。
- **English (draft).** *Static is a no-evolution persistence baseline (\(z_{t+1}=z_t\)); it plays the role of a memorization lower bound analogous to EdgeBank (Poursafaei et al., 2022).*

## 2. RNN + VGAE（`rnn`）

- **実装**: `RNNLatentPredictor`（[models.py:1087-1102](../models.py#L1087-L1102)）。ノードごとに \(K\) 年（既定 `rnn_history_len=4`）の潜在系列を系列モデルに通し、\(z_{t+1}=W\cdot\mathrm{cell}(z_{t-K+1:t})_{[-1]}\)。
- **⚠ 実装上の注意（正確な記述）**: セルは **GRU ではなく LSTM**（`nn.LSTM`, [models.py:1092](../models.py#L1092)）であり、docstring は **VGRNN 系**とタグ付けしている。会話・過去表で「RNN+VGAE」と呼んだ手法の実体はこの**潜在空間 LSTM** である。GRU はリポ内では別用途（履歴融合 `PNodeHistoryFuseGRU`、および別変種 `neural_ode_gru`）にのみ出現する。
- **帰属**: 動的グラフの変分系列モデル **VGRNN**（Hajiramezanali et al. 2019）に対応づけるのが最も近い（ただし本実装は潜在系列のみを畳み、トポロジ生成は行わない簡略版）。系列セルは LSTM（Hochreiter & Schmidhuber 1997）、GNN+RNN の源流として GCRN（Seo et al. 2018）を併記。
- **English (draft).** *RNN+VGAE evolves per-node latents with an LSTM over a K-year history in the spirit of VGRNN (Hajiramezanali et al., 2019); note the cell is an LSTM (Hochreiter & Schmidhuber, 1997), not a GRU.*

## 3. Neural ODE + VGAE（`neural_ode`）

- **実装**: `NeuralODEPredictor`／`StandardODEFunc`（[models.py:827-861](../models.py#L827-L861)）。ドリフトは**素の MLP**（ポテンシャル勾配ではない）で \(dz/dt=\tanh(\text{scale})\cdot\mathrm{MLP}(z)\)。`torchdiffeq.odeint_adjoint`・`dopri5`（rtol=atol=1e-3）で \([0,\Delta t]\) を積分。時間条件版 `NeuralODEPredictorTime`（年埋め込みを結合）も存在。
- **帰属と差分**: **Neural ODE**（Chen et al. 2018）を VGAE 潜在に適用した黒箱ベクトル場。潜在状態に対する ODE という点で **Latent ODE**（Rubanova et al. 2019）、グラフ系への展開として **LG-ODE**（Huang et al. 2020, NeurIPS）を関連手法として挙げる。P-NODE との差分は「任意ベクトル場 vs 単一スカラー場の勾配流」で、本実装では両者が**同一エンコーダ・同一デコーダ・積分器**を共有するため、\(\Phi\) 構造の帰納バイアスの有無だけが対照になる。
- **English (draft).** *Neural ODE+VGAE integrates a black-box MLP drift over the latents (Chen et al., 2018; Rubanova et al., 2019), serving as the unconstrained-vector-field counterpart to the gradient-flow P-NODE.*

## 4. P-NODE（`pnode`, 提案）

- **実装**: `GradientNeuralODEPredictor`／`GradientODEFunc`（[models.py:394-411, 704-745](../models.py#L394-L411)）。学習可能スカラー場 \(\Phi_\theta\)（`PotentialNet`, [models.py:50-109](../models.py#L50-L109)）の**勾配流** \(dz/dt=-\alpha\nabla_z\Phi_\theta(z)\)、\(\alpha=\mathrm{softplus}(\log\text{scale})\)。\(\Phi\) は RFF（sin/cos ランダム特徴）または MLP 特徴 → spectral-norm 付き MLP head → Softplus（\(\Phi\ge0\)）。
- **帰属**: Neural ODE（Chen et al. 2018）に**エネルギー様の構造化帰納バイアス**を与えた勾配流。スカラー関数からベクトル場を導く点で **Hamiltonian NN**（Greydanus et al. 2019）／**Lagrangian NN**（Cranmer et al. 2020）と同系。物理的背景としてポテンシャル地形（Wang et al. 2008）。実装要素として spectral normalization（Miyato et al. 2018）で Lipschitz を抑え、RFF（Rahimi & Recht 2007）で特徴写像を構成。
- **English (draft).** *P-NODE evolves latents as the gradient flow of a single learned scalar potential, \(dz/d\tau=-\alpha\nabla_z\Phi_\theta(z)\), giving an energy-like inductive bias in the spirit of Hamiltonian/Lagrangian neural networks (Greydanus et al., 2019; Cranmer et al., 2020).*

## 5. Dual-Force P-NODE（`dual_force`, 提案）

- **実装**: `DualForcePotentialODEFunc`／`DualForcePNODEPredictor`（[dual_force_models.py:6-105](../dual_force_models.py#L6-L105)）。当年のトピック潜在 \(P_j\) と、成長・衰退モメンタム \(D^+_j,D^-_j\) を key に用いた**二系統アテンション**で、成長トピックへの**引力** \(v_\text{in}=\sum_j\alpha^+_{ij}(P_j-H_i)\) と衰退トピックからの**反発** \(v_\text{out}=\sum_j\alpha^-_{ij}(P_j-H_i)\) を分離し、\(dz/dt=v_\text{in}-|\gamma|\,v_\text{out}\)（学習可能 \(\gamma\)）。著者行のみ積分、トピック行は固定。
- **⚠ 差分**: これは **P-NODE と違い純粋な勾配流ではない**。\(v_\text{in}-v_\text{out}\) は一般に**非保存（渦あり）**の力場であり、単一スカラーポテンシャルには還元できない。
- **帰属**: 非平衡系の**「ポテンシャル＋フラックス（回転）」分解**（Wang et al. 2008; Ao 2004）に対応づける——保存項（引力）にエネルギーへ還元できない非保存項（反発・循環）を明示的に足す構成。アテンション重み \(\alpha^\pm\) は Transformer 型注意（Vaswani et al. 2017）。
- **English (draft).** *Dual-Force P-NODE adds an explicit attraction-to-growing / repulsion-from-declining attention field, \(v_\text{in}-\gamma\,v_\text{out}\); unlike P-NODE this is a non-conservative field, matching the potential-plus-flux decomposition of non-equilibrium dynamics (Wang et al., 2008; Ao, 2004).*

## 6. TAP-NODE（`tap_node`, 提案）

- **実装**: `TrendAnchoredPotentialODEFunc`（[tap_node_models.py:9-78](../tap_node_models.py#L9-L78)）。トピック潜在を固定アンカー \(P_j\)（detach）とし、質量・トレンドで井戸の深さを重み付けた**単一スカラーポテンシャル**
  \(\Phi(z,t)=-\mathrm{logsumexp}_j[\log w_j(t)-\|z-P_j\|^2/(2h^2)]\)、\(\log w_j=\kappa\log(1+M_j)+b\,\tilde D_j\)、\(\tilde D_j=\log(1+D^+_j)-\log(1+D^-_j)\)。勾配流は**解析形** \(dz/dt=\alpha(\sum_j s_j(z)P_j-z)/h^2\)、\(s_j=\mathrm{softmax}_j(\cdot)\)。学習パラメータは \(\kappa,b,\log h,\log\text{scale}\) の**スカラー4個のみ**。
- **帰属**: 複数アトラクタを結ぶ**準ポテンシャル地形**（Zhou et al. 2012）と Waddington 地形の定量化（Wang et al. 2011）に対応。更新則 \(z\to\) trend 重み付きアンカーの softmax 加重平均は **mean-shift**（Comaniciu & Meer 2002）と同型で、これを引用に用いる。衰退（\(\tilde D_j<0\)）は井戸を浅くする形で単一スカラー場に表現される。
- **English (draft).** *TAP-NODE is a closed-form gradient flow of a single trend-weighted mixture-of-wells potential whose attractors are anchored to topic positions (a quasi-potential in the sense of Zhou et al., 2012); its update is a trend-weighted mean-shift (Comaniciu & Meer, 2002) with only four scalar parameters.*

## 7. その他のリポ内対照

- **EvolveGCN-O**（`EvolveGCNOPredictor`, [models.py:1205-1234](../models.py#L1205-L1234)）: GCN 重み行列を RNN で時間発展させる EvolveGCN（Pareja et al. 2020）の潜在空間版。GRU 隠れ状態が重み行列 \(W\in\mathbb{R}^{d\times d}\) を兼ね、\(z_{t+1}=z_t W_t^\top+b\)。
  **⚠ 会場の訂正**: リポの docstring は "NeurIPS 2020" だが、EvolveGCN は正しくは **AAAI 2020**（Proc. AAAI 34(04):5363–5370, arXiv:1902.10191）。本文・コード docstring とも AAAI 2020 に統一する。
- **ROLAND**（`ROLANDPredictor`, [models.py:1237-1263](../models.py#L1237-L1263)）: 静的 GNN を階層状態の再帰更新で動的化する ROLAND（You et al. 2022, KDD）の潜在空間版。ノードごとゲート \(\gamma=\sigma(W_\gamma z_t)\) で \(z_{t+1}=(1-\gamma)z_t+\gamma\,\mathrm{MLP}(z_t)\)。
- **CoPE-VGAE**（`cope`, `UnifiedVGAE`）: **本研究の拡張**。時間発展は P-NODE と同型の勾配流で、デコーダ logit に \(w_{\mathrm{pot}}(\Phi(z_i)+\Phi(z_j))\) を加え、力学と尤度で \(\Phi\) を共有する（ポテンシャル一貫性）。
  **⚠ 注記**: CIKM'21 の同名論文 **"CoPE: Modeling Continuous Propagation and Evolution on Interaction Graph"（Zhang et al. 2021）とは無関係**である旨を本文で一文明記し、査読者の取り違えを防ぐ。
- **English (draft).** *We additionally compare against latent-space adaptations of EvolveGCN-O (Pareja et al., 2020) and ROLAND (You et al., 2022); our CoPE-VGAE shares the potential \(\Phi\) between dynamics and decoder and is unrelated to the identically named CoPE of Zhang et al. (2021).*

---

## 実装と正準手法の差分（チェックリスト）

査読前に本文へ必ず反映する 5 点:

1. **`rnn` = LSTM**（GRU ではない, [models.py:1092](../models.py#L1092)）。「LSTM ベース／VGRNN 系」と記述。
2. **`static` = persistence**（学習可能な時間発展なし）。EdgeBank を naive 対照の出典に。
3. **`neural_ode` のドリフトは素の MLP**（ポテンシャル勾配でない）。実験で AUC≈0.5 に退化した事実は結果節で言及。
4. **EvolveGCN は AAAI 2020**（リポ docstring の "NeurIPS 2020" は誤り）。
5. **`cope` は自著拡張**で、CIKM'21 の CoPE とは別物。

---

## References

共有構成要素・ベースライン
- **GAT** — Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., Bengio, Y. (2018). Graph Attention Networks. *ICLR 2018*. arXiv:1710.10903.
- **VGAE** — Kipf, T. N., Welling, M. (2016). Variational Graph Auto-Encoders. *NeurIPS Bayesian Deep Learning Workshop 2016*. arXiv:1611.07308.
- **GCN** — Kipf, T. N., Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*. arXiv:1609.02907.
- **LayerNorm** — Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). Layer Normalization. arXiv:1607.06450.
- **Residual (ResNet)** — He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*, pp. 770–778. DOI: 10.1109/CVPR.2016.90. arXiv:1512.03385.
- **EdgeBank / evaluation** — Poursafaei, F., Huang, S., Pelrine, K., Rabbany, R. (2022). Towards Better Evaluation for Dynamic Link Prediction. *NeurIPS 2022 Datasets & Benchmarks Track*. arXiv:2207.10128.
- **VGRNN** — Hajiramezanali, E., Hasanzadeh, A., Narayanan, K., Duffield, N., Zhou, M., Qian, X. (2019). Variational Graph Recurrent Neural Networks. *NeurIPS 2019*. arXiv:1908.09710.
- **LSTM** — Hochreiter, S., Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation* 9(8): 1735–1780. DOI: 10.1162/neco.1997.9.8.1735.
- **GCRN** — Seo, Y., Defferrard, M., Vandergheynst, P., Bresson, X. (2018). Structured Sequence Modeling with Graph Convolutional Recurrent Networks. *ICONIP 2018*, LNCS 11301. arXiv:1612.07659.

時間発展・Neural ODE 系
- **Neural ODE** — Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. (2018). Neural Ordinary Differential Equations. *NeurIPS 2018*. arXiv:1806.07366.
- **Latent ODE** — Rubanova, Y., Chen, R. T. Q., Duvenaud, D. (2019). Latent ODEs for Irregularly-Sampled Time Series. *NeurIPS 2019*. arXiv:1907.03907.
- **LG-ODE** — Huang, Z., Sun, Y., Wang, W. (2020). Learning Continuous System Dynamics from Irregularly-Sampled Partial Observations. *NeurIPS 2020*. arXiv:2011.03880.
- **EvolveGCN** — Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., Leiserson, C. E. (2020). EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. *AAAI 2020*, 34(04): 5363–5370. DOI: 10.1609/aaai.v34i04.5984. arXiv:1902.10191.
- **ROLAND** — You, J., Du, T., Leskovec, J. (2022). ROLAND: Graph Learning Framework for Dynamic Graphs. *KDD 2022*, pp. 2358–2366. DOI: 10.1145/3534678.3539300. arXiv:2208.07239.
- **CoPE (別手法・注記用)** — Zhang, Y., Xiong, Y., Li, D., Shan, C., Ren, K., Zhu, Y. (2021). CoPE: Modeling Continuous Propagation and Evolution on Interaction Graph. *CIKM 2021*, pp. 2627–2636. DOI: 10.1145/3459637.3482419.

構造化力場・ポテンシャル（提案手法の背景）
- **Hamiltonian NN** — Greydanus, S., Dzamba, M., Yosinski, J. (2019). Hamiltonian Neural Networks. *NeurIPS 2019*. arXiv:1906.01563.
- **Lagrangian NN** — Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., Ho, S. (2020). Lagrangian Neural Networks. *ICLR 2020 Workshop DeepDiffEq*. arXiv:2003.04630.
- **Attention** — Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. arXiv:1706.03762.
- **Spectral Normalization** — Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR 2018*. arXiv:1802.05957.
- **Random Fourier Features** — Rahimi, A., Recht, B. (2007). Random Features for Large-Scale Kernel Machines. *NeurIPS 2007*.
- **Mean shift** — Comaniciu, D., Meer, P. (2002). Mean Shift: A Robust Approach Toward Feature Space Analysis. *IEEE TPAMI* 24(5): 603–619. DOI: 10.1109/34.1000236.

非平衡ポテンシャル・準ポテンシャル（提案手法の物理的背景）
- **Potential + flux** — Wang, J., Xu, L., Wang, E. (2008). Potential landscape and flux framework of nonequilibrium networks. *PNAS* 105(34): 12271–12276. DOI: 10.1073/pnas.0800579105.
- **A-type decomposition** — Ao, P. (2004). Potential in stochastic differential equations: novel construction. *J. Phys. A: Math. Gen.* 37(3): L25–L30. DOI: 10.1088/0305-4470/37/3/L01.
- **Quasi-potential** — Zhou, J. X., Aliyu, M. D. S., Aurell, E., Huang, S. (2012). Quasi-potential landscape in complex multi-stable systems. *J. R. Soc. Interface* 9(77): 3539–3553. DOI: 10.1098/rsif.2012.0434.
- **Waddington landscape** — Wang, J., Zhang, K., Xu, L., Wang, E. (2011). Quantifying the Waddington landscape and biological paths for development and differentiation. *PNAS* 108(20): 8257–8262. DOI: 10.1073/pnas.1017017108.

---

## 関連ドキュメント
- [PNODE_PAPER_FRAMING.md](PNODE_PAPER_FRAMING.md) — 記号表・評価プロトコル段落
- [EXPERIMENTAL_SETTINGS.md](EXPERIMENTAL_SETTINGS.md) — 学習・評価・指標
- [EXTERNAL_BASELINE_PLAN.md](EXTERNAL_BASELINE_PLAN.md) — 外部ベースライン計画（同一プロトコル）
- [../PAPER_WORKFLOW.md](../PAPER_WORKFLOW.md) — 手法↔実装対応・表の書き方
