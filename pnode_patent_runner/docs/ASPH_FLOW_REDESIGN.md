# ASPH-Flow 再設計仕様書 (v2) — Relatedness-Anchored Honest Entry Forecasting

**Status**: v1（外部LLM生成のLangevin SDE仕様）は Stage-0 ゲートで棄却（2026-07-16 測定）。
本書は v1 を、本プロジェクトの確定証拠（simple-beats-complex ×9、velocity attribution、
conformal 転移）と Gate 0/0b/0c の実測値のみで組み直した後継仕様。
**本書のすべての数値は実測済み**（訓練不要・決定論、シード分散ゼロ）。

- Gate 0 実装: `pnode_patent_runner/asph_stage0_static.py`
- Gate 0b/0c 実装: `pnode_patent_runner/asph_stage0b_blend.py`
- 結果: `pnode_patent_runner/outputs/asph_stage0/construction_maingroup{,_gate0bc}.json`

---

## 1. v1 棄却の実測記録（改訂の根拠）

プロトコル: leak-free Task B（construction・maingroup・portfolio<2019 → 初参入 2019–2023・
全未参入候補ランキング・n_pairs=4,112）。

| スコアラ | MRR | Hit@10 | 判定 |
|---|---|---|---|
| v1 静的幾何（content→SVD-2D→S¹ cosine） | 0.076 | 0.166 | **popularity 未満** |
| 同 2D・ユークリッド距離 | 0.087 | 0.215 | S¹ 射影は純減 |
| content 384次元 cosine（無圧縮） | 0.168 | 0.387 | content の天井 |
| popularity | 0.097 | 0.222 | |
| **relatedness（バー）** | **0.213** | **0.473** | 既知の 0.213 を再現 |

診断: (i) SVD-2D 圧縮（説明分散 52%）で content 信号が半減、(ii) S¹ 射影でさらに減少
（デコーダがポテンシャル地形の動径情報を消去する設計矛盾の実測確認）、
(iii) 残差 0.076→0.213 を動力学で埋めるには +180% 必要だが、velocity attribution
（6ドメイン）で連続ドリフトの信号は R²≈0.01–0.05 と確定済み。

## 2. 設計原則（証拠に紐付く制約）

- **P1: 予測スコアは高次元 or 構造ベース。** 2次元への圧縮を予測経路に置かない
  （根拠: Gate 0 の 0.168→0.087→0.076 の分解）。
- **P2: 連続潜在動力学に精度主張をさせない。** SDE/ODE は Layer 1 に入れない
  （根拠: velocity attribution、PI-SDE Poisson Phase B ΔMRR −0.010、Task B free-z0 artifact）。
- **P3: 2次元空間は可視化専用。** 予測とデコードから切断し、記述・説明にのみ使う
  （根拠: bipartite landscape min-exp — 可視化は成立、growth-aware Φ は holdout 不成立）。
- **P4: 不確実性は conformal で定量化し、天井の低さをユーザーに開示する**
  （根拠: non-eq Langevin 実データで唯一転移したのが被覆 CovConf 0.96、本書 Gate 0c で Task B でも確認）。
- **P5: 学習要素の追加は事前登録ゲート通過が条件**（§6）。通過実績のある学習要素は現状ゼロ。

## 3. アーキテクチャ（3層）

### Layer 1 — 予測層（訓練不要）

主スコア: **relatedness**（訓練期間のみの firm-year 共起グラフ）

$$s_{\text{rel}}(u, c) = \sum_{i \in \Pi_u} \text{cooc}_{<T}(c, i)$$

（$\Pi_u$ = 企業 $u$ の訓練期ポートフォリオ、$\text{cooc}_{<T}$ = テスト開始年より前のみで構築した共起）

任意オプション: **content ブレンド**（候補集合上で企業ごとに z 標準化して線形結合）

$$s_{\beta}(u, c) = (1-\beta)\, z\!\left[s_{\text{rel}}(u,\cdot)\right](c) + \beta\, z\!\left[\cos(h_u, h_c)\right](c), \qquad h_u = \frac{1}{|\Pi_u|}\sum_{i \in \Pi_u} h_i$$

**実測（Gate 0b）**: β は検証遷移（portfolio<2017 → 参入 2017–2018）で選択し β*=0.55。
テスト遷移で MRR 0.215 vs 0.213、Hit@10 0.486 vs 0.473。企業単位ペア・ブートストラップ
（2000回）で ΔMRR = +0.0026、95%CI [+0.0003, +0.0049]、P(Δ≤0)=0.013。
**統計的に有意だが実用上は僅少**。検証曲線は β∈[0, 0.85] でほぼ平坦（0.224–0.228）で
β=1 のみ崩落（0.177）→ β の選択に敏感でない。RRF 融合は 0.206 で劣後。

採用判断: 既定は relatedness 単独（シンプルさ優先）。Hit@k を直接ユーザーに見せる
レコメンダ文脈でのみ blend を許可（+1.3pt Hit@10）。
既知の注意: `cpc_content_*.npz` の CPC タイトルサンプルは年フィルタなし
（content 側に軽微なリーク可能性 → blend の増分はやや楽観的な上限とみなす）。

### Layer 2 — 不確実性層（conformal ランク被覆）

真の参入の非適合度 = スコアラ下でのランク。検証遷移のランク分布の
$(1-\alpha)$ 分位（$(n{+}1)$ 補正付き）を $r^*$ とし、予測集合 = 上位 $r^*$ 候補。

$$r^*(\alpha) = \text{Quantile}_{\lceil (n+1)(1-\alpha) \rceil / n}\left(\{\text{rank}(y_j)\}_{j=1}^{n}\right)$$

**実測（Gate 0c、検証 2017–2018 で較正 → テスト 2019–2023 で被覆検証）**:

| α | スコアラ | r* | 実測被覆 | 目標 |
|---|---|---|---|---|
| 0.10 | relatedness | 79 | 0.901 | 0.90 |
| 0.10 | blend β=0.55 | 107 | 0.893 | 0.90 |
| 0.20 | relatedness | 45 | 0.795 | 0.80 |
| 0.20 | blend β=0.55 | 48 | 0.776 | 0.80 |

被覆は時間分割をまたいで転移する（C8 の Task B 実証）。そして **honest-tool の中核数値**:
実参入の 90% を捕捉するには 231 コード中上位 79（候補の約 1/3）が必要。
これが「white-space 予測」の実力の定量開示であり、ツール UI はこの $r^*$ を明示する。

### Layer 3 — 可視化層（記述専用・予測主張なし）

v1 の「ポテンシャル地形」はここでのみ生存する。content 埋め込みの SVD-2D
（または既存 UMAP パイプライン）上に密度地形を描き、以下をオーバーレイする:

- 企業の実参入軌跡（離散イベントとして。連続軌道として描かない）
- Layer 1 の top-k 予測と Layer 2 の conformal 集合サイズ（局所的な予測可能性の開示）
- 成長はオーバーレイ表示（地形の谷≠成長は確定済み。地形に成長の意味を持たせない）

流用: `run_bipartite_landscape.py` / `interactive_landscape*.py` / EDGPAT viz（legibility 合格済み）。
クラマース遷移・山谷の語彙は**説明のメタファーとしてのみ**使用可。図の注記に
「地形は記述であり参入予測は Layer 1 による」旨を必須記載。

## 4. v1 からの削除要素と理由（トレーサビリティ）

| v1 要素 | 処置 | 根拠 |
|---|---|---|
| S¹ コサインデコーダ | 削除 | Gate 0: 2D ユークリッドより悪い（0.076 < 0.087）。動径情報の消去 |
| SVD-2D を予測経路に使用 | Layer 3 へ隔離 | Gate 0: content 信号が 0.168→0.087 に半減 |
| DeepMLP ポテンシャル $U_\theta$ | 削除 | PI-SDE Phase B: 学習補正は holdout ΔMRR −0.010 |
| Langevin SDE / ハミルトン動力学 | 削除 | velocity attribution: 連続ドリフト R²≈0.01–0.05（6ドメイン） |
| $\mathcal{L}_{\text{Onsager}}$ | 削除 | 潜在 $p$ は非観測 → 自己言及損失（$\sigma_i \to 0$ 圧力にしかならない） |
| $\mathcal{L}_{\text{entropy}}$ 斥力 | 削除 | 角度デコーダ削除に伴い動径平衡問題ごと消滅。$O(N^2)$ |
| 質量 $m_i$ / 摩擦 $\gamma$ / 揺らぎ $\sigma_i$ | 削除 | 同定不能（anchored でない自由 NN 下では ω すら非同定の前例） |
| 加熱冷却アニーリング (1.2/2.5/0.05) | 削除 | 根拠のない自由パラメータ3個。「跳躍の発見」がスケジュール調整と区別不能 |
| モンテカルロ期待値 ($M=200$) | 置換 | conformal ランク集合が同じ「不確実性の定量化」を保証付き・訓練不要で提供 |
| 予測対象「未知の未来の萌芽的リンク」 | 再定義 | novelty hazard: 跳躍(WHAT-NEW)は無信号確定。対象は WHERE（参入先ランキング）+ 被覆開示 |

## 5. 評価プロトコル（固定）

- データ: `data/processed/bipartite_{domain}_firm.csv`、granularity=maingroup（決戦粒度）
- 分割: 検証遷移 = portfolio<2017 → 初参入 2017–2018（モデル選択・conformal 較正のみ）、
  テスト遷移 = portfolio<2019 → 初参入 2019–2023（1回だけ評価）
- 指標: MRR / Hit@{5,10,20}（全未参入候補ランキング）、conformal 被覆、
  有意性は企業単位ペア・ブートストラップ 2000 回
- 共起・ポートフォリオ・企業特徴はすべて分割前年のみで構築（leak-free）

## 6. 将来の学習要素に対する事前登録ゲート（P5 の運用）

任意の追加提案（GNN・時系列・LLM 特徴・動力学の再導入を含む）は、実装投資の前に:

1. **Gate S（静的アブレーション、半日）**: その提案の「動力学・学習なし」版を §5 プロトコルで評価。
   relatedness 0.213 未満なら学習で埋める余地の説明責任が提案側に発生。
2. **Gate L（最小学習版、1日）**: 合格基準を事前登録 —
   (a) 5 シードの分散を超えるマージンで Gate S 版を上回る、かつ
   (b) blend β* (0.215) を上回る。
3. 不合格の結果も `outputs/asph_stage0/` に保存し、limits ペーパーの証拠表に追記する。

## 7. 残作業（優先順）

1. **多ドメイン再現**: Gate 0b/0c を energy（`cpc_content_energy.npz` 既存）ほかへ拡張。
   blend の僅少増分と conformal 転移がドメイン普遍かを確認（RESEARCH_PLAN の検証 D と統合可能）。
2. content npz の年フィルタ版再構築（blend 増分の上限バイアス除去）。
3. Layer 3 の統合デモ: 既存 landscape パイプラインに conformal 集合サイズのオーバーレイを追加
  （KG-ATLAS の honest-ify 路線の最初の成果物候補）。
4. レコメンダ文脈（recommender_firm.py）での blend + isotonic 較正の再評価
  （既知課題: fusion の top-k 汚染・ECE 0.165 と突き合わせ）。

## 8. 位置づけ

本仕様は独立した新モデルではなく、(i) limits-of-prediction ペーパーの C1/C7/C8 を
1 つの運用可能なツール設計に落としたもの、(ii) KG-ATLAS の「証明済みに死んだターゲット」
（Prophet 予測・white-space）を honest 化する置換部品、である。v1 の物理語彙で新規性を
主張することは本仕様の下では禁止（P2/P3）。新規性は「予測可能性の天井を測定し、
その天井ごとユーザーに開示するツール設計」に存する。
