# Dual-Force P-NODE 再設計仕様 (v2) — Regime-Conditioned Attraction-Repulsion Attention

**Status**: 提案①（引力・反発アテンション ODE）は **実装済み・未ゲート**。
`dual_force_models.py`（2026-04-22 実装）が仕様どおりの構造 —— 成長/衰退モメンタムを Key にした
二系統アテンション、著者ごとの Query、引力−反発の合成ベクトル場、著者行のみを Neural ODE 積分 ——
をすでに持っている。本書は、それを査読可能な結果に変換するための **再設計 (v2)**。
埋めるべき穴は3つ: holdout 評価が未実装、既存の予備比較が無効なドメインで行われている、
4つの主張（成長率 Key・researcher-wise アテンション・引力反発の分離・連続時間 ODE）が
成分分解されずに1モデルに詰め込まれている。

- v1 実装: `dual_force_models.py`, `dual_force_training.py`, `dual_force_eval.py`
- v1 予備結果（3 seed, transductive）: `outputs/dual_force_compare/multiseed/dual_force_vs_pnode_seed{42,7,123}.json`
- 参照する確定済み証拠: `WHY_NEURALODE_FAILS_ja.md`、`docs/RESEARCH_PLAN_PREDICTABILITY.md`、
  `outputs/predictability_map/RESULTS.md`（検証 B/C/GEM）、TAP-NODE の holdout 結果
  (`outputs/tap_node/tap_node_holdout_seed*.json`)、velocity attribution（6ドメイン力場診断）
- 外部文献サーベイ（トップ会議・確立文献との接続）: `docs/RELATED_WORK_SURVEY.md`。
  §4・§5・§9 の変更点はこのサーベイの反映

---

## 1. v1 の実測記録と、なぜこのままでは論文にならないか

### 1.1 予備結果（3 seed, author_topic, transductive）

| seed | Dual-Force AUC | P-NODE(B+D) AUC | 差 |
|---|---:|---:|---:|
| 42 | 0.650 | **0.878** | −0.228 |
| 7 | 0.603 | 0.595 | +0.008 |
| 123 | 0.653 | 0.667 | −0.014 |
| 平均 | 0.635 | 0.713 | **−0.078** |

3 seed だけで両手法とも分散が28pt（0.595〜0.878）に達しており、方向すら確定しない
（本プロジェクトの基準は ≥5 seed・対応あり検定 — `PAPER_WORKFLOW.md` 3節）。かつこの比較は
**transductive**（`dual_force_eval.py` に holdout 分岐が存在せず、`years[-2]→years[-1]` を無条件に使う。
`grep holdout dual_force_eval.py` はヒットしない）。本プロジェクトの確定則
（transductive は系統的に精度を過大評価する、`RESEARCH_PLAN_PREDICTABILITY.md` C6）に照らすと、
Dual-Force は TAP-NODE や P-NODE がすでに通過した holdout ゲートを一度も通っていない。

### 1.2 さらに悪いことに、比較に使ったドメイン自体が無効に近い

検証C（`outputs/predictability_map/RESULTS.md`）は上記と**同じ author_topic ドメイン**で、
学習なしの人気度スコアだけで AUC 0.941〜0.944 が出ることを示した。同ドメインでの学習モデルの
holdout AUC は次の通りで、いずれも天井から大きく下にいる:

| 手法 | AUC (holdout, author_topic) |
|---|---:|
| 人気度のみ（学習なし） | **0.944** |
| RNN | 0.752±0.088 |
| TAP-NODE | 0.651±0.083 |
| P-NODE | 0.615±0.089 |
| static | 0.554±0.008 |

検証Cの結論は「粗粒度ベンチでのモデル比較はほぼ無意味」——つまり v1 の予備比較
（Dual-Force 0.635 vs P-NODE 0.713）は、そもそも手法間の差を測るのに向かないドメインで
測った差である可能性が高い。v2 は評価ドメインを差し替える（§3 P1）。

### 1.3 プロジェクトの現在地は、すでにこの方向の"その先"を試して単純化に振れている

```
2026-04-22  Dual-Force 実装（アテンション・二系統力・学習パラメータ多数）
2026-06-12  WHY_NEURALODE_FAILS 診断 — Dual-Force を含む手法系列を
            「relatedness-bound」と名指しで総括
2026-07-05  TAP-NODE 実装（スカラー4個のみ・closed-form 勾配流）
            → holdout 10 seed で P-NODE 平均に対し同等以上、分散は約1/3
            （P-NODE の単発高値 0.933 は seed 運と判明）
```

本プロジェクト自身が、アテンション＋多パラメータ（Dual-Force）→ スカラー4個の closed-form
（TAP-NODE）という**単純化の方向にすでに一度進んでいる**。提案①（成長率を Key 化・researcher-wise
アテンション・引力反発の分離・連続時間 ODE）は、その単純化以前の地点に**複雑さを足し戻す**提案に
なっている。悪いわけではないが、「なぜ今度は複雑さが報われるのか」を個別の仮説として立てて
検証しない限り、本プロジェクトの型（simple-beats-complex ×9 件、predictability map メモ）の
10件目になる公算が高い。

### 1.4 もう一つの障害: 連続時間 ODE という枠組み自体への反証

velocity attribution（6ドメイン）は、力場（グローバル場・relatedness・回転）が firm×CPC の連続速度を
説明する分散が R²≈0.01〜0.05 にとどまり、「動きは連続的なドリフトではなく、離散的で relatedness に
ゲートされた参入である」と結論している。提案①の核である「Neural ODE で技術関心の変化を連続時間で
モデル化する」は、この確定済みの機構診断と正面から衝突する。v2 では連続時間 ODE 自体を対照群として
検証する（§4 成分 d）。

---

## 2. それでも提案①に価値がありうる場所

全否定ではない。以下3点は本プロジェクトの証拠と矛盾しない、あるいはむしろ支持される。

- **GEM skill score**（`gem_skill.py`, 10 seed）は、momentum の係数が全ドメインで負（平均回帰の罰則）
  だが **mom×burst 交互作用は特許ドメインで正**（バースト期は罰則が緩む）。これは提案①の
  「成長分野への引力」が線形 momentum ではなく **burst 条件付きの非線形形**で初めて実データに
  支持される、という具体的な設計示唆である（§3 P4）。
- firm×CPC の特許ドメインでは、Neural ODE が relatedness に条件付けたとき MRR +126% という
  本プロジェクト最大級の正の結果が出ている（Task B ODE findings）。Dual-Force のアテンションが
  growth momentum だけでなく relatedness も Key に含めるなら、この既知の勝ちパターンと接続できる
  （§3 v2 アーキテクチャ）。
- 「researcher-wise の興味の強さ」は、精度で勝てなくても**記述的診断**として独立した価値を持ちうる
  （ASPH-Flow と同じ「honest tool」路線）。

---

## 3. v2 設計原則（証拠に紐付く制約）

- **P1: 評価は fine-grained 特許ドメイン（agrifood, construction; CPC maingroup）を主とする。**
  author_topic は検証Cにより無効化済み（§1.2）。energy は人気度天井が学習モデルを上回る中間ドメイン
  なので補助的に使う。
- **P2: holdout-test-year を実装してからでないと結果を主張しない。**
  `unified_training.split_bundle_holdout_test_year` と同じ分割ロジックを dual_force 系にも適用する
  （最初の実装タスク、低コスト）。
- **P3: 連続時間 ODE 積分を対照群として持つ。**
  1ステップ離散更新（アテンション適用を1回だけ・ODE 積分器なし）を同じ損失で学習し、性能差が
  なければ「連続時間」という主張の主要部分は落とす（§1.4 と整合）。
- **P4: growth-rate-as-Key は線形 momentum でなく burst 条件付きで実装する。**
  Key 入力を `[P_j, D_j]` から `[P_j, D_j, burst_j]`（burst_j = 直近数年の加速度や分散で定義する
  レジーム指標）に拡張する。burst の定義は `FILING_COUNT_FORECAST_DESIGN.md` の Gate 0 と共有する
  （②の成果をそのまま Key の特徴として輸入する設計）。
- **P5: 学習要素は Gate S / Gate L を通過して初めて主張に使う。**
  Gate S の相手は GEM ロジスティック回帰（`gem_skill.py` と同じ特徴: rel, mom, mom×burst, seen）——
  アテンション・ODE の機構を足す前に、まず同じ特徴量で線形モデルを上回れるかを見る。

---

## 4. v2 アーキテクチャと成分分解（提案①の4主張を独立に検証可能にする）

提案①は4つの独立した主張を1つのモデルに詰め込んでいる。v2 はこれを崩さず、各主張を
**個別に ON/OFF できるアブレーション**として実装する。

| # | 提案①の主張 | v2 での対照（OFF条件） | 事前登録した予想 |
|---|---|---|---|
| a | 成長率を Key に組み込む | Key = `[P_j]` のみ（momentum抜き） | burst 込みの `[P_j,D_j,burst_j]` が勝つ（§3 P4 の根拠） |
| b | researcher-wise のアテンション | Query を全著者共通の定数に固定 | **弱い予想（負）**: human-aware 診断は pure-inductive で個人差の学習リフトがほぼ出ないことを示している。効くとしても transductive（既知著者）に限られる可能性が高い |
| c | 引力・反発の分離（dual force） | `gamma=0` 固定（単一の引力のみ、TAP-NODE 型の「井戸が浅くなる」表現に近づける） | **弱い予想（負）**: TAP-NODE の単一ポテンシャル（4スカラー）がすでに P-NODE 平均に並ぶため、分離した反発項が追加で稼ぐ余地は小さい |
| d | 連続時間 Neural ODE | 離散1ステップ（積分器なし、同じアテンション機構を1回だけ適用） | **弱い予想（負）**: velocity attribution（R²≈0.01–0.05）より、離散版と連続版は同等になる可能性が高い |

**このプロジェクトの型を踏まえた事前予想**: a（burst 条件付き momentum）だけが有意に効き、
b/c/d はほぼ null になる、というのが最も証拠と整合的なシナリオ。もしそうなら、勝者は
「Dual-Force」ではなく「TAP-NODE + burst-conditioned trend weight」という**さらに単純な**モデルに
なる（TAP-NODE の `\tilde D_j = \log(1+D^+_j)-\log(1+D^-_j)` を burst 項で重み付けするだけの拡張、
パラメータ +1〜2個）。この場合は v2 の主結果としてそれを主表に出し、Dual-Force（アテンション full 版）
は「同等の性能をより多いパラメータ・より高い分散で達成する」というネガティブな比較対象として
脇に置く——本プロジェクトにとって新しい種類の着地ではなく、想定内の帰結である。

### 4.1 外部文献に基づく追加候補（`docs/RELATED_WORK_SURVEY.md` 参照）

TAP-NODE（4スカラー）と Dual-Force full（自由アテンション）の間には、外部文献に基づく
**もう一段階の候補**を挟むべきことがサーベイで判明した。

- **ODNet/UniGO 型の単一関数版**（Lv et al. 2023 arXiv:2310.01272; Lv, Zhou et al. 2025 *UniGO*,
  WWW 2025）: `alpha_ij = phi(sim(h_i, P_j), D_j)` という**区分関数1本**で引力（同質性: 似た位置×
  成長トピック）・反発（異質性: 遠い位置×衰退トピック）を統一的に表現する。Dual-Force の
  Query/Key二系統アテンションよりパラメータが少なく、over-smoothing 回避の理論的裏付けも
  UniGO（WWW 2025, coarsen-refine で均衡状態を保ちつつ over-smoothing 回避、合成→実データの
  汎化を実証）にある。**Gate S と Gate L の間に挿入する新しい対照**として実装する。
- **TI-ODE 型の基底数アブレーション**（Wang et al. 2026, arXiv:2604.24811, 掲載先未確認）:
  グラフ ODE の発展関数を「学習可能な複数の相互作用基底関数の時間変化する重み付き和」に
  一般化する提案。Dual-Force の `v_in − γ·v_out` は、この一般形の **意味論固定・基底数K=2** の
  特殊ケースとみなせる。「意味論固定K=2」vs「意味論自由K=2」vs「K>2自由基底」を比較し、
  意味論を人間側で固定すること自体が性能を犠牲にしていないかを追加アブレーションとして見る。
- **GSNOP 型の Neural Process 化**（Luo et al., WSDM 2023, arXiv:2211.08568,
  コード: github.com/RManLuo/GSNOP）: §1.1 の seed 分散問題（P-NODE が 3 seed で 0.595〜0.878）
  への直接対応。点推定でなく関数上の分布を学習する定式化は、疎な動的グラフでの過学習緩和を
  目的として設計されており、10 seed で分散を力技で均す前に、まずこのアーキテクチャ変更で
  分散そのものを減らせないかを試す価値がある。

---

## 5. 評価プロトコル（固定）

- ドメイン: **construction, agrifood**（maingroup、検証Cで構造学習に実質価値ありと確認済み）を主表、
  energy と author_topic は補足（§3 P1）。
- 分割: `--holdout-test-year`（実装後）、テスト遷移は1回のみ評価。
- seed: **10**（TAP-NODE と揃える。3 seed では §1.1 のとおり方向すら確定しない）。
- ベースライン階層（すべて上から順に上回って初めて次の主張に進む）:
  1. popularity / seen-before（訓練不要）
  2. relatedness（訓練不要）
  3. GEM ロジスティック回帰（rel, mom, mom×burst, seen; `gem_skill.py` と同一特徴）
  4. TAP-NODE（4スカラー）
  5. ODNet/UniGO 型の単一関数版（区分関数1本、§4.1）
  6. P-NODE（単一学習ポテンシャル）
  7. Dual-Force v2（フル）
- 指標: AUC/AP/ECE ＋ **天井比 skill score**（`gem_skill.py` と同じ定義:
  `(AUC-AUC_ceil)/(1-AUC_ceil)`）。生 AUC だけを主表に出さない（検証Cの教訓）。
- 検定: 対応あり Wilcoxon、10 seed。

## 6. 事前登録ゲート（Gate S / Gate L）

1. **Gate S**（半日）: Dual-Force v2 の「学習・ODE なし」版（= GEM ロジスティック回帰、burst 特徴込み）
   が relatedness を有意に上回るか。上回らなければ以降のニューラル投資は不要。
2. **Gate L**（1〜2日）: TAP-NODE（+burst 拡張）が Gate S を 10 seed の分散を超えるマージンで
   上回るか。上回らなければ TAP-NODE+burst を主表に採用し、Dual-Force（full attention）の投資は
   ここで止める。
3. **Dual-Force full の投資条件**: Gate L を通過した場合のみ、§4 の成分 b と c を追加検証し、
   TAP-NODE+burst を有意に超えるかを見る。超えない場合は Dual-Force を主表から落とし、
   「シンプルな拡張で十分だった」を追加の simple-beats-complex 事例として記録する。

## 6.1 Gate S / Gate L 実測結果（2026-07-22 実行）

**実装**: `dual_force_data_patent.py`（burst_j(t) 付き patent ドメインローダー）、
`tap_node_models.py` の burst 交互作用スカラー `c`、`run_tap_node_patent_domain.py`
（holdout 対応ランナー）。ドメイン: construction / agrifood、`year_range=(2017,2021)`、
`holdout_test_year=2021`、5 seed（{0,1,7,42,123}）、epochs=10、`--scalar-lr 0.05`。
**Gate S（既出）**は `gem_skill.py`（10 seed、construction/agrifood で既に relatedness を
有意に上回っていることを前回確認済み）。**Gate L**は今回新規実行。

| ドメイン | TAP-NODE+burst holdout AUC (5 seed) | Gate S (GEM full) | relatedness | 差 | 対応検定 p | 判定 |
|---|---:|---:|---:|---:|---:|---|
| **construction** | **0.8111 ± 0.0234**（範囲 0.786–0.839） | 0.7658 | 0.7268 | **+0.0453** | **0.0179** | **Gate L PASS** |
| agrifood | 0.7466 ± 0.0447（範囲 0.668–0.792） | 0.7716 | 0.7130 | −0.0250 | 0.326 | Gate L NOT PASSED |

**読み**:

- **construction は本プロジェクトで初めて「学習モデルが GEM（訓練不要+線形特徴の天井）を
  有意に上回った」事例**。TAP-NODE+burst（5スカラーのみ）が GEM full（4特徴のロジスティック回帰）
  を +0.045、p=0.018 で上回った。§1.3 で懸念した「10件目の simple-beats-complex」にはならず、
  逆に「解釈可能な少数パラメータの学習モデルが線形天井を破った」初の事例になった。
- **agrifood は Gate L 不通過**。TAP-NODE+burst は relatedness は上回るが GEM full には届かない
  （むしろ僅かに下回る、有意差なし）。同じ burst 拡張・同じプロトコルでドメイン間の結果が
  割れており、「burst 条件付き trend 重み」の効果はドメイン依存であることが示唆される。
- **burst 交互作用 `c` の符号**: construction は seed 間で符号不安定（範囲 [-0.28, +0.33]、
  平均 -0.083 は事実上ゼロと区別できない）。agrifood は一貫して負（5 seed 中 4 つが負、
  平均 -0.418）——「バースト期には trend への感応度をむしろ下げる」方向で、
  GEM の mom×burst（正、罰則緩和）とは逆符号の効果に見える。ただし TAP-NODE の `c` は
  ソフトマックス混合重みの対数への寄与であり、GEM のロジット係数と直接同じ量ではないため、
  符号の対応を過度に解釈しない。construction の Gate L 通過が `c` 自体（burst 項）によるものか、
  単に TAP-NODE の基本構造（κ, b, h, α の4スカラー）が GEM の線形特徴より表現力があるだけかは
  未分離——**次の優先アブレーションは `c_init` を学習させない（c=0固定）版との比較**
  （burst 項の寄与を単離する）。
- **n=5 seed の注意**: 本プロジェクトの基準（≥5 seed）は満たすが、TAP-NODE holdout の前例
  （10 seed）より少ない。construction の p=0.018 は 5 seed でも報告基準の 0.05 を十分に
  下回るが、10 seed への拡張は取り消せない結論にする前の推奨事項として残す。

**結論**: 提案①は construction ドメインにおいて、**burst 条件付き TAP-NODE 拡張が
GEM 天井を上回るという、事前予想と異なる正の結果**を得た。§4 の事前予想（「a のみ効き
b/c/d は null」）のうち a（成長率の burst 条件付き Key 化）はここまでの結果と整合するが、
burst 項自体の寄与（c≠0 の効果）はまだ単離できていない。§8 の直近タスクを更新する。

## 6.2 Dual-Force full の再設計（v2、2026-07-22 実装）

Gate L 実行後、`dual_force_models.py` を提案①の原設計（引力・反発、成長率を Key に組み込む、
研究者ごとの Attention）によりコード忠実に合わせて書き直した。v1 との差分:

| 項目 | v1（これまでの実装・§1 の予備結果はこちら） | v2（今回改訂） |
|---|---|---|
| トピック位置 $P_j$ | エンコード前の生特徴 $x_j$ を線形射影 `P_proj` | エンコーダ潜在 $z_j$ を直接使用（**TAP-NODE と同じアンカー方式に統一**、detach） |
| トレンド情報 | $D_j^+, D_j^-$ を別々に保持 | 符号付き単一スカラー $D_j = M_j(t)-M_j(t-1)$ |
| Key | $K_j^+=W_K[P_j\Vert D_j^+]$, $K_j^-=W_K[P_j\Vert D_j^-]$（2系統） | $K_j = W_K(P_j+D_j)$（**1系統**、$D_j$をbroadcast加算） |
| Attention | $\alpha^+, \alpha^-$ を別々のsoftmaxで計算 | $\alpha_{ij}=\mathrm{softmax}_j(Q_i\cdot K_j/\sqrt{h})$ を1回計算し、$D_j$の符号でマスクして引力・反発に配分 |
| ベクトル場 | $v_{in}-|\gamma|v_{out}$（$v_{in},v_{out}$は別々のsoftmaxから） | $v_{in}-|\gamma|v_{out}$（**同じ$\alpha_{ij}$**をマスクで分配、式の形は同じだが中身が単純化） |

v1の予備結果（§1.1、平均AUC 0.635、P-NODEに負け）は**この改訂で無効**——構造が変わったため
再測定が必要。次のタスクは、v2をconstructionドメイン（Gate L通過済み）でholdout評価し、
TAP-NODE+burst（AUC 0.811）を上回るかを見ること（§8 タスク4）。

## 6.3 未解決の設計論点への対処と実測（2026-07-22〜23実行）

§6.1の未解決論点（①Attentionの捨てられる確率質量、②D_jの生スケール問題）それぞれに
2つの対処法を実装し、5設定（現行=raw／②Aのみ=learnable／②Bのみ=zscore／①B+②A=learnable_renorm／
①B+②B=zscore_renorm）を construction/agrifood で 3 seed（42, 7, 123）ずつ holdout 評価した。

**実装上の障害と対処**（両方とも construction の大トピック数 = 4,940 CPCコードに起因）:
- `diff = P_j.unsqueeze(0) - h_i.unsqueeze(1)` で (著者数, トピック数, 潜在次元) の3次元
  テンソルを明示的に作る素朴な実装は、GPUメモリを容易に使い切る（1プロセスで最大23GB超、
  RTX 3090 24GB でOOM）。**分配法則**
  $\sum_j \alpha_{ij}(P_j-h_i) = \alpha_i \cdot P - h_i \sum_j \alpha_{ij}$
  を使い、3次元テンソルを一切materializeしない行列積ベースの実装に書き換えて解決
  （`dual_force_models.py` forward()、以後この最適化はraw構成も含め全設定に適用済み）。
- ①B（マスク後の再正規化）で分母のclamp下限を`1e-8`にすると、成長側/衰退側の確率質量が
  ほぼゼロの研究者で最大1億倍近い増幅が起き、ODE積分中にNaNへ発散する事例を実測
  （`learnable_renorm`, construction, seed=7）。clamp下限を`0.05`に緩め、増幅率を
  最大20倍に制限して解決。

### 結果（holdout AUC、3 seed平均 ± SD）

| ドメイン | 設定 | AUC | vs GEM(0.766/0.772) | vs TAP-NODE+burst | 判定 |
|---|---|---:|---:|---:|---|
| construction | raw（現行） | 0.7054±0.0106 | −0.0604 | −0.1057 | 天井未達 |
| construction | learnable（②Aのみ） | 0.6977±0.0130 | −0.0681 | −0.1134 | 天井未達 |
| construction | zscore（②Bのみ） | 0.7661±0.0814 | +0.0003 (n.s.) | −0.0449 | 高分散・不安定 |
| construction | learnable_renorm（①B+②A） | 0.6775±0.0404 | −0.0883 | −0.1336 | 最下位 |
| construction | **zscore_renorm（①B+②B）** | **0.8349±0.0130** | **+0.0691 (p=0.017)** | **+0.0238** | **3/3 seedがGEM・TAP-NODE+burst双方を上回る** |
| agrifood | raw〜learnable_renorm | 0.653〜0.659 | 負 | 負 | 天井未達 |
| agrifood | zscore_renorm | 0.8245±0.1061 | +0.0529 (n.s., p=0.55) | +0.0779 | **高分散のため信頼不可**（seed42=0.675 vs seed7=0.909 と大きく割れる） |

**読み**:

- **construction の zscore_renorm は、この改訂で最も強い正の結果**。個別 seed
  （0.850, 0.818, 0.836）すべてが GEM（0.766）と TAP-NODE+burst の5seed平均（0.811）の
  両方を上回り、しかも TAP-NODE+burst（SD 0.023）よりタイトな分散（SD 0.013）。
  これは①（再正規化）と②（zscore標準化）の**どちらか一方だけでは効かず、両方を同時に
  適用して初めて効く**——`learnable_renorm`（①B単独に近い）は最下位、`zscore`（②B単独）は
  高分散で不安定、という他の4設定の結果と対照的。2つの論点は独立ではなく、
  **相互に補完し合う**ことが示唆される。
- **agrifood は同じ設定でも再現しない**（高分散）。fine-grained ドメインでも
  construction と agrifood でここまで挙動が割れるのは、TAP-NODE+burst の Gate L でも
  観測された「construction は通り agrifood は通らない」というドメイン依存パターン
  （§6.1）の繰り返し。
- 3 seed はこのプロジェクトの完全な確証基準（TAP-NODE同様の10 seed）に達していない。
  construction の zscore_renorm は現時点で最有力候補だが、**5〜10 seedへの拡張と
  対応あり検定での再確認が次の必須タスク**（TAP-NODE+burst との共通 seed {42,7,123} での
  対応差は +0.046, −0.021, +0.050 と符号混在で、方向性はあるが n=3 では有意性を主張できない）。

## 6.4 連続時間ODE vs 離散1ステップのアブレーション（2026-07-23実行、想定外の結果）

§4成分dの事前予想（「velocity attributionより、離散版と連続版は同等になる可能性が高い」）を
検証するため、TAP-NODE+burst（Gate L通過構成、construction）を **dopri5（適応的多段積分・既定）**
と **Euler法1ステップのみ**（`--ode-method euler --ode-n-steps 1`、$z_{new}=z+f(z)$を1回だけ）で
比較した。Gate Lと**同一の5 seed**（0,1,7,42,123）を使った単一仮説の対応あり検定
（複数設定を探索していない、§6.3で問題視した手続きとは異なる）。

| | dopri5（連続、既定） | Euler 1ステップ（離散） | 差 |
|---|---:|---:|---:|
| 平均AUC (5 seed) | 0.8111±0.0234 | 0.6488±0.0914 | **−0.1623** |
| 対応あり検定 | | | **p=0.0198**（5/5 seedすべて離散版が負け） |

**事前予想は外れた**。離散1ステップ版は全seedで明確に負けており、連続時間積分（または少なくとも
複数ステップでの数値精度）に実測できる価値がある。

**解釈上の注意**: これは「連続時間という概念自体が本質的に正しい」ことの証明ではない。
TAP-NODEの場は非線形性が強く（softmax重みが$z$の位置に依存）、Euler法1ステップ（刻み幅1.0）は
出発点だけで評価した勾配で一気に飛ぶためオーバーシュートしやすい——dopri5は適応的に細かく
刻んで軌道を追跡する。**「連続時間性」と「数値積分の分割の細かさ」が交絡している**ため、
まだ「velocity attributionの結論（連続フロー抽象は誤り）」を覆したとは言い切れない。

**次に必要な切り分け**: 固定刻みでステップ数を増やした版（`--ode-method euler --ode-n-steps 8`
または `rk4 --ode-n-steps 4`）を追加で比較する。ステップ数を増やすだけで dopri5 に近づくなら
「数値精度の問題」（velocity attributionと矛盾しない）、増やしても届かないなら
「適応的な連続時間性そのものが必要」という、より強い主張になる。

## 6.5 正しい検証遷移/テスト遷移分割での再検証（2026-07-23、§6.3の欠陥を修正）

§6.3のzscore_renorm(AUC 0.835, 3 seed)は「5設定を同一テスト遷移(2020→2021)に投入して
事後的に選んだ」という手続き上の欠陥があった（テストセットでのモデル選択）。
正しい手順でやり直した:

- **Phase 1（検証遷移 2019→2020, 3 seed）**: 5設定を construction・独立した年遷移で比較。
  結果: raw 0.684 / learnable 0.689 / zscore 0.751 / learnable_renorm 0.686 /
  **zscore_renorm 0.811**（最良）。§6.3と同じ順位が**独立した遷移でも再現**——
  これ自体が§6.3の結果が偶然ではないことの傍証。
- **Phase 2（テスト遷移 2020→2021、1回のみ、5 seed）**: 検証遷移で選ばれた zscore_renorm
  **だけ**を、TAP-NODE+burstと同じ5 seed（0,1,7,42,123）で評価。

| 比較 | 平均AUC | 検定 | 判定 |
|---|---:|---|---|
| zscore_renorm (0.8468±0.0196) vs GEM (0.7658) | 一標本t検定 t=8.27, **p=0.0012** | **有意に上回る（確定）** |
| zscore_renorm vs TAP-NODE+burst (0.8111±0.0234) | 対応あり t検定 t=2.28, **p=0.0845** | **有意水準未達**（4/5 seedで上回る、平均差+0.036） |

**結論（§6.3からの更新）**: Dual-Force v2 + zscore_renorm は、**正しい手続きの下でも Gate S（GEM）
を明確に上回る**——Gate L相当の基準を独自に満たした2つ目のモデルとして確定して良い。
一方「TAP-NODE+burstより優れている」という主張は、**方向は一貫しているが（4/5 seed勝ち）
n=5では統計的に確定しない**（p=0.085）。§6.3時点の「TAP-NODE+burstを上回った」という
表現は、GEMに対しては確定・TAP-NODE+burstに対しては未確定、と修正する。

**次に必要なら**: TAP-NODE+burstとの優劣を確定させるにはseed数を8〜10に拡張し検出力を上げる。
ただし研究上の主要な問いへの答えとしては、「引力・反発Attention構造（Dual-Force）は、
数値的健全化（②Bのzscore標準化＋①Bの再正規化）を施せば、少なくともTAP-NODE+burstと
同等以上の予測力を持つ」で十分——単純なTAP-NODEに対する明確な"敗北"は避けられた。

## 6.6 Attention崩壊の発見とランク変換による修正（2026-07-29、現チャンピオン更新）

### 発見: zscore_renormはAttentionが実質2トピックに崩壊していた

速度場を可視化する過程で、zscore_renormモデルの挙動を診断したところ、82,561社中
**86.8%が同じ1トピック**（衰退, D_j=−99）、**12.9%がもう1トピック**（成長, D_j=+108）
に最大Attentionを置いており、4,931トピック中**誰かの1位になったのはわずか4トピック**
だったことが判明した。z-score標準化は外れ値の相対的な大きさをそのまま残すため、
極端な生のトレンド値（$D_j=-99, +108$など、典型的な値幅の桁違い）がKeyの計算
$K_j=W_K(P_j+\hat D_j)$を支配し、企業の位置$h_i$にほぼ関係なくAttentionが
この2トピックに張り付いていた。§6.5までの「zscore_renormがTAP-NODE+burstを上回る」
という結果は、位置に応じた繊細なAttentionというより、**この2つの極端なトピックまでの
距離をほぼ見ているだけ**という、かなり単純化されたメカニズムで達成されていた可能性が高い。

### 対処: ランク（分位点）変換

外れ値対策の統計学的定番手法である quantile normalization を`d_scale_mode="rank"`
として追加した:

$$\hat D_j = \Phi^{-1}\!\left(\frac{\mathrm{rank}(D_j)-0.5}{J}\right)$$

z-scoreと異なり、成長・衰退の**順序関係は完全に保持**しつつ、**最大値の大きさを有界**にする
（同じ極端な外れ値でも、1位と1000位の差が無限に発散しない）。

### 再検証（正しい検証遷移→テスト遷移の手順を厳守）

**Phase 1（検証遷移 2019→2020, 3 seed）**: rank単独 0.795±0.040、
**rank_renorm 0.871±0.0045**（zscore_renormの0.811を上回り、分散は約21分の1）。

**Phase 2（テスト遷移 2020→2021, 5 seed, 1回のみ）**: rank_renormのみを評価。

| 比較 | 平均AUC | 検定 | 判定 |
|---|---:|---|---|
| rank_renorm (0.8638±0.0109) vs GEM (0.7658) | 一標本t検定 p=0.0001 | **極めて有意** |
| rank_renorm vs TAP-NODE+burst (0.8111±0.0234) | 対応あり t検定 **p=0.0291, 5/5 seed勝利** | **有意に上回る（プロジェクト初）** |
| rank_renorm vs zscore_renorm (0.8468±0.0196) | 対応あり t検定 p=0.172 | 方向は上だが有意差なし |

§6.5のzscore_renormはTAP-NODE+burstに対しp=0.085で有意水準未達だったが、
**rank_renormは初めてこの壁を越えた**（p=0.029）。分散もzscore_renormの約半分。

## 6.7 共有エンコーダ・ベースライン比較（2026-08-06実行、rank_renormの優位性が消滅）

### 動機

§6.6までの全ての比較（vs GEM, vs TAP-NODE+burst）は、比較対象が別のエンコーダ・別の学習
パイプラインを持つモデルだった。Dual-Force・TAP-NODE・static・RNN・NeuralODE・P-NODEは
実は全て`run_benchmark_comparison.py`経由で**同一の共有GAT-VGAEエンコーダ**を土台にでき、
エンコーダ＋時間発展モジュール＋デコーダをend-to-endで同一の未来リンク予測損失で学習する
（`unified_training.py`）。この共有エンコーダ自体がどれだけ「答えを吸収」できるかを切り分け
るため、時間発展モジュールだけを static（恒等写像・パラメータ0）/ RNN+VGAE / 素のNeuralODE /
P-NODE（幾何デコーダのみ）に差し替えたベースライン群を、rank_renormと同じ5 seed
（0, 1, 7, 42, 123）で比較した。

### 結果: construction, テスト遷移2020→2021, 5 seed

```
static       : 0.8607 ± 0.0041
RNN+VGAE     : 0.8563 ± 0.0166
NeuralODE    : 0.8616 ± 0.0031
P-NODE       : 0.8617 ± 0.0037
rank_renorm  : 0.8638 ± 0.0109   （§6.6の現チャンピオン）
```

対応ありt検定（同一5 seed、rank_renorm vs 各ベースライン）:

| 比較 | 平均差 | t | p値 | rank_renormの勝ち数 |
|---|---:|---:|---:|---:|
| vs static | +0.0031 | 0.599 | 0.582 | 2/5 |
| vs RNN+VGAE | +0.0075 | 0.829 | 0.454 | 2/5 |
| vs NeuralODE | +0.0022 | 0.389 | 0.717 | 3/5 |
| vs P-NODE | +0.0021 | 0.421 | 0.695 | 3/5 |

**全て非有意（p>0.45）、勝敗もほぼコイントス（2〜3/5 seed）。**

### 結論（§6.6までの結果の再解釈）

rank_renormはGEM（訓練不要ベースライン）には極めて有意に勝つ（p=0.0001, §6.6）。しかし
**時間発展を一切しないstaticを含む、同一エンコーダ上のどのベースラインに対しても統計的優位性
を示せなかった**。つまり§6.1〜§6.6で確認してきた「GEMを上回る」効果の大部分は、Dual-Force
固有の引力・反発Attention機構ではなく、**共有GAT-VGAEエンコーダ自体**（4層GATメッセージパッシ
ングが今年のグラフ構造から来年の参入先を先読みする能力）に由来する可能性が高い。

機構的な理由（`unified_training.py`のコード確認による）: エンコーダ・時間発展モジュール・
デコーダは1つのAdam最適化器（`torch.optim.Adam(model.parameters())`）でend-to-end学習され、
未来リンク予測損失の勾配はエンコーダの重みまで遮断なしに逆伝播する
（`train_one_epoch`, `z_t = model.encode(...)` に`.detach()`が無い）。static
（時間発展モジュールのパラメータがゼロ）の場合、この勾配の**全量**がエンコーダに吸収される。
時間発展モジュールに自由度があっても同じ経路は残り続けるため、「エンコーダに答えを埋め込む」
という抜け道が常に開いている。これがstatic/RNN/NeuralODE/P-NODE/Dual-Forceの成績が
軒並み横並びになる直接的な原因と考えられる（未検証の対処案: エンコーダ出力と時間発展モジュール
の間にstop-gradientを挿入し、エンコーダを純粋な「今年の再構成」だけで学習させた上で再比較する）。

### この結果が①研究ライン全体に持つ意味

これは本プロジェクトの「simple-beats-complex」パターンの中でも特に厳密な実例である。
GEMという訓練不要ベースラインは超えたが、それは複雑なAttention機構の手柄ではなく、
共有エンコーダの汎用的な表現力の手柄である可能性が高い、というのが5 seedの統計的検証を
経た現時点の結論。Dual-Force固有の付加価値は、この評価設計の範囲では実証できていない。

### Attention崩壊は「軽減」されたが「解消」はしていない（正直な限界）

同じ診断をrank_renormのseed=42チェックポイントに対して再実行した:

| | zscore_renorm | rank_renorm |
|---|---:|---:|
| 1位になったユニークトピック数 | 4 / 4,931 | 8 / 4,931 |
| 最大シェアのトピック | 86.8% | 41.2% |
| 上位3トピックの合計シェア | ほぼ100%（実質2トピック） | 91.7% |

集中度は大きく改善したが、**依然として少数の極端なトピックにAttentionの大半が
集中する構造は残っている**。ランク変換は外れ値の「大きさ」を有界にしたが、
「極端なトレンドを持つ少数のトピックが選ばれやすい」という順序自体は意図的に保持している
ため、崩壊の完全な解消ではなく軽減にとどまる。この点は今後の課題として残す
（例: Attentionのエントロピー正則化、top-kマスキングなど）。

### 現チャンピオンの更新

**rank_renorm（construction, d_scale_mode="rank" + renorm_masked_attention）が新チャンピオン。**
zscore_renormより高い平均・大幅に低い分散・TAP-NODE+burstへの初の有意な勝利、という
3点で上回る。ただしzscore_renorm自体との直接比較では有意差はまだない（p=0.172）ため、
「rank_renormがzscore_renormより真に優れている」との主張は今後もう少しseed数を
増やしてから確定させるのが望ましい。

**位置づけの更新**: Dual-Force v2 は §7 で想定した「TAP-NODE+burst に投資判断を委ねる」段階から、
**construction 限定で TAP-NODE+burst を上回る候補が見つかった**段階に進んだ。ただし
勝因が「Attentionによる研究者ごとの興味推定」自体（§4 成分b）にあるのか、それとも
①②の数値的な健全化だけによるものかはまだ切り分けられていない——次のアブレーションは
Attention を無効化（Query を全著者共通の定数に固定）した対照を zscore_renorm 設定に
追加することで、この2つを分離する。

---

## 7. どちらに転んでも論文になる設計

- **Gate 通過（正の場合）**: 「trend momentum を burst で regime-conditioning すると、fine-grained
  技術参入予測に実際に増分がある」という、predictability map の C2（HOW-MUCH は無信号）に
  **例外を与える**新規性のある正の結果。TAP-NODE 系譜の自然な拡張として KDD/WWW の主結果に
  組み込む。
- **Gate 不通過（TAP-NODE+burst で頭打ち、Dual-Force full は不要と判明）**: 「アテンション機構による
  表現力の追加は、trend 情報をスカラー化するだけで代替できる」という10件目の
  simple-beats-complex として、Limits of Predictability ペーパー（`WHY_NEURALODE_FAILS_ja.md` 案A）
  の証拠表に追加する。この場合でも researcher-wise アテンション重みは「誰が成長を追い誰が守りに
  入るか」の**可視化・診断**（案B/C路線）として転用できる（α+/α− を KG-ATLAS の persona ごとの
  技術関心プロファイルに使う）。

## 8. 直近のタスク（優先順、2026-07-23 更新）

**完了**: holdout 対応、burst 特徴実装、Gate S（既存 `gem_skill.py`）、Gate L
（construction PASS / agrifood NOT PASSED、§6.1）、Dual-Force v2 の①②論点アブレーション
（construction zscore_renorm が TAP-NODE+burst を上回る、§6.3）。

1. **construction zscore_renorm を 5〜10 seed に拡張**（現状3 seed）し、TAP-NODE+burstとの
   対応あり検定で有意性を確認する（最優先——§6.3 の最有力候補の頑健性確認）。
2. **Attention無効化アブレーション**（zscore_renorm設定のまま、Queryを全著者共通の定数に固定）:
   zscore_renormの勝因が「研究者ごとのAttention」（§4成分b）自体にあるのか、①②の数値的健全化
   だけによるものかを切り分ける。
3. **`c=0` 固定アブレーション**（TAP-NODE側、construction, 5 seed）: burst 項自体の寄与を単離する。
4. **agrifood の診断**: zscore_renormが高分散（seed42=0.675 vs seed7=0.909）で信頼できない理由を
   調べる（学習率・epoch数のスイープ、agrifood固有のburst定義の妥当性確認）。
5. §4.1 の ODNet/UniGO 型単一関数版・TI-ODE 型基底数アブレーションは、1.の頑健性確認後に着手。

## 9. 位置づけ

本仕様は Dual-Force を否定する仕様ではなく、(i) すでに実装済みのコードを本プロジェクトの
査読基準（holdout・多 seed・天井比較・成分分解・事前登録ゲート）に載せる工程表、
(ii) 提案①の4主張のうち、証拠に支持されるもの（burst 条件付き成長シグナル）だけを
生き残らせるフィルタ、(iii) 「引力・反発」という枠組みを opinion dynamics × GNN
（ODNet/UniGO, WWW 2025）と技術代替モデル（Fisher-Pry 1971, Morris & Pratt 2003）という
**2つの確立された文献系譜に明示的に接続する**工程表、である（`docs/RELATED_WORK_SURVEY.md`）。
Dual-Force という名前・アーキテクチャそのものに新規性を主張することは本仕様の下では保留とし、
新規性は「どの成分が実際に増分を持つか」の検証結果と、「集計競合ダイナミクス（LV/Fisher-Pry）を
個体レベルの学習可能な潜在流に持ち上げた」という位置づけに依存させる。

## 6.8 ノード種別・ドメインを変えた頑健性チェック（2026-08-06、seed=42単発）

§6.7の「rank_renormがstaticに勝てない」という結果が、construction・発明者ノード・特定seedに
固有の現象でないかを、ノード種別とドメインを変えて確認した（いずれもseed=42のみの単発チェック、
5seedでの確定検証はまだ）。

**ノード種別（construction, 発明者→企業, `bipartite_construction_firm.csv`）**:
```
P-NODE       0.8900
RNN+VGAE     0.8836
static       0.8795
NeuralODE    0.8784
rank_renorm  0.8762   ← 5モデル中最下位
```

**ドメイン変更（agrifood, 発明者）**:
```
static       0.9175   ← 最高
NeuralODE    0.9130
P-NODE       0.9121
rank_renorm  0.9063
RNN+VGAE     0.8555   ← 最下位（agrifoodはRNNが弱い、7月の検証Aと整合）
```

**pharmaドメイン**: 469,617ノードと非常に大きく、staticのみ完走（AUC=0.8463）。
NeuralODE/P-NODEはGPU 24GB環境でも学習開始直後にOOM——このパイプラインの現実装は
大規模ドメインにスケールしない（アーキテクチャ上の制約、既知の課題として記録）。

### 結論

**3つの独立した設定（construction/発明者、construction/企業、agrifood/発明者）すべてで
rank_renormはstaticを上回れていない**（agrifoodでは明確に下回る: 0.9063 vs 0.9175）。
§6.7の結果は特定条件に固有の偶然ではなく、ノード種別・ドメインを変えても再現する頑健な
パターンである可能性が高い。ただしいずれも単発seedであり、5seedでの統計的確認はまだ。
