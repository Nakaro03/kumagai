# 関連研究サーベイ — Dual-Force / 出願数予測の設計への反映

**目的**: `DUAL_FORCE_REDESIGN.md`（提案①）と `FILING_COUNT_FORECAST_DESIGN.md`（提案②）を、
本プロジェクトの内部証拠だけでなく **査読で照合される可能性が高い外部文献**に接続する。
`COMPARISON_METHODS.md` と同じ規律で、各文献を **確認できた掲載先の確度でティア分け**する
（本プロジェクトはこれまでも実装 docstring の誤記載 [EvolveGCN の venue 誤り] を自ら発見・訂正した
実績があり、ここでも未確認の venue を確定済みとして書かない）。

---

## ティア1 — トップカンファレンス／トップジャーナル掲載が確認できたもの

| 文献 | 掲載先 | 本設計への意味 |
|---|---|---|
| Kleinberg, J. (2002). *Bursty and Hierarchical Structure in Streams.* | **KDD 2002**（journal版: *Data Mining and Knowledge Discovery* 7(4), 2003, DOI:10.1023/A:1024940629314） | ②の Gate 0（バースト検出）を、状態遷移オートマトン＋動的計画法という**正式な推定手続き**に置き換える候補。テキスト・出版物ストリームのバースト検出における標準ツール |
| Zhao, Q., Erdogdu, M. A., He, H. Y., Rajaraman, A., Leskovec, J. (2015). *SEISMIC: A Self-Exciting Point Process Model for Predicting Tweet Popularity.* | **KDD 2015** | ②のバースト・レジーム検出をHawkes型自己励起過程として定式化する代替案。カスケードの「これから伸びるか」を確率過程のパラメータ（re-tweet rate等）から予測する枠組みは、出願数のburst検出に転用できる |
| Zhou, G. et al. (2019). *Deep Interest Evolution Network for Click-Through Rate Prediction.* | **AAAI 2019** | ①の「researcher-wise の興味の強さをアテンションで推定する」という主張と最も近い、実運用実績（Taobao, CTR+20.7%）のあるアーキテクチャ。Interest Extractor（GRU）+ Interest Evolving（AUGRU, attentional update gate）の二層構造は、Dual-Forceの一層アテンションより**興味の時間発展を明示的にモデル化**しており、成分bの検証（researcher-wise attentionは本当に個人差を捉えているか）の対照として使える |
| Luo, L. et al. (2023). *Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs* (GSNOP). arXiv:2211.08568 | **WSDM 2023**（DOI:10.1145/3539597.3570465, コード: github.com/RManLuo/GSNOP） | 本プロジェクト最大の実務的困りごと（holdout AUCのseed分散が最大28pt）に**直接対応**。疎な動的グラフでの過学習をNeural Process的定式化（点推定でなく関数上の分布）で緩和する設計は、Dual-Force/TAP-NODE双方にアーキテクチャとして移植可能 |
| Lv, O., Zhou, B. et al. (2025). *UniGO: A Unified Graph Neural Network for Modeling Opinion Dynamics on Graphs.* arXiv:2502.11519 | **WWW 2025**（*Proc. ACM Web Conf. 2025*, pp.530–540） | 提案①の核心（引力・反発の学習）が**確立された研究分野の一部**であることを示す最重要文献。coarsen-refineによりopinion dynamicsの均衡状態を保ちながらover-smoothingを回避、合成データでの事前学習が実データへ汎化することを実証済み。前身: Lv et al. (2023) *A Unified View on Neural Message Passing with Opinion Dynamics for Social Networks* (ODNet), arXiv:2310.01272（区分関数φ(sim)による同質性=引力・異質性=反発の統一的定式化。venue未確認、UniGOの前身研究として位置づけ） |

## ティア2 — 分野の確立済み基礎文献（会議論文ではないが、査読者が前提知識として期待する）

| 文献 | 掲載先 | 本設計への意味 |
|---|---|---|
| Bass, F. M. (1969). *A New Product Growth for Model Consumer Durables.* | *Management Science* 15(5), 215–227 | 技術拡散のS字カーブモデルの原点。②のPhase 1でNeural ODE/LSTMを再発明する前に検討すべき最も基本的なベースライン |
| Fisher, J. C., Pry, R. H. (1971). *A Simple Substitution Model of Technological Change.* | *Technological Forecasting and Social Change* 3, 75–88（被引用1,000件超） | 「新技術が旧技術を置き換える速度は、残存する旧技術のシェアに比例する」という、技術予測分野で半世紀使われてきたロジスティック代替モデル |
| Morris, S. A., Pratt, D. (2003). *Analysis of the Lotka–Volterra competition equations as a technological substitution model.* | *Technological Forecasting and Social Change* 70, 103–133 | **提案①の`v_in − γ·v_out`と数式構造が同型**の、集計レベル技術競合モデル。引力・反発を「種間競争」として定式化する半世紀の蓄積がある |
| Meade, N., Islam, T. (2006). *Modelling and forecasting the diffusion of innovation – A 25-year review.* | *International Journal of Forecasting* 22, 519–545（被引用700件超） | Bass/Fisher-Pry/Lotka-Volterra系モデル群の標準的なレビュー。関連研究節の一次参照先として使える |
| Hausman, J., Hall, B. H., Griliches, Z. (1984). *Econometric Models for Count Data with an Application to the Patents-R&D Relationship.* | *Econometrica* 52(4), 909–938 | 特許カウントデータの標準的な尤度（Poisson/負の二項）。②のPhase 1で「ガウスMSEで出願数を回帰する」という初心者的な誤りを避ける根拠 |
| Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks.* | *International Journal of Forecasting* 36(3), 1181–1191（arXiv:1704.04110, Amazon） | カウントデータ向け負の二項尤度＋自己回帰RNNの産業実績あるデファクト標準。②のGate L候補モデルとして、独自アーキテクチャより先に検討すべき |

## ティア3 — 直近のarXivプレプリント（掲載先未確認、参考情報として扱う）

| 文献 | 状態 | 本設計への意味 |
|---|---|---|
| Wang, X., Wang, Z., Liang, J., Zhao, X., Dang, C., Jin, Z., Liang, J. (2026). *Time-varying Interaction Graph ODE for Dynamic Graph Representation Learning* (TI-ODE). arXiv:2604.24811 | 2026-04投稿、掲載先未確認 | グラフODEの発展関数を**複数の学習可能な相互作用基底関数**の時間変化する重み付き和に一般化。Dual-Forceの「引力・反発の2基底」はこの一般形の意味論固定・K=2の特殊ケースとみなせる |
| Ofer, D., Linial, M. (2023). *What's next? Forecasting scientific research trends.* | *Heliyon*（Cell Press, arXiv:2305.04133）— 会議論文ではなくジャーナル論文だが直接の先行研究 | 125トピック・40年でトピック人気度を5年先まで予測。**特許は科学トピックの先行指標**、**レビュー/原著比率が衰退シグナル**という2つの具体的知見は②のburst特徴設計にそのまま使える |

---

## v2設計への反映まとめ

### `DUAL_FORCE_REDESIGN.md` への追加ゲート候補・アブレーション軸

1. **ODNet/UniGO型の単一関数版**を Gate S と Gate L の間に新設: `alpha_ij = phi(sim(h_i, P_j), D_j)`
   という区分関数1本で引力・反発を表現する版（Dual-Forceの二系統アテンションより軽量）。
   TAP-NODE（意味論固定・パラメータ4個）とDual-Force full（自由アテンション）の**中間の複雑さ**を
   埋める、証拠に基づく新しい対照。
2. **TI-ODE型の基底数アブレーション**: 「意味論固定K=2」（現行Dual-Force）vs
   「意味論自由K=2」vs「K>2自由基底」を比較し、意味論を固定すること自体が性能を犠牲にしていないかを見る。
3. **GSNOP型のNeural Process化**を、10 seed平均で分散を押さえ込む前にまず試す
   （§1.1のseed分散28pt問題への直接対応、アーキテクチャ変更で分散そのものを減らす）。
4. 関連研究の物理的背景を Waddington地形（生物学）から **Fisher-Pry / Lotka-Volterra
   （技術予測の一次文献）**に差し替え、「集計競合ダイナミクスを個体レベルの学習可能な潜在流に
   持ち上げた」という新規性の主張をそこに置く。

### `FILING_COUNT_FORECAST_DESIGN.md` への追加

1. **Gate 0の推定手続きを形式化**: Kleinberg (2002) のバースト検出オートマトン、または
   Hamilton型2状態Markov-switching回帰（filtered probability使用）のどちらかを正式な
   burst_j(t)の推定法として採用する（素朴な閾値分割をやめる）。filtered probabilityは
   t時点までの情報のみを使うため、P4（リークフリー）の要件を構造的に満たす。
2. **クロスソースlead-lag特徴をburst_j(t)に追加**: Ofer & Linial (2023) の「特許は科学トピックの
   先行指標」「レビュー比率が衰退シグナル」という知見を、KG-ATLASのhype-vs-substance/lead-lag
   路線と接続する形でGate 0の特徴量に組み込む。単一ソースの出願数momentumだけでは
   拾えない情報である可能性がある。
3. **Phase 1のモデル選定を独自実装からDeepAR（負の二項尤度）に変更**、HHG84をカウント尤度の
   一次根拠として明記する。
4. **Fisher-Pry / Lotka-Volterra競合方程式をGate Sのもう一つの候補**として追加: 関連CPCコード間の
   結合方程式（一方の成長が他方を食う）が、単変量momentum×burst回帰よりchange-on-changeを
   説明できるかを検証する。単変量で信号ゼロという結果を「局所競合が見えていないだけ」という
   仮説で説明できるかを試す、低コストな追加実験。

---

## 注記（引用の確度について）

ティア3の文献は本セッションのWeb検索で発見した直近のプレプリントであり、掲載先・査読状況は
未確認。本文・投稿版に引用する場合は、投稿時点で再度 arXiv/DOI と掲載状況を確認すること
（`COMPARISON_METHODS.md` の運用方針と同じ）。特に TI-ODE (arXiv:2604.24811) は2026年4月投稿で
本プロジェクトのセッション時点（2026年7月）でもまだ新しく、正式な査読結果は確認できていない。
