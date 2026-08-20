# トピック出願数予測 設計仕様 — Regime-Gated Filing-Count Forecasting

**Status**: 提案②（トピックの出願数予測）は、本プロジェクトの検証B（`predictability_ceilings.py`、
`docs/RESEARCH_PLAN_PREDICTABILITY.md`）と**同じ問題設定**（HOW-MUCH: I_j(t+1) の予測）にすでに
大きく重なる。検証Bは訓練不要の診断としてほぼ答えを出している——このままモデルを増やしても
X5 / DRIFT / TAP change-on-change の二の舞になる。本書は②を「もう一つの Neural ODE 予測器を作る」
案から、「検証Bが見つけたレジーム構造を使って、予測可能な場所だけを gated forecasting する」案に
組み替える。

- 既存インフラ: `predictability_ceilings.py`（訓練不要天井）、`predictability_popularity_auc.py`
- 既存結果: `outputs/predictability_map/RESULTS.md`（検証 B, C, GEM）
- 参照: 技術トレンド予測の verdict 群（growth-rate 予測は momentum/persistence に負ける）、
  DRIFT go/no-go（過学習で失敗）、predictability map（HOW-MUCH は無信号）、
  `docs/RESEARCH_PLAN_PREDICTABILITY.md`
- 外部文献サーベイ（トップ会議・確立文献との接続）: `docs/RELATED_WORK_SURVEY.md`。
  §5・§9 の変更点はこのサーベイの反映

---

## 1. すでにわかっていること（検証Bの実測、再投資前に必ず踏まえる）

7ドメイン×年遷移での Spearman 相関（`outputs/predictability_map/RESULTS.md` より）:

| ドメイン | 人気度天井 ρ(M_t,I_t+1) | change-on-change |
|---|---:|---:|
| agrifood | +0.92 | −0.20 |
| construction | +0.93 | −0.23 |
| energy | +0.97 | −0.17 |
| semiconductor | +0.96 | −0.00 |
| pharma | +0.97 | −0.01 |
| **computing** | +0.98 | **+0.34±0.17**（5/5 遷移すべて正） |
| author_topic | +0.91 | −0.17 |

読み方はすでに確定している。「来年どのトピックの出願数が多いか」は今年の出願数だけでほぼ決まる
（ρ0.91〜0.98）——これは予測ではなく自明な慣性。「今年の伸び率が来年の伸びの変化を予測するか」
（change-on-change、正味の予測情報）は7ドメイン中6つでゼロ〜負。**唯一の例外が computing**
（AI/ML ブーム 2016–2021 に相当、5遷移すべて正）。

これは growth-rate 予測全般の verdict（X5 LOO≈0）・DRIFT go/no-go（過学習で失敗）と完全に整合する。
**「トピックの出願数予測」を額面通りの時系列回帰タスクとして新しいモデルにやらせても、
6/7ドメインで同じ壁にぶつかる公算が高い。**

## 2. 中心的な問い: computing の例外はリアルタイムに検出できるレジームか、それとも後知恵か

これが②全体の成否を決める、唯一かつ最重要の分岐点である。

- **もし後知恵でしか分からない**（＝「AI/ML ブームだった」と今から振り返るから正に見えるだけで、
  2016年時点で computing を他ドメインと区別する特徴がない）なら、②は予測タスクとして成立しない。
  検証Bの結果を予測可能性マップの1セルとして記録するだけで終わる。
- **もしリアルタイムに検出できる**（＝バースト初期の2〜3年の加速度パターンが、他ドメインの
  非バースト期と統計的に区別可能で、かつその区別が"後から見た正解ラベル"を使わずに得られる）なら、
  ②は「どのトピック・どの時期が regime-informative か」を判定するゲート付き予測器として成立しうる。

**この問いに答える実験を Gate 0 として最優先で行う（§5）。モデル設計（Phase 1）は Gate 0 通過後。**

## 3. なぜ額面通りの「予測モデル」に飛びついてはいけないか（統計的な落とし穴3点）

査読で確実に突かれる、かつ本プロジェクトのドキュメント群にまだ明記されていない3点。

1. **出願数はカウントデータ**。ガウス MSE/RMSE で回帰すると過分散や0過多を無視する。
   Poisson / 負の二項回帰（Hausman, Hall & Griliches 1984 の特許カウントモデルが標準的な参照点）を
   尤度に使い、評価も Poisson deviance / sMAPE で報告する。
2. **公開ラグによる右打ち切り**。多くの特許制度では出願から約18ヶ月後に公開される。直近1〜2年の
   「出願数」はまだ公開されていない出願を含まないため系統的に過小になる。これを見逃すと、
   データ境界付近で「トピックが衰退している」という偽シグナルを拾い、change-on-change の符号が
   汚染される。評価ウィンドウはデータ境界から十分（公開ラグ相当年数以上）離すか、既知の打ち切り率で
   補正する。**`predictability_ceilings.py` の I_j(t+1) 定義がこれを考慮しているかを最初に監査する
   （未確認・要着手）。**
3. **レジーム判定自体のリーク**。「computing はブームだった」というラベルを使って学習・評価データを
   選ぶと、Gate 0 の答えが循環論法になる。バースト指標は**判定対象の遷移より前の情報だけ**
   （t 以前の加速度・分散）で構築し、正解ラベル（実際に伸びたかどうか）を一切参照しない。

## 4. 設計原則

- **P1: モデルを増やす前にレジームを分類する。** Phase 0（診断）→ Phase 1（予測器）の順を守り、
  Phase 1 は Phase 0 で「momentum-informative」と判定されたセルにのみ投資する。
- **P2: カウントデータの尤度を正しく使う**（§3-1）。
- **P3: 公開ラグを明示的に扱う**（§3-2）。
- **P4: バースト指標はリークフリー・実時間構築**（§3-3）。
- **P5: ベースライン階層を必ず先に確定する**: persistence → 線形トレンド外挿（Holt/ETS）→
  momentum×burst 回帰（GEM と同型）→ 学習モデル。学習モデルはこの階層を Gate S/Gate L で
  上回って初めて主張に使う（ASPH-Flow、Dual-Force v2 と同じ規律）。

## 5. フェーズ設計

### Phase 0 — レジーム分類（`predictability_ceilings.py` の拡張、低コスト・訓練不要）

`RESEARCH_PLAN_PREDICTABILITY.md` の次段3・4（大規模3ドメイン拡張、粒度スイープ）をそのまま
このフェーズの作業項目として使う。追加すること:

- 全ドメイン×granularity×**5年ローリングウィンドウ**で change-on-change を再計算し、
  「どの (ドメイン, 粒度, 時期) セルが持続的に正か」を地図化する（computing 以外にも
  semiconductor や pharma の一部期間にサブレジームが眠っている可能性がある）。
- バースト指標 `burst_j(t)` を **t 以前のみ**の情報で定義する（例: 直近3年の出願数の二階差分、
  または成長率の分散）。
- **Gate 0**: burst_j(t) が高い (トピック, 年) の集合について、change-on-change ρ を
  burst_j(t) が低い集合と比較する。computing 以外の**独立したドメイン・時期**でもバースト高群の
  change-on-change が有意に正なら、Gate 0 通過（レジームはリアルタイム検出可能）。computing だけで
  しか再現しないなら Gate 0 不通過（後知恵の疑い強）——この場合は Phase 1 に進まず、検証Bの結果
  として確定させる。

### 5.1 burst_j(t) の推定方法を素朴な閾値分割から正式な手続きに置き換える（`RELATED_WORK_SURVEY.md`）

当初案（burst_j(t) の高低で単純に2群分割）は査読で「恣意的な閾値」と指摘されやすい。
確立された2つの代替を Gate 0 の実装として採用する。

- **Kleinberg (2002, KDD) のバースト検出**: 状態遷移オートマトン＋動的計画法で、
  「低頻度状態→高頻度状態→低頻度状態」という遷移コスト最小の状態列を求める。
  テキスト・出版物ストリームのバースト検出における標準ツールで、閾値を人手で決めない。
- **Hamilton 型 2状態 Markov-switching 回帰**: 各年の出願数系列に2レジーム
  （通常/バースト）の切替モデルを当てはめ、**filtered probability**（t 時点までの情報のみで
  計算するレジーム確率、smoothed probability ではない）を burst_j(t) として使う。
  filtered probability を使うことで P4（リークフリー）の要件を推定手続きレベルで保証できる。

いずれかを Gate 0 の正式な burst_j(t) 定義として採用し、素朴な閾値分割は感度分析としてのみ残す。

### 5.2 burst_j(t) にクロスソース lead-lag 特徴を追加する

Ofer & Linial (2023, *Heliyon*) の "What's next?" は、125トピック・40年のスケールで
科学トピックの人気度を5年先まで予測し、(a) **特許出願が科学トピックの先行指標になる**、
(b) **レビュー論文/原著論文の比率が衰退シグナルになる**ことを報告している。本プロジェクトの
burst_j(t) を出願数の自己momentum（単一ソース）だけで定義するのではなく、KG-ATLASの
hype-vs-substance/lead-lag路線（ニュース・論文・特許のクロスソース時差）と接続した特徴を
追加候補として Gate 0 に含める。単一ソースの change-on-change が無信号でも、
クロスソースの時差情報には残っている可能性がある——これは検証Bでまだ試されていない
「未踏フロンティア」に該当する。

### Phase 1 — Gate 0 通過セル限定の gated forecaster（Gate 0 不通過なら実施しない）

- 対象: Gate 0 で正と確認された (ドメイン, 粒度, 時期) セルのみ。
- モデル: momentum×burst 交互作用を持つ Poisson/NegBin GLM（GEM skill score の回帰式をカウント
  予測に転用、GEM の学習係数がそのまま初期値の参考になる）。ニューラルモデルへの投資が
  Gate S を超えて正当化された場合も、独自アーキテクチャを新規に設計するのではなく
  **DeepAR**（Salinas et al. 2020, *International Journal of Forecasting*; 負の二項尤度の
  自己回帰RNN、カウントデータ予測の産業実績あるデファクト標準）を第一候補とする
  （§4 P5, `docs/RELATED_WORK_SURVEY.md`）。
- **もう一つの Gate S 候補（結合方程式）**: 関連 CPC コード間の競合を明示的にモデル化する
  Fisher-Pry (1971) / Lotka-Volterra 技術代替方程式（Morris & Pratt 2003）を、単変量
  momentum×burst 回帰と並ぶ Gate S 候補として追加する。単変量の change-on-change が
  無信号なのは、「一方の成長が関連コードの出願を食う」ゼロサム的な局所競合が単変量分析では
  見えないためという仮説を検証する、低コストな追加実験。
- ①（Dual-Force / TAP-NODE 系）との接続: `burst_j(t)` は ①のアテンション Key 拡張
  （`DUAL_FORCE_REDESIGN.md` §3 P4）とまったく同じ量として共有する。②の Gate 0 が burst の
  実時間検出力を確認する作業そのものが、①の Key 設計を正当化する事前実験を兼ねる。

## 6. 評価プロトコル

- 分割: 訓練 < Y、テストは Y のみ（`--holdout-test-year` と同型、検証Bの既存プロトコルを踏襲）。
- 指標: 方向性は Spearman ρ / AUC（既存指標と接続）、量的予測は Poisson deviance・sMAPE。
  すべて**naive ベースライン比の skill score**で報告する（生の相関・誤差だけを主張しない）。
- 頑健性: ブートストラップ（企業・トピック単位）、複数ドメイン・複数時期での再現。
- 打ち切り監査: 評価対象年が公開ラグの影響を受けていないことを一文でチェックリスト化する
  （§3-2、`PAPER_WORKFLOW.md` の限界節に相当する項目を追加）。

## 7. 想定される着地点と論文フレーミング

| 結果 | フレーミング |
|---|---|
| Gate 0 不通過（computing のみの後知恵） | predictability map C2 の確定事例として記録。「なぜ後知恵バイアスが生じるか」の診断自体を Limits of Predictability ペーパーの1節に格上げできる（"burst is not detectable ex ante" という否定結果も証拠になる）。 |
| Gate 0 通過・Phase 1 が Gate S/L も通過 | 「技術トレンドの HOW-MUCH 予測は一般には無信号だが、実時間検出可能なバーストレジームでは例外的に予測可能」という predictability map への初めての正の例外。KDD ADS track、または科学計量系ジャーナル（応用寄りの実証結果として）。 |
| Gate 0 通過だが Phase 1 が Gate S/L で頭打ち | 「レジームは検出できるが、検出できても線形 momentum×burst 以上の予測増分は学習モデルにはない」——これも simple-beats-complex の一種として記録しつつ、GLM ベースの実用ツール（KG-ATLAS の Prophet 置き換え）として案Cに合流。 |

## 8. 直近のタスク（優先順）

1. `predictability_ceilings.py` の I_j(t+1) 定義が公開ラグを考慮しているか監査し、必要なら
   評価ウィンドウをデータ境界から2年以上離す。（未着手）
2. ~~リークフリーな burst_j(t) を定義し、computing 以外のドメイン・複数の5年ローリングウィンドウで
   Gate 0 を実行。~~ **完了（2026-08-06、`gate0_regime_detectability.py`）。**
   結果は `outputs/predictability_map/RESULTS.md` の「Gate 0」節を参照。
   事前登録ルール上は形式的PASS（pharmaのみ有意に正, p_cluster=5e-06）だが、computing自身の
   元の効果がこの統制付き回帰では再現せず（全ローリング窓で負）、6ドメイン中5つが非有意または
   逆符号。**「バーストレジームは一般に実時間検出可能」という主張はデータに支持されない。**
   Phase 1への全ドメイン投資は正当化されない——pharma限定でのみ検討の余地あり。
3. ~~Gate 0 の結果を `outputs/predictability_map/RESULTS.md` に追記する~~ 完了。
4. Phase 1はpharma限定にスコープダウンした上でのみ着手を検討（Poisson/NegBin GLM、momentum×burst
   ＋ユーザー提案のKDE密度特徴 `models.py` `KDEDensityField` の追加）。全ドメイン一般化は不可。
5. `DUAL_FORCE_REDESIGN.md` 側の burst 特徴実装と定義を同期する。（未着手）

## 9. 位置づけ

本仕様は「出願数予測モデルを作る」提案を却下するものではなく、(i) 検証Bがすでに答えを出している
6/7ドメインへの再投資を止め、(ii) 唯一未決着の問い（バーストレジームの実時間検出可能性）に
実験を集中させ、(iii) Gate 0 の推定手続きとPhase 1のモデル選定を、確立された外部文献
（Kleinberg 2002 / Hamilton型regime-switching / DeepAR / Fisher-Pry・Lotka-Volterra、
`docs/RELATED_WORK_SURVEY.md`）に接続する工程表である。Gate 0 は半日〜1日程度で実行可能であり、
この結果次第で②の投資規模（「予測可能性マップの1セルとして記録して終わり」か
「独立した応用予測ペーパー」か）がほぼ確定する。
