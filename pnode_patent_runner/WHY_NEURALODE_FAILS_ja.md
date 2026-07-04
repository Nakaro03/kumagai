# Neural ODE による技術トレンド予測がうまくいかない理由（教授説明用ドラフト）

作成: 2026-06-11 / 対象: 指導教員ミーティング / ステータス: テスト草稿

---

## 0. 一言でいうと

**「Neural ODE（および時間依存ポテンシャル・SDE）が悪いのではなく、"企業×技術分野（firm×CPC）の2部グラフで、企業が次にどの技術分野へ進出するかを構造データから予測する" というタスクが、近接（relatedness）の範囲内でしか予測できない」**ことが、複数アーキテクチャ・複数定式化の横断実験で繰り返し確認された。Neural ODE は表現力では負けていないが、**学習可能なもの（既存技術に近い隣接分野への進出）は単純な近接指標で届き、単純指標で届かないもの（既存から離れた真の新規進出 = jump）は原理的に予測不能**という二択に挟まれている。これが本質的な失敗理由である。

---

## 1. 何を予測しようとしていたか

- 設定: **企業×CPC（技術分類）の2部グラフ**。ノードは企業と技術分野、エッジは「その企業がその分野で特許を出した」。
- 予測タスク: **ある企業が将来どの技術分野（CPC）に進出するか**（= 2部グラフのリンク予測 / inventor-technology expansion recommendation）。
- 補助情報: 特許/CPC の共起・引用の時系列、content（LLM テキスト埋め込み）、inventor expertise。
- 手法系列:
  - `PNodeEnergyTD` 系（時間依存ポテンシャル φ(z,t) を学習する Neural ODE）
  - `PISDE`（連続時間ポテンシャル SDE）
  - `DRIFT`（recurrent-depth transformer。X5 の後継として設計）
  - Dual-Force P-NODE（ポテンシャル力＋追加力の分解）
  - VGAE / fusion 系の recommender prototype

狙いは「ポテンシャル地形 φ の谷＝企業が落ち着く技術ポジション、流れ＝進出方向」を学習し、**地形から企業の次の技術進出を予測する**という物理アナロジーだった。

---

## 2. なぜうまくいかないか（実証的証拠）

以下はすべて本プロジェクトの実験ログ／メモリに残っている結果。**個別の不運ではなく、独立な5つ以上の経路で同じ結論に収束**している点が重要。

### 証拠1 — 進出予測は近接（relatedness）の範囲でしか当たらない
- 企業×CPC の進出（リンク）予測で、**learned embedding は Adamic–Adar など近接ベース指標を hard negative 上で上回れない**。
- 「予測可能 = ほぼ近接の言い換え」。すなわち**企業が既存技術に隣接した分野へ広がる動きは当たるが、それは2部グラフの近接構造で説明がつく自明な部分**。Neural ODE 固有の上乗せがほぼない。

### 証拠2 — 単純ベースラインに勝てない（simple-beats-complex を複数回観測）
複雑な学習地形が**学習不要の単純指標に並ばれる／負ける**現象が繰り返し起きた:
1. **recommender prototype**: precision@k で **relatedness-only が fusion に勝つ**。fusion は top-k を人気度（popularity）で汚染する。複雑なモデルを足すほど実用指標が悪化。
2. **bipartite landscape**: 成長対応 Φ は holdout に**汎化しない**（HOLDOUT B +0.025 ≈ A +0.033）。φ-valley ≠ growth で、成長は地図上に散在。
3. **X3-clean**: φ の流れは、同じデータを与えた**学習不要の密度変化ベースライン**に並ぶ（"成長色に勝つ" は撤回）。
4. **novelty hazard**: brokerage の AUC 0.69 は実質**約2-hop の近接**で、固有寄与は +0.012 のみ。Neural ODE は静的指標を超えない。
5. **early-warning**: 月次の **critical slowing down（分散・自己相関トレンド）は当てずっぽう**。WHEN を当てるのは momentum / level だけ。

### 証拠3 — 特徴量を足しても帰納の壁を越えない
- content（LLM テキスト埋め込み）を入れると sparse 構築で fusion を **+0.031**（プロジェクト初の正の寄与）。
- さらに inventor expertise を足すと**学習リフトはプロジェクト最大の +0.052** まで伸びた。
- **しかし pure-inductive 設定（未知の企業・未知の組合せ）では content+human でも依然 chance（0.504）。content 単体では 0.413。**
- → 上乗せは「既知の近接を補強」しているだけで、**新規進出そのものを生み出す予測力ではない**。

### 証拠4 — 本当に価値のある "jump"（企業の非自明な新分野進出）は原理的に予測不能
- 関連性（relatedness / proximity）の範囲内は予測可能だが、**企業が既存ポジションから離れた分野へ飛ぶ動き（真の新規進出）は当てられない**。
- これは予測対象が「どこへ」だけでなく「いつ」でも同様（proximity-bound in time）。
- 結論: **企業の技術進出予測は relatedness-bound。自明な隣接拡大は近接指標で十分、非自明な飛びは全手法 chance。**

### まとめ図（因果）
```
企業の技術分野進出予測 (firm×CPC link prediction)
   ├─ 近接で説明できる進出(隣接分野への拡大) → Adamic-Adar/relatedness で十分 → Neural ODE 不要
   └─ 近接で説明できない進出(非自明なjump)   → content+human でも chance(0.504) → 原理的に予測不能
                                   ↑
                  Neural ODE はこの "谷間" に挟まれて勝てない
```

### 証拠5 — 進出 base-rate の実測（仮説Aを棄却、仮説Bを確定）
教授の問い「タスク自体に問題があるのでは？ 企業は技術領域をあまり変えないのでは？」に答えるため、
construction ドメイン（firm×CPC, 12,322社, 99,916件の新規進出）で base-rate を直接測定した。
スクリプト: `diagnose_entry_baserate.py`、出力: `data/processed/entry_baserate_construction.json`。

**指標④の relatedness は時系列リークなし版（進出年 t 未満の共起だけでグラフを構築）を正式値とする。**
当初の「全期間共起」版は、進出という行為自体が同年に c–既存CPC のエッジを作るため値を水増ししていた
（92.1%）。これを自分で発見・除去したのが下表の頑健性チェックで、**結論はむしろ強化された**。

| 指標 | 値 | 意味 |
|---|---|---|
| ① ポートフォリオ持続性 (Jaccard) | 0.29 (≈28.8%) | 年次で7割入れ替わる（活発） |
| ② CPC残存率 (stickiness) | 38.3 % | 前年分野の38%しか翌年残らない → **企業はよく動く** |
| ③ 新規進出数 / 社・年 | 2.50 件 | **毎年新分野に出ている**（停滞ではない） |
| ④ 進出が既存に近接(1-hop)な割合 ※リークなし | **77.8 %** | 動きの約8割は既存近接 |
| ④' 非自明なjump ※リークなし | **22.2 %** | 価値ある飛びは約2割 |
| ④'' ランダムnullの近接率 | 21.3 % | **+56.5pt のリフト** → 近接は密度の人工物でなく本物の強い信号 |
| ⑤ 「変化なし」予測の天井 | 23.4 % | 低い → **停滞仮説(A)を棄却** |

参考（リークあり・全期間共起版）: 近接 92.1% / jump 7.9% / null 33.1% / リフト +59.0pt。
**リークを除いても近接リフトはほぼ不変（+59.0→+56.5pt）** = relatedness 信号はリークに依存しない頑健な事実。

**読み:** 企業は活発に動く（②③⑤）が、その動きの**約8割は relatedness で説明でき、ランダムより+56.5pt**。
すなわち **「タスクが悪い／企業が動かない」のではなく、"動きの予測可能な部分が relatedness という単純構造に尽きている"**。
残る22%の非自明な jump こそ価値があるが、証拠3・4の通り content+human を入れても pure-inductive は chance。
→ **仮説B（relatedness 支配）を数値で確定。** これが Neural ODE が勝てない構造的理由の決定的裏付け。

> 方法注: relatedness は「同じ企業が同じ年に共に出願した CPC ペア」を共起エッジとする標準的近接指標。
> リークなし版は進出年 t 未満の共起のみでグラフを張り、進出による自己生成エッジを排除している。
> `diagnose_entry_baserate.py --leak-free` で再現可能。

---

## 3. 根本原因の理論的説明（教授の想定質問への備え）

| 想定質問 | 回答の要点 |
|---|---|
| 「モデルの容量/学習が足りないのでは？」 | 容量を上げた DRIFT でも**過学習**して失敗。容量問題ではなく**信号不在**。pure-inductive が chance(0.504) なのが決定的。 |
| 「特徴量が貧弱では？ テキストを入れれば？」 | content+human を投入し**学習リフト +0.052（最大）**を得たが pure-inductive は 0.504。**特徴を足しても帰納の壁は越えない。** |
| 「relatedness-only に負けるのはなぜ？」 | fusion は top-k を **popularity で汚染**する。近接の強い信号に弱い信号を混ぜると実用指標(precision@k)が下がる。**単純さがロバスト。** |
| 「ポテンシャル地形は何の役に立つ？」 | **予測ではなく記述（descriptive / VIS）**として有効。企業ポジションと技術空間の可視化・説明の道具。φ-valley ≠ growth、holdout 非汎化は確認済み。 |
| 「Neural ODE 固有の強みは？」 | 連続時間・物理アナロジーは美しいが、**本タスクの信号構造（近接支配）では静的グラフ指標と等価**。タスクが ODE の強みを要求していない。 |
| 「なぜ近接が支配的だと言える？」 | 勝てる予測はすべて 2-hop 近接に還元でき、近接から外れた進出は全手法 chance。**独立5経路で同じ天井**。 |

**核心メッセージ:** これは "実装の失敗" ではなく **"予測可能性の地図 (predictability map)" の発見**。
- 隣接分野への進出 = 予測可能だが近接指標で自明
- 非自明な新分野進出（jump）= 原理的に予測不能（relatedness-bound）
- 較正（calibration）も未解決（recommender prototype の ECE 0.165）

---

## 4. では何をトップ会議に通すか（代替案）

ネガティブ結果そのものを資産化する案を上位に置く。各案にターゲット会議を付す。

### 案A（推奨）— "Limits of Predictability" 論文
- **主張**: 技術トレンド成長予測は relatedness-bound であり、構造データから novelty は予測できない。これを**複数アーキテクチャ・複数定式化の横断実験で厳密に示す**。
- **強み**: 本プロジェクトの5経路の証拠がそのまま方法論になる。再現性・反証可能性が高い。査読で刺さる「驚き」がある（みんな予測できると思っている）。
- **必須要素**: 強い単純ベースライン（Adamic-Adar / density / momentum）との厳密比較、pure-inductive プロトコル、ablation。
- **ターゲット**: KDD / WWW（Applied Data Science or Research track）, EPJ Data Science, *Nature Communications*/*PNAS Nexus*（science-of-science 系）。

### 案B — 記述・可視化（VIS）として地形を売る
- **主張**: 予測ではなく、技術空間の**ポテンシャル地形による説明・探索ツール**。成長は予測値ではなく overlay として提示。
- **強み**: φ 地形パイプラインは既に動作（firm×CPC, EDGPAT viz は Moran's I ~0.15–0.23 で legibility 合格）。
- **ターゲット**: IEEE VIS, EuroVis, CHI（探索的分析）。
- **注意**: 「予測できる」とは絶対に主張しない。growth は overlay と明記（メモリの教訓）。

### 案C — 実務ツールの "honest-ify"（研究×実務の橋渡し / 最有望の実用貢献）
- **背景**: 既に 熊谷組向けに KG-ATLAS（特許＋論文＋ニュース、5ペルソナ）を deploy 済み。その Prophet 予測・white-space は**本研究で死亡確認済みのターゲット**。
- **主張**: 「予測の限界」をプロダクトに組み込み、**過信を較正（calibration）**する。具体的には:
  - **hype-vs-substance**: ニュース由来の誇張と特許/論文の実体の乖離をクロスソースで定量化（まずここから）。
  - **lead-lag**: ソース間の先行/遅行関係。
  - **calibration**: 予測信頼度の isotonic 較正（recommender prototype の ECE 0.165 を改善）。
- **ターゲット**: KDD ADS track, CIKM, *WWW* (Industry), あるいは科学計量系ジャーナル。
- **強み**: 実データ・実ユーザー・「正直さ」という新規フレーミングの三拍子。

### 案D — まだ枯れていないフロンティアを攻める（ハイリスク）
- predictability map で**未検証**の領域: より細かい粒度、引用/emergence の forecasting、テキスト特徴の別の使い方。
- ただし human-aware 診断で pure-inductive は chance のままだったので、**期待値は低い**。やるなら早期に go/no-go 診断を設計。

### 推奨ルート
**案A（限界の論文）を主軸**にし、**案Cを実用貢献として併走**、**案Bを figure 資産として流用**。この3点セットが、ネガティブ結果を最小リスクでトップ会議の貢献に変換する最短経路。

---

## 5. 教授ミーティングでの提示順（1枚スライド想定）

1. 当初の狙い（企業×CPC 2部グラフから、企業の次の技術進出を地形で予測）
2. 結果: relatedness-only が fusion に勝つ／pure-inductive は chance(0.504)／5回の simple-beats-complex
3. 解釈: predictability map（隣接進出=自明 / 非自明なjump=予測不能 / 較正も未解決）
4. 転換: 「失敗」ではなく「限界の発見」を貢献にする（案A）
5. 並行: 実務ツールの honest-ify（案C）と可視化資産（案B）
6. 依頼: 主軸を案Aで進めてよいか、ターゲット会議の合意

---

### 付記（出典メモ）
本資料の数値・結論は本プロジェクトの実験ログ（X5 verdict, DRIFT go/no-go, convergence signal, novelty hazard, early-warning timing, bipartite landscape, content/human diagnostic, KG-ATLAS bridge）に基づく。具体スクリプトは `aggregate_x5_*.py`, `analyze_h1_results.py`, `baseline_*.py` 等を参照。
