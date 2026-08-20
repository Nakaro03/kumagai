# 研究計画: The Predictability Map of Technology Trends

**ワンライナー**: 技術トレンド予測を「タスク族 × ドメイン × 時間解像度」の行列に分解し、
どこに予測信号が実在しどこで消えるかを、強い自明ベースラインと統一プロトコルで体系的に確定する。
ニューラル時系列グラフモデル（Neural ODE / ポテンシャル流 / RNN）が付加価値を持つ条件を特定し、
持たない場所では「なぜ持てないか」の診断（離散的・relatedness ゲート付き参入という生成機構）を与える。

**ターゲット**: KDD / WWW（Web & Society or Graph 系トラック）。副案: ICWSM, EPJ Data Science。
ベンチマーク+診断型（"On the Predictability of ..." 系譜、例: 人流予測の Song et al./Lu et al. 系）。

---

## 1. なぜこれが採択水準になり得るか

単発の否定結果は弱い。しかし本プロジェクトには **2 年分・9 回以上の「単純ベースラインが複雑モデルと同着以上」**
という系統的証拠が独立実験として蓄積されており、これを一つの行列に整理すると:

1. **体系性**: タスク族（WHERE / HOW-MUCH / WHEN / WHAT-NEW）× 7 ドメイン × 粒度 × プロトコル
   （transductive vs holdout）の完全な地図は存在しない。
2. **強いベースライン**: 人気度・持続性・relatedness・momentum の「天井」を訓練不要で確定してから
   学習モデルの増分を測る、という設計自体が方法論的貢献。
3. **診断**: 「連続フロー抽象が誤り。動きは離散的・relatedness ゲート付き参入」（velocity attribution,
   6 ドメインで力場 R²≈0.01-0.05）という機構的説明が、否定結果群を一つの原因に還元する。
4. **実務接続**: 予測が成立しない場所で何を提供すべきか（conformal 区間・較正・記述的地形）への指針。
   KG-ATLAS（実運用ツール）の Prophet 予測・white-space がまさに「死んだターゲット」であることの告発と修正。

## 2. 主張（Claims）と対応する証拠

| # | 主張 | 証拠の状態 |
|---|------|-----------|
| C1 | WHERE（誰がどの技術に参入するか）には信号があるが、relatedness/人気度で飽和し、学習モデルの増分はない | Task B 天井 MRR 0.213 / PI-SDE Poisson Phase B 失敗 / TAP-NODE 10 シード（本セッション） |
| C2 | HOW-MUCH（成長率・流入量の変化）は変化対変化ではほぼ無信号 | X5 LOO≈0 / DRIFT 失敗 / TAP change-on-change ρ=−0.18 n.s. → **検証 B で全ドメイン化** |
| C3 | WHEN（タイミング・早期警戒）はレベル/momentum 以外に信号なし | early-warning 実験（var/AC1 at chance） |
| C4 | レベル→レベルの「人気度天井」は極めて高く（ρ≈0.8+）、学習モデルの見かけの精度はほぼこれ | TAP pull vs mass: 偏相関 0.05 → **検証 B で全ドメイン化** |
| C5 | 上記は単一ドメインの偶然ではない（ドメイン横断で再現） | → **検証 A（クロスドメイン GNN ベンチ）+ 検証 B** |
| C6 | transductive 評価は系統的に精度を過大評価する | author_topic で −0.09〜−0.11 → **検証 A で全ドメイン化** |
| C7 | 機構診断: 参入は離散・relatedness ゲート付きで、連続潜在ドリフトでは表現できない | velocity attribution 済（6 ドメイン） |
| C8 | 予測不能な場所でも conformal 被覆は移転する（honest tool の設計指針） | non-eq Langevin 実データ CovConf 0.96（追加検証は次段） |

## 3. 検証行列（本セッションで着手したもの）

### 検証 A: クロスドメイン学習モデルベンチ（実行中）
- ドメイン: agrifood / construction / energy（+著者–トピック済み）。大規模 3 ドメインは後続。
- プロトコル: year-range 2017–2021, **holdout-test-year 2021**, 3 seeds, epochs 10,
  methods = static / RNN+VGAE / NeuralODE / P-NODE(K=4,GRU)。
- 問い: 時間ダイナミクスの学習は static を**どのドメインでも**上回るか？（C5, C6）
- 出力: `outputs/predictability_map/crossdomain/{dom}_holdout_seed{S}.json`

### 検証 B: 訓練不要の予測可能性天井（実行中）
- 全 6 特許ドメイン × CPC 粒度（maingroup / subclass）+ arXiv 著者–トピック。
- 各年遷移で: popularity ρ(M_t, I_{t+1}) / persistence ρ(I_t, I_{t+1}) /
  trend→level / **人気度統制の偏相関** / **change-on-change**（ハードテスト）。
- 問い: 人気度天井の高さと、トレンドの増分情報ゼロはドメイン普遍か？（C2, C4）
- 実装: `predictability_ceilings.py` → `outputs/predictability_map/ceilings.json`

### 検証 C: 同一評価ペア上の訓練不要 AUC（完了）
- `predictability_popularity_auc.py`。結果は `outputs/predictability_map/RESULTS.md`。
- **主要発見**: author_topic は人気度のみで AUC 0.944（全学習モデルはその下）。energy も天井以下。
  agrifood/construction のみ構造学習が +0.11〜+0.16 の実質価値 → **予測可能性は粒度の関数**（旧検証 F が先行確定）。

### 検証 A+B+C の統合結論（2026-07-06 時点）
- 時間ダイナミクス学習が static に holdout で勝つドメイン: **0 / 4**。
- 人気度天井 ρ(M_t, I_{t+1}) = 0.91–0.98（7 ドメイン普遍）。人気度統制後のトレンド増分 ≈ 0。
- change-on-change は computing のみ正（+0.34、5 遷移すべて）— ブーム・レジーム例外。
- ドメイン 3 レジーム分類: 構造学習有効（agrifood/construction）/ 人気度飽和（energy）/
  人気度支配（author_topic 粗粒度）。

### 次段（未着手、優先順）
1. **検証 D**: conformal 区間の被覆がドメイン横断で移転するか（C8 の一般化）
2. **検証 E**: content（LLM 埋め込み）の増分 — 予測可能性マップの未踏フロンティア
3. 大規模 3 ドメイン（semiconductor / pharma / computing）への検証 A 拡張（OOM 対策済みスクリプトあり）
4. 粒度スイープの体系化（subgroup / maingroup / subclass を同一ドメイン内で比較）
5. シード数 10 への拡張と全ペア検定（現状 3 シードは方向性確認レベル）

## 4. 論文構成案

1. Introduction — 技術予測ツールの氾濫と、その中核タスクの検証されなさ
2. The Predictability Map — タスク族の形式化（WHERE / HOW-MUCH / WHEN / WHAT-NEW）
3. Training-free ceilings — 検証 B（Table 1: ドメイン × 相関族）
4. Do learned dynamics add value? — 検証 A + C（holdout, 多シード, 偏相関）
5. Why: the discrete gated-entry mechanism — velocity attribution の再掲+統合
6. What to build instead — conformal / 較正 / 記述的地形（TAP-NODE の同定性・可視化を「解釈可能な記述」として位置づけ）
7. Implications for deployed tools — KG-ATLAS 事例

## 5. 誠実性の原則（このプロジェクトの規律）

- すべての精度主張は **holdout + ≥3 シード + 対応あり検定**（author_topic の教訓: 単一シードの +0.22 は seed 運）。
- 学習モデルの増分は必ず**天井との偏相関/差分**で報告（見かけの ρ ではなく）。
- 否定結果は「弱かった」ではなく「何と同着だったか」まで特定する。
