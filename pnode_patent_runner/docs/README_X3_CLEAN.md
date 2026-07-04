# X3-clean — 論文提出前まとめ (Descriptive Landscape)

本書は X3-clean を論文として提出する前の段階で、「何を作ったのか」「何を主張し、何を主張しないのか」「研究上の問い (RQ) は何か」「証拠と図はどう対応するのか」「投稿前に何が残っているのか」を文章でまとめた README である。詳細な根拠は [X3_DESCRIPTIVE_FRAMING.md](X3_DESCRIPTIVE_FRAMING.md) と [PISDE_ROLE_DECISION.md](PISDE_ROLE_DECISION.md) に置き、本書はそれらを束ねた入口として機能する。

## 1. X3-clean とは何か

X3-clean は、論文・特許とそのトピックからなる二部グラフの上に、時間依存のエネルギー関数 Φ(z, t) を学習する手法である。Φ は「負の対数密度 (−log p)」としての物理的意味を持ち、低い Φ の領域は研究が高密度に集積した谷、高い Φ の領域は研究が手薄な尾根として解釈される。学習は Sinkhorn による周辺分布のマッチングと、Φ-anchor 制約 (各トピック重心での Φ を成長率と対応づける項) の組み合わせで行う。

「clean」という名称は、前身である X3-baseline が成長率 g_n をモデル入力に含めてしまい、評価時に identity copy のリーク経路を生んでいた問題を取り除いた版であることを示す (実装上は MODE=mask)。このリーク除去によって、X3-clean は「入力に答えが混ざっていない、正直な記述モデル」になっている。

学習・評価は 4 ドメイン (論文 PNode_Paper、特許 Energy、arXiv Construction、JP Construction) × 5 seed、加えて paper の leave-one-out × 5 seed の計 25 ラン分が [RESULTS_X3_ABLATION/](../../RESULTS_X3_ABLATION/) に揃っており、集約は [aggregate_x3_clean_validity.py](../aggregate_x3_clean_validity.py) が担う。

## 2. 当初の主張からの転換 (重要)

X3 は当初「Φ = −log p というエネルギー基底の定式化により、単一ハイパラで多ドメイン横断の未来トピック成長を予測できる」と主張していた。しかし leave-one-out 検証 (paper t=3 を holdout して 5 seed) で、held-out 時点に対する成長率の Spearman ρ が −0.35 ± 0.08 と負相関に転落した。これは、学習時と評価時で同じ時点を使う alltime 評価での好成績 (ρ = 0.69〜0.98) が memorization の産物であったことを意味する。

この結果を受けて、X3-clean は予測モデルとしての主張を全面的に撤回し、「観測済みデータに対する解釈可能な回顧的 (retrospective) ランドスケープを構成するツール」として再定義された。予測の野心は別系統 (DRIFT 提案) に移管されている。したがって本論文で予測能力を主張することは、自分たちの実証結果と矛盾するため厳禁である。

## 3. 研究上の問い (RQ)

X3-clean が記述的可視化ツールである以上、RQ は予測精度ではなく「地形が観測構造を忠実に表現できるか」「整合的か」「汎用か」「人がそこから読み取れるか」を問う形になる。

- **RQ1 (忠実性):** 学習されたエネルギー地形 Φ は、観測済みトピックの密度とランキング構造をどれだけ正確に捉えるか。→ alltime の Φ-rank Spearman ρ と NDCG@10 で測る。現状 ρ ≈ 0.6 で 3/4 ドメイン有意。
- **RQ2 (整合性):** 学習目的に組み込んだ growth-anchor 制約は実際に成立し、地形が成長率と物理的に対応するか。→ Φ(c_j, t) と −g̃_j(t) の Pearson r で測る。集約 r = 0.80、paper t=3 で r = 0.94 と成立済み。
- **RQ3 (汎用性):** 同一手法・単一ハイパラ (λ_growth) のまま、論文・特許・arXiv・JP の複数ドメインに適用できるか。→ 4 ドメイン × 5 seed、seed 間 std ≤ 0.09。3/4 ドメインで有効、arXiv のみ弱い。
- **RQ4 (解釈支援):** この地形可視化は、専門家が研究トレンドや異常領域を読み取る作業を、素朴な記述ベースライン (PCA + 成長率カラー) より助けるか。→ user study (10 人 × 2 タスク) と baseline 比較で測る。**現状未実施で、最大のギャップ。**

メインの問いは「二部グラフ上の学習エネルギー地形 Φ は、観測済み研究領域の構造を解釈可能かつドメイン横断で忠実に表現できるか」であり、RQ1〜RQ3 で記述的・整合的・汎用的であることを示し、RQ4 で可視化ツールとしての実用価値を示す構成になる。なお「未来予測できるか」「Φ = −log p を経験的に証明できるか」は RQ に立てない (前者は否定済み、後者は Panel E の Pearson r = 0.24 と弱く limitation 扱い)。

## 4. 主張のスコープ

言ってよいこと: 観測済みデータに対する記述的ランキングが高いこと、古典・現代の時系列ベースライン (alltime 評価) より強いこと、multi-seed で安定なこと、multi-domain で汎用なこと、単一ハイパラで動くこと、解釈可能な可視化を 3 種類提供すること、growth-anchor 制約が成立すること。

言ってはいけないこと: 未来時点のトピック成長を予測できる、Φ = −log p を経験的に証明した、unseen な時点への外挿に強い、リークありの X3-baseline が予測能力を持つ、実時間で次年度のホットトピックを推薦できる — これらはいずれも leave-one-out 破綻もしくは弱い相関で反証されている。

グレーゾーン: ベクトル場 −∇Φ を score function と呼ぶこと (理論的には正しいが PCA-2D では分散の 4% しか可視化できず近似的)、回顧的に成長要因を発見すること (case study レベルでは可能だが定量主張には user study が要る)、他の PI-SDE 変種より優れること (Φ-rank は X1 が強いので「X1 と同等かつ interpretable」と限定的に述べる)。

## 5. 図とその役割

X3-clean は 4 ドメイン × seed=42 × 4 種類 = 16 ファイルの可視化を出力する。中核は 6 パネルの静止画 (`landscape_x3clean_t{T}.png`) で、上段は解釈、下段は理論検証に対応する。パネル A は観測点を年で色分けして「何を学習対象にしたか」を示し、パネル B は Φ ヒートマップで研究集積の谷と空白の尾根を一目で示す本命図、パネル C はトピック重心の配置に実成長率 g を重ねる。下段のパネル D は Φ ランクと実 g ランクの相関で RQ1 の忠実性を裏づけ、パネル E は Φ_θ と経験的 −log p̂ の関係で EBM 主張を検証 (ただし弱い)、パネル F は Φ_θ(c_j, t) と −g̃_j(t) の関係で RQ2 の anchor 整合性を示す。

これに加えて、時点ごとと集約の anchor 整合性を示す `anchor_x3clean_all_t.png`、t スライダで地形の時間変化を動かせる `landscape_x3clean_interactive.html`、注目トピックの軌跡を追える `case_study_x3clean.png` がある。図全体の価値は、研究戦略家や特許アナリストがトピックの密度地形 (Φ) と成長方向 (−∇Φ) を一枚で読み取れるという解釈支援にあり、予測精度ではなく可視化としての trend visibility が貢献の核になる。

生成スクリプトは [plot_pisde_x3_landscape.py](../plot_pisde_x3_landscape.py) (6 パネル)、[plot_pisde_x3_anchor.py](../plot_pisde_x3_anchor.py) (anchor)、[run_interactive_landscape_pisde_x3.py](../run_interactive_landscape_pisde_x3.py) (HTML)、[plot_pisde_x3_case_study.py](../plot_pisde_x3_case_study.py) (case study)、学習は [run_pisde_x3_ablation.py](../run_pisde_x3_ablation.py) (MODE=mask) である。

## 6. ターゲット venue

本命は IEEE VIS / EuroVis / TVCG で、可視化ツールとしての貢献を主張するには task analysis、design rationale、user study が要る。Scientometrics / Journal of Informetrics は研究領域分析として既に完成度が高く、RQ1+RQ2 中心なら user study 無しでも投稿可能。KDD / WWW の industry track は大規模実データと business 解釈を加えれば適合する。NeurIPS / ICML / ICLR は予測主張がなく理論貢献として弱いため撤退する。

## 7. 投稿前に残っている作業

必須は、想定ユーザ (research strategist, patent analyst) の task をリストアップする task analysis、なぜ Φ-anchor や PCA/UMAP を可視化に使うかの design rationale、そして trend identification と anomaly spotting の 2 タスクを 10 人で行う user study プロトコルの設計と実施 (RQ4 の解決)。推奨は、PCA + 成長率カラーという記述ベースラインとの trend visibility 比較、および X1・X2 でも同条件の landscape を生成して「Φ-rank は X1 が強いが可視化解釈は X3-clean が容易」という差別化を示すこと。

## 8. Limitations (論文冒頭で自発的に明記する)

本手法は未来予測ではなく、全 metric は観測データに対する fit であり leave-one-out で破綻することは検証済みである。学習データに対する高い ρ は memorization を含むが、descriptive claim の範囲では問題にならないため誤読を避けるために明記する。PCA-2D は分散の 4〜15% しか説明できず score field の矢印は近似である。30 トピック程度の小スケールで動作確認しており 1000+ トピックでの挙動は未検証である。arXiv Construction では Φ-rank が弱く (1/5 seed 有意)、一般化には追加実験が要る。これらを冒頭から明示することで、honest な descriptive contribution として査読者の信頼を得る位置を取る。
