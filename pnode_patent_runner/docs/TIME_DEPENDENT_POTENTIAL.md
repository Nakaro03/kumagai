# 時間依存ポテンシャル Φ(z, year)

既存の `PotentialNet`（Φ(z) のみ）は [ARCHITECTURE.md](ARCHITECTURE.md) を参照。このノートは **Φ(z, year)** を別モジュールで学習・可視化する経路の説明である。

## 定義

- **埋め込み**: カレンダー年 `y ∈ [year_min, year_max]` を整数インデックスにし、`nn.Embedding` でベクトル化して `sin(zB), cos(zB)` と連結し、MLP でスカラー Φ を出力（[`time_dependent_potential.py`](../time_dependent_potential.py)）。
- **ODE**: 1 年先予測は **出発年 `y0` 固定**の Φ(z, y0) に沿った勾配流（`GradientNeuralODEPredictorTime`）。
- **デコーダ（CoPE 型）**: 再構成は **年 `y0`**、未来リンクは **年 `y1`** の Φ を使用（[`unified_vgae_td.py`](../unified_vgae_td.py)）。

## 学習

```bash
python -m pnode_patent_runner.run_train_unified_vgae_td \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2010 2020 --epochs 30 \
  --save pnode_patent_runner/outputs/cope_landscape/unified_vgae_td.pt
```

checkpoint は `state_dict` に加え `year_min`, `year_max` 等を含む辞書で保存される。

## 可視化

各スライダー年 `y` について **Φ(z, y)** を格子上に評価し、[`interactive_landscape_vector_field_td.py`](../interactive_landscape_vector_field_td.py) 経由で HTML に埋め込む。

```bash
python -m pnode_patent_runner.run_interactive_landscape_td_vector_field \
  --data notebooks/work/dataset/topic_info3.csv \
  --year-range 2015 2018 \
  --load-checkpoint pnode_patent_runner/outputs/cope_landscape/unified_vgae_td.pt \
  --output pnode_patent_runner/outputs/cope_landscape/map_cope_alt_dark_td.html
```

## 既存 CoPE との関係

- `UnifiedVGAE` / `run_train_unified_vgae_checkpoint.py` は変更しない。
- 時間依存版は **別クラス `UnifiedVGAETD`** と **別 CLI** で扱う。
