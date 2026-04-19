"""チェックポイント読み込み（shape 不一致のパラメータはスキップ）。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


def load_state_dict_skip_shape_mismatch(
    model: nn.Module,
    state_dict: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    現在のモデルと同じキーかつ同じ shape のテンソルだけを読み込む。
    企業数・ノード数が学習時とずれている場合でも、エンコーダ等は可能な限り復元する。

    Returns
    -------
    skipped : 読み飛ばしたキーと理由のログ行
    missing_after : strict=False 後も未代入の model キー（参考）
    """
    model_sd = model.state_dict()
    to_load: Dict[str, Any] = {}
    skipped: List[str] = []

    for k, v in state_dict.items():
        if k not in model_sd:
            skipped.append(f"{k}: チェックポイントにのみ存在")
            continue
        if not torch.is_tensor(v):
            continue
        if model_sd[k].shape != v.shape:
            skipped.append(
                f"{k}: shape 不一致 ckpt={tuple(v.shape)} 現在={tuple(model_sd[k].shape)}"
            )
            continue
        to_load[k] = v.to(device=model_sd[k].device, dtype=model_sd[k].dtype)

    incompat = model.load_state_dict(to_load, strict=False)
    if incompat.missing_keys:
        skipped.append(f"[要約] model 側で未代入のキー: {len(incompat.missing_keys)} 個")
    if incompat.unexpected_keys:
        skipped.append(f"[要約] ckpt にあり model に無いキー: {len(incompat.unexpected_keys)} 個")
    return skipped, list(incompat.missing_keys)
