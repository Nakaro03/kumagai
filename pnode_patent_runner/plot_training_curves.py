"""
学習履歴（`train_model_improved` / `train_model_td` の `history`）から曲線を PNG 保存する。

時間依存ポテンシャル関連は `train_components["potential"]` と `["trajectory"]`（Φ の正則化と勾配整合項）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import japanize_matplotlib  # noqa: F401 — 日本語ラベル用フォント
except ImportError:
    pass


def _epochs(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=float)


def plot_training_history(
    history: Dict[str, Any],
    title: str = "",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    `history` には少なくとも `loss`, `val_auc` が必要。
    `train_components` があれば potential / trajectory 等を重ね描画する。
    """
    loss = history.get("loss") or []
    val_auc = history.get("val_auc") or []
    tc = history.get("train_components")
    tc_len = 0
    if isinstance(tc, dict) and tc:
        first = next(iter(tc.values()), None)
        if isinstance(first, list):
            tc_len = len(first)
    n = max(len(loss), len(val_auc), tc_len)
    if n == 0:
        raise ValueError("history に loss / val_auc / train_components が無いか空です")
    ep = _epochs(n)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax0, ax1, ax2, ax3 = axes.flat

    if loss:
        ax0.plot(_epochs(len(loss)), loss, label="train total", color="C0")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.set_title("訓練損失（合計）")
    ax0.grid(True, alpha=0.3)

    if val_auc:
        ax1.plot(_epochs(len(val_auc)), val_auc, label="val ROC-AUC", color="C1")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("検証（最終年ペア future-link）")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.02)

    if isinstance(tc, dict):
        pot = tc.get("potential")
        traj = tc.get("trajectory")
        gpn = tc.get("grad_phi_l2")
        if pot:
            ax2.plot(_epochs(len(pot)), pot, label="potential (raw term)", color="C2")
        if traj:
            ax2.plot(_epochs(len(traj)), traj, label="trajectory", color="C3", linestyle="--")
        if gpn:
            ax2.plot(
                _epochs(len(gpn)), gpn, label="grad_phi L2 (mean ||∇Φ||)", color="C7", linewidth=1.0
            )
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("mean batch value")
        ax2.set_title("Φ 関連（potential / trajectory / ||∇Φ||）")
        ax2.legend(loc="best", fontsize=8)
        ax2.grid(True, alpha=0.3)

        for name, c in (
            ("recon", "C4"),
            ("future_link", "C5"),
            ("latent_pred", "C6"),
        ):
            s = tc.get(name)
            if s:
                ax3.plot(_epochs(len(s)), s, label=name, color=c)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("mean batch value")
        ax3.set_title("その他損失項（生値）")
        ax3.legend(loc="best", fontsize=8)
        ax3.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "train_components なし", ha="center", va="center", transform=ax2.transAxes)
        ax3.text(0.5, 0.5, "train_components なし", ha="center", va="center", transform=ax3.transAxes)

    if title:
        fig.suptitle(title)

    return fig


def save_training_history_png(
    history: Dict[str, Any],
    out_path: Path,
    title: str = "",
    dpi: int = 120,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_training_history(history, title=title)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def load_history_from_benchmark_json(
    path: Path,
    method_key: Optional[str] = None,
) -> Dict[str, Any]:
    """`run_benchmark_comparison` の JSON から 1 手法分の history 相当を復元する。"""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("JSON に results 配列がありません")
    if method_key is None:
        if len(results) != 1:
            raise ValueError("method_key を指定するか、results が 1 件の JSON を使ってください")
        row = results[0]
    else:
        row = next((r for r in results if r.get("key") == method_key), None)
        if row is None:
            raise ValueError(f"key={method_key!r} が見つかりません")
    loss = row.get("train_loss_per_epoch")
    val_auc = row.get("val_auc_per_epoch")
    tc = row.get("train_components_per_epoch")
    h: Dict[str, Any] = {}
    if isinstance(loss, list):
        h["loss"] = [float(x) for x in loss]
    if isinstance(val_auc, list):
        h["val_auc"] = [float(x) if x is not None else float("nan") for x in val_auc]
    if isinstance(tc, dict):
        h["train_components"] = {
            k: [float(x) for x in v] for k, v in tc.items() if isinstance(v, list)
        }
    return h


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="ベンチマーク JSON または手動で渡した履歴から学習曲線 PNG を出力"
    )
    p.add_argument("input", type=str, help="benchmark_*.json のパス")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="出力 PNG（省略時は input と同じ stem に _curves.png）",
    )
    p.add_argument("--key", type=str, default="", help="手法キー（例: pnode_energy）")
    p.add_argument("--title", type=str, default="")
    args = p.parse_args(argv)

    inp = Path(args.input)
    if not inp.is_file():
        raise SystemExit(f"ファイルがありません: {inp}")
    key = args.key.strip() or None
    hist = load_history_from_benchmark_json(inp, method_key=key)
    out = Path(args.output) if args.output.strip() else inp.with_name(inp.stem + "_curves.png")
    save_training_history_png(hist, out, title=args.title)
    print(f"Wrote: {out.resolve()}")


if __name__ == "__main__":
    main()
