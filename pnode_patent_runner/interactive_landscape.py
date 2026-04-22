"""
学習済み潜在ベクトルと CSV から、Plotly 用 JSON ペイロードを組み立てる。

- ``build_interactive_payload``: 企業–特許
- ``build_interactive_payload_author_paper``: 著者–論文（キー構造は前者と同型でテンプレ共用）
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from torch_geometric.data import Data


def _clip(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


def _cell_str(row: pd.Series, key: str) -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _patent_detail_dict(row: pd.Series, summary_max: int) -> Dict[str, str]:
    """右パネル・クリック用のプレーンテキスト（キーは固定）。"""
    out: Dict[str, str] = {}
    vcorp = row.get("corporation")
    if vcorp is not None and not (isinstance(vcorp, float) and pd.isna(vcorp)):
        if isinstance(vcorp, list):
            out["corporation"] = ", ".join(str(x) for x in vcorp)
        else:
            out["corporation"] = str(vcorp).strip()
    for k in (
        "patent_number",
        "patent_name",
        "date",
        "ipc",
        "lead_ipc",
        "fi",
        "fterm",
        "keyword",
        "year",
        "year_month",
        "topic_id",
        "inventors",
        "lead_inventor",
    ):
        if k in row.index:
            v = row.get(k)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                out[k] = str(v).strip()
    summ_cols = ("abstract", "summary", "description", "patent_abstract")
    for c in summ_cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            out["_summary_field"] = c
            out["description"] = _clip(str(row[c]), summary_max)
            break
    return out


def _patent_hover_row(row: pd.Series, summary_max: int) -> str:
    """ホバー用 HTML（コントラストの高いラベル付き・読みやすい幅）。"""
    parts: List[str] = []
    pid = row.get("patent_number", "")
    if pd.notna(pid):
        parts.append(
            f"<span style=\"color:#f8fafc;font-size:14px;font-weight:600;letter-spacing:0.02em\">"
            f"{html.escape(str(pid))}</span>"
        )

    ptitle = ""
    for c in ("patent_name", "title", "patent_title", "invention_title"):
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            ptitle = _clip(str(row[c]), 160)
            break
    if ptitle:
        parts.append(
            f"<span style=\"color:#e8edf4;font-size:12px;line-height:1.35;display:block;max-width:400px\">"
            f"{html.escape(ptitle)}</span>"
        )

    vcorp = row.get("corporation")
    if vcorp is not None and not (isinstance(vcorp, float) and pd.isna(vcorp)):
        if isinstance(vcorp, list) and vcorp:
            corp_s = ", ".join(str(x) for x in vcorp[:12])
            if len(vcorp) > 12:
                corp_s += " …"
        else:
            corp_s = str(vcorp).strip()
        if corp_s:
            parts.append(
                f"<div style=\"color:#bae6fd;font-size:11px;margin-top:4px;max-width:400px\">"
                f"<b style=\"color:#7dd3fc\">企業:</b> {html.escape(_clip(corp_s, 200))}</div>"
            )

    kv_style = "color:#e2e8f0;font-size:11px;line-height:1.45;max-width:400px"
    kv: List[str] = []
    for label, col in (
        ("Lead_IPC", "lead_ipc"),
        ("IPC", "ipc"),
        ("FI", "fi"),
        ("F-term", "fterm"),
        ("year", "year"),
        ("year_month", "year_month"),
        ("topic_id", "topic_id"),
    ):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            kv.append(
                f"<div style=\"{kv_style}\"><b style=\"color:#94a3b8;min-width:4.5em;display:inline-block\">"
                f"{label}</b> "
                f"{html.escape(_clip(str(row[col]), 180))}</div>"
            )
    if kv:
        parts.append(
            "<div style=\"margin-top:6px;padding-top:6px;border-top:1px solid rgba(100,116,139,0.45)\">"
            + "".join(kv)
            + "</div>"
        )

    summ_cols = ("abstract", "summary", "description", "patent_abstract")
    for c in summ_cols:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(
                f"<div style=\"margin-top:8px;color:#cbd5e1;font-size:10.5px;line-height:1.45;"
                f"border-top:1px solid rgba(148,163,184,0.35);padding-top:8px;max-width:420px\">"
                f"{html.escape(_clip(str(row[c]), summary_max))}</div>"
            )
            break

    inner = "<br/>".join(parts) if parts else html.escape(str(pid))
    return (
        "<div style=\"min-width:260px;max-width:440px;padding:2px 0;font-family:system-ui,sans-serif\">"
        + inner
        + "</div>"
    )


def build_interactive_payload(
    df: pd.DataFrame,
    data: Data,
    num_corps: int,
    corps: List,
    patents: List,
    z_np: np.ndarray,
    year_val: int,
    summary_max: int = 450,
) -> Dict[str, Any]:
    """
    指定年のグラフ ``data`` と潜在座標 ``z_np`` から、インタラクティブ地図用 dict を返す。

    - ``patents``: id, x, y, hover, joint, coApplicants, leadIpc, fi, fterm, detail
    - ``corporations``: name, x, y, hover, patentsYear, patentsTotal
    - ``corpToPatentIds``: 企業名 -> その年にグラフ上で隣接する特許 id のリスト
    """
    patent_to_idx = {p: num_corps + i for i, p in enumerate(patents)}
    corp_to_idx = {c: i for i, c in enumerate(corps)}

    ei = data.edge_index.detach().cpu().numpy()
    pairs: set = set()
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a < num_corps <= b:
            pairs.add((a, b))
        elif b < num_corps <= a:
            pairs.add((b, a))

    active_corp_idx = sorted({c for c, _ in pairs})
    active_pat_idx = sorted({p for _, p in pairs})

    corp_to_patent_ids: Dict[str, List[str]] = {}
    for c_idx, p_idx in pairs:
        cname = corps[c_idx]
        pid = patents[p_idx - num_corps]
        corp_to_patent_ids.setdefault(cname, []).append(str(pid))
    for k in corp_to_patent_ids:
        corp_to_patent_ids[k] = sorted(set(corp_to_patent_ids[k]))

    # 特許 id -> 代表行（先頭）
    first_row: Dict[Any, pd.Series] = {}
    for _, row in df.iterrows():
        pn = row.get("patent_number")
        if pn not in first_row:
            first_row[pn] = row

    patent_entries: List[Dict[str, Any]] = []
    for p_idx in active_pat_idx:
        pid = patents[p_idx - num_corps]
        row = first_row.get(pid)
        if row is None:
            continue
        z = z_np[p_idx]
        clist = row["corporation"] if isinstance(row["corporation"], list) else []
        patent_entries.append(
            {
                "id": str(pid),
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": _patent_hover_row(row, summary_max),
                "joint": len(clist) > 1,
                "coApplicants": [str(c) for c in clist],
                "leadIpc": _cell_str(row, "lead_ipc"),
                "fi": _cell_str(row, "fi"),
                "fterm": _cell_str(row, "fterm"),
                "detail": _patent_detail_dict(row, summary_max),
            }
        )

    corp_entries: List[Dict[str, Any]] = []
    for c_idx in active_corp_idx:
        cname = corps[c_idx]
        z = z_np[c_idx]
        n_year = len(corp_to_patent_ids.get(cname, []))
        n_total = sum(
            1 for _, r in df.iterrows() if cname in (r.get("corporation") or [])
        )
        corp_entries.append(
            {
                "name": cname,
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": (
                    f"<span style=\"color:#f8fafc;font-weight:600\">{html.escape(str(cname))}</span><br/>"
                    f"<span style=\"color:#cbd5e1;font-size:11px\">この年のグラフ: {n_year} 件<br/>"
                    f"全期間: {n_total} 件</span>"
                ),
                "patentsYear": n_year,
                "patentsTotal": int(n_total),
                "detail": {
                    "name": str(cname),
                    "patentsYear": str(n_year),
                    "patentsTotal": str(int(n_total)),
                    "year": str(int(year_val)),
                },
            }
        )

    return {
        "patents": patent_entries,
        "corporations": corp_entries,
        "corpToPatentIds": corp_to_patent_ids,
        "year": int(year_val),
    }


def _paper_hover_row_arxiv(row: pd.Series, summary_max: int) -> str:
    """著者–論文 CSV 行からホバー HTML（`paper_id` / `authors_list` 想定）。"""
    parts: List[str] = []
    pid = row.get("paper_id", "")
    if pd.notna(pid):
        parts.append(f"<b>{html.escape(str(pid))}</b>")
    for c in ("title",):
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(html.escape(_clip(str(row[c]), 140)))
            break
    for c in ("description", "abstract"):
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            parts.append(html.escape(_clip(str(row[c]), summary_max)))
            break
    meta = []
    if "year" in row.index and pd.notna(row["year"]):
        meta.append(f"year: {html.escape(str(row['year']))}")
    for c in ("url", "topic"):
        if c in row.index and pd.notna(row[c]):
            meta.append(f"{c}: {html.escape(str(row[c])[:120])}")
    if meta:
        parts.append("<br/>".join(meta))
    return "<br/>".join(parts) if parts else str(pid)


def build_interactive_payload_author_paper(
    df: pd.DataFrame,
    data: Data,
    num_authors: int,
    authors: List,
    papers: List,
    z_np: np.ndarray,
    year_val: int,
    summary_max: int = 450,
) -> Dict[str, Any]:
    """
    著者–論文二部グラフ用。キー構造は `build_interactive_payload` と同一（HTML テンプレート共用）。

    - ``patents`` スロット: 論文（id = ``paper_id``）
    - ``corporations`` スロット: 著者名
    - ``joint`` / ``coApplicants``: 共著（``authors_list`` の長さ）
    """
    paper_to_idx = {p: num_authors + i for i, p in enumerate(papers)}
    author_to_idx = {a: i for i, a in enumerate(authors)}

    ei = data.edge_index.detach().cpu().numpy()
    pairs: set = set()
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a < num_authors <= b:
            pairs.add((a, b))
        elif b < num_authors <= a:
            pairs.add((b, a))

    active_author_idx = sorted({c for c, _ in pairs})
    active_paper_idx = sorted({p for _, p in pairs})

    author_to_paper_ids: Dict[str, List[str]] = {}
    for a_idx, p_idx in pairs:
        aname = authors[a_idx]
        pid = papers[p_idx - num_authors]
        author_to_paper_ids.setdefault(aname, []).append(str(pid))
    for k in author_to_paper_ids:
        author_to_paper_ids[k] = sorted(set(author_to_paper_ids[k]))

    first_row: Dict[Any, pd.Series] = {}
    id_col = "paper_id"
    for _, row in df.iterrows():
        key = str(row.get(id_col, ""))
        if key and key not in first_row:
            first_row[key] = row

    paper_entries: List[Dict[str, Any]] = []
    for p_idx in active_paper_idx:
        pid = papers[p_idx - num_authors]
        row = first_row.get(str(pid))
        if row is None:
            continue
        z = z_np[p_idx]
        alist = row["authors_list"] if isinstance(row.get("authors_list"), list) else []
        paper_entries.append(
            {
                "id": str(pid),
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": _paper_hover_row_arxiv(row, summary_max),
                "joint": len(alist) > 1,
                "coApplicants": [str(x) for x in alist],
            }
        )

    author_entries: List[Dict[str, Any]] = []
    for a_idx in active_author_idx:
        aname = authors[a_idx]
        z = z_np[a_idx]
        n_year = len(author_to_paper_ids.get(aname, []))
        n_total = sum(
            1
            for _, r in df.iterrows()
            if aname in (r.get("authors_list") or [])
        )
        author_entries.append(
            {
                "name": aname,
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": html.escape(str(aname))
                + f"<br/>この年のグラフ: {n_year} 本<br/>全期間: {n_total} 本",
                "patentsYear": n_year,
                "patentsTotal": int(n_total),
            }
        )

    return {
        "patents": paper_entries,
        "corporations": author_entries,
        "corpToPatentIds": author_to_paper_ids,
        "year": int(year_val),
    }


def _topic_hover_row(
    row: pd.Series, topic_column: str, summary_max: int
) -> str:
    t = row.get(topic_column, "")
    parts: List[str] = []

    parts.append(
        f'<span style="color:#f8fafc;font-size:14px;font-weight:700;letter-spacing:0.02em">'
        f'{html.escape(str(t))}</span>'
    )

    if "title" in row.index and pd.notna(row["title"]) and str(row["title"]).strip():
        parts.append(
            f'<span style="color:#e2e8f0;font-size:12px;line-height:1.4;display:block;max-width:400px">'
            f'{html.escape(_clip(str(row["title"]), 160))}</span>'
        )

    meta: List[str] = []
    if "year" in row.index and pd.notna(row["year"]):
        meta.append(
            f'<b style="color:#94a3b8">year</b> {html.escape(str(row["year"]))}'
        )
    if "url" in row.index and pd.notna(row["url"]) and str(row["url"]).strip():
        meta.append(
            f'<b style="color:#94a3b8">url</b> {html.escape(_clip(str(row["url"]), 80))}'
        )
    if meta:
        parts.append(
            '<div style="margin-top:5px;color:#cbd5e1;font-size:11px;line-height:1.4">'
            + "<br/>".join(meta)
            + "</div>"
        )

    if "description" in row.index and pd.notna(row["description"]) and str(row["description"]).strip():
        parts.append(
            f'<div style="margin-top:6px;color:#a8b4c8;font-size:10.5px;line-height:1.4;'
            f'border-top:1px solid rgba(148,163,184,0.3);padding-top:6px;max-width:420px">'
            f'{html.escape(_clip(str(row["description"]), summary_max))}</div>'
        )

    inner = "<br/>".join(parts) if parts else html.escape(str(t))
    return (
        '<div style="min-width:220px;max-width:440px;padding:2px 0;font-family:system-ui,sans-serif">'
        + inner
        + "</div>"
    )


def build_interactive_payload_author_topic(
    df: pd.DataFrame,
    data: Data,
    num_authors: int,
    authors: List,
    topics: List,
    z_np: np.ndarray,
    year_val: int,
    *,
    topic_column: str = "topic",
    summary_max: int = 450,
) -> Dict[str, Any]:
    """
    著者–トピック二部グラフ用。JSON キーは論文版と同一（右ノードを ``patents`` スロットに載せる）。
    """
    topic_to_idx = {t: num_authors + i for i, t in enumerate(topics)}
    author_to_idx = {a: i for i, a in enumerate(authors)}

    ei = data.edge_index.detach().cpu().numpy()
    pairs: set = set()
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        if a < num_authors <= b:
            pairs.add((a, b))
        elif b < num_authors <= a:
            pairs.add((b, a))

    active_author_idx = sorted({c for c, _ in pairs})
    active_topic_idx = sorted({p for _, p in pairs})

    author_to_topic_ids: Dict[str, List[str]] = {}
    for a_idx, t_idx in pairs:
        aname = authors[a_idx]
        tid = topics[t_idx - num_authors]
        author_to_topic_ids.setdefault(aname, []).append(str(tid))
    for k in author_to_topic_ids:
        author_to_topic_ids[k] = sorted(set(author_to_topic_ids[k]))

    first_row: Dict[str, pd.Series] = {}
    if topic_column not in df.columns:
        raise ValueError(f"列 '{topic_column}' がありません")
    for _, row in df.iterrows():
        key = str(row[topic_column]) if pd.notna(row[topic_column]) else ""
        if key and key not in first_row:
            first_row[key] = row

    topic_entries: List[Dict[str, Any]] = []
    for t_idx in active_topic_idx:
        tid = topics[t_idx - num_authors]
        sk = str(tid)
        row = first_row.get(sk)
        if row is None:
            row = pd.Series({topic_column: tid})
        z = z_np[t_idx]

        detail: Dict[str, str] = {"topic": sk}
        for col in ("title", "description", "url", "year"):
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                val = str(row[col]).strip()
                if col == "description":
                    val = _clip(val, summary_max)
                detail[col] = val

        n_authors_linked = sum(
            1 for a_idx, t_idx2 in pairs if t_idx2 == t_idx
        )
        detail["linked_authors"] = str(n_authors_linked)

        topic_entries.append(
            {
                "id": sk,
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": _topic_hover_row(row, topic_column, summary_max),
                "joint": False,
                "coApplicants": [],
                "detail": detail,
            }
        )

    author_paper_counts: Dict[str, int] = {}
    for _, r in df.iterrows():
        for a in (r.get("authors_list") or []):
            author_paper_counts[a] = author_paper_counts.get(a, 0) + 1

    author_entries: List[Dict[str, Any]] = []
    for a_idx in active_author_idx:
        aname = authors[a_idx]
        z = z_np[a_idx]
        n_year = len(author_to_topic_ids.get(aname, []))
        n_total = author_paper_counts.get(aname, 0)
        author_entries.append(
            {
                "name": aname,
                "x": float(z[0]),
                "y": float(z[1]),
                "hover": html.escape(str(aname))
                + f"<br/>この年のグラフ: {n_year} トピック<br/>全期間: {n_total} 本",
                "patentsYear": n_year,
                "patentsTotal": int(n_total),
            }
        )

    return {
        "patents": topic_entries,
        "corporations": author_entries,
        "corpToPatentIds": author_to_topic_ids,
        "year": int(year_val),
    }
