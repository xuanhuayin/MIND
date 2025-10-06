# -*- coding: utf-8 -*-
"""
Build a unified TEXT_EVT parquet for sub-01 (Friends + Movies).
(Updated) Support aggregated schema with columns:
['text_per_tr', 'words_per_tr', 'onsets_per_tr', 'durations_per_tr']
"""

from __future__ import annotations
from pathlib import Path
import os, re, csv, ast
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

PIPE_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
STIM_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors/stimuli")
GRID_2HZ   = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"

OUT_PARQUET   = PIPE_ROOT / "timelines" / "text_events_sub-01.parquet"
OUT_LOG_CSV   = PIPE_ROOT / "timelines" / "text_events_sub-01_log.csv"
OUT_DUMP_DIR  = PIPE_ROOT / "timelines" / "text_events_dumps_sub-01"
OUT_DUMP_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "sub-01"

RE_TASK    = re.compile(r"task-([A-Za-z0-9_]+)$")
RE_FRIENDS = re.compile(r"^(s\d{2}e\d{2}[a-d])$", re.IGNORECASE)
RE_BOURNE  = re.compile(r"^(bourne\d+)$", re.IGNORECASE)

# 允许 token 里出现连字符 -
RE_TASK = re.compile(r"task-([A-Za-z0-9_\-]+)$")

def dataset_to_task(ds: str) -> Optional[str]:
    """
    从 dataset 名里抽取 task token，例如：
      ses-008_task-figures05_run-1 -> figures05_run-1
      ses-003_task-wolf04          -> wolf04
      ses-007_task-life03_run-1    -> life03_run-1
    """
    m = RE_TASK.search(ds.strip())
    return m.group(1).lower() if m else None

def resolve_transcript_path(task: str) -> Optional[Path]:
    """
    根据 task 名在 transcripts 目录下寻找对应 tsv。
    - 先按已知结构尝试（friends、movie10/bourne）
    - 再尝试 figures/life/wolf 的常见命名
    - 最后做递归 rglob 兜底（含若干 task 变体）
    """
    base = STIM_ROOT / "transcripts"

    # Friends: sXXeYY[a-d]
    if RE_FRIENDS.match(task):
        season = task[:3]
        for p in [
            base / "friends" / season / f"friends_{task}.tsv",
            base / "friends" / season / f"text_event_{task}.tsv",
        ]:
            if p.exists(): return p

    # Bourne: bourne\d+
    if RE_BOURNE.match(task):
        for p in [
            base / "movie10" / "bourne" / f"movie10_{task}.tsv",
            base / "movie10" / "bourne" / f"text_event_{task}.tsv",
            base / "movie10" / "bourne" / f"{task}.tsv",
        ]:
            if p.exists(): return p

    # 其它类别（figures / life / wolf），直接尝试若干常见目录
    for cat in ["figures", "life", "wolf"]:
        for p in [
            base / cat / f"{task}.tsv",
            base / cat / f"text_event_{task}.tsv",
            base / cat / f"{cat}_{task}.tsv",
        ]:
            if p.exists(): return p

    # 变体：有些文件可能不带 run 后缀，或连接符不同
    variants = {task}
    # 去掉 _run-1 / -run-1
    variants.add(re.sub(r"[_-]run-?\d+$", "", task))
    # 把下划线换成连字符、或相反
    variants.add(task.replace("_", "-"))
    variants.add(task.replace("-", "_"))

    # 递归兜底：在 transcripts 根目录下找 *{variant}*.tsv
    candidates: list[Path] = []
    if base.exists():
        for v in variants:
            candidates += list(base.rglob(f"*{v}*.tsv"))
    if candidates:
        candidates.sort(key=lambda x: (len(str(x)), str(x)))
        return candidates[0]

    return None

# ---------- robust parsing helpers ----------
def _maybe_literal_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None

def _split_guess(s: str) -> List[str]:
    # remove brackets
    t = s.strip().strip("[]()")
    # try semicolon, then comma, then whitespace
    if ";" in t:
        parts = [x.strip() for x in t.split(";")]
    elif "," in t:
        parts = [x.strip() for x in t.split(",")]
    else:
        parts = [x.strip() for x in t.split()]
    # drop empties & surrounding quotes
    parts = [p.strip().strip("'").strip('"') for p in parts if p.strip() != ""]
    return parts

def _parse_seq_column_to_list(x) -> List:
    """
    Try to parse a column cell that should represent a sequence.
    Accept: python-literal list/tuple/np array as string, or delimited string.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if not isinstance(x, str):
        return [x]
    val = _maybe_literal_eval(x)
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return _split_guess(x)

def _to_float_list(vs: List) -> List[float]:
    out = []
    for v in vs:
        try:
            out.append(float(v))
        except Exception:
            # if it was something like "0.23s"
            try:
                out.append(float(str(v).rstrip("s")))
            except Exception:
                out.append(np.nan)
    return out

# ---------- reader ----------
def read_transcript_tsv(path: Path) -> pd.DataFrame:
    """
    Normalize any of the following into per-word rows with columns: onset, offset, word
    Schemas:
      A) onset, offset, word
      B) start, end, word/text
      C) onset, duration, word/text  (offset = onset+duration)
      D) t0, t1, word/text
      E) aggregated per-turn:
         text_per_tr, words_per_tr, onsets_per_tr, durations_per_tr
            - each cell holds a sequence; expand to per-word
    """
    df = pd.read_csv(path, sep="\t", header=0, quoting=csv.QUOTE_NONE, encoding="utf-8", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]

    def has(*cols):
        return all(c in df.columns for c in cols)

    # A
    if has("onset", "offset", "word"):
        out = df[["onset", "offset", "word"]].copy()
    # B
    elif has("start", "end") and ("word" in df.columns or "text" in df.columns):
        txtcol = "word" if "word" in df.columns else "text"
        out = df.rename(columns={"start": "onset", "end": "offset", txtcol: "word"})[["onset", "offset", "word"]]
    # C
    elif has("onset", "duration") and ("word" in df.columns or "text" in df.columns):
        txtcol = "word" if "word" in df.columns else "text"
        out = df.rename(columns={txtcol: "word"})[["onset", "duration", "word"]].copy()
        out["offset"] = pd.to_numeric(out["onset"], errors="coerce") + pd.to_numeric(df["duration"], errors="coerce")
        out = out[["onset", "offset", "word"]]
    # D
    elif has("t0", "t1") and ("word" in df.columns or "text" in df.columns):
        txtcol = "word" if "word" in df.columns else "text"
        out = df.rename(columns={"t0": "onset", "t1": "offset", txtcol: "word"})[["onset", "offset", "word"]]
    # E aggregated per-turn
    elif has("words_per_tr", "onsets_per_tr", "durations_per_tr"):
        words_col = "words_per_tr"
        # try to prefer per-word column if exists
        # Some files have 'text_per_tr' as whole sentence; we prefer 'words_per_tr'
        W_list = df[words_col].apply(_parse_seq_column_to_list).tolist()
        S_list = df["onsets_per_tr"].apply(_parse_seq_column_to_list).apply(_to_float_list).tolist()
        D_list = df["durations_per_tr"].apply(_parse_seq_column_to_list).apply(_to_float_list).tolist()

        rows = []
        for words, onsets, durs in zip(W_list, S_list, D_list):
            n = min(len(words), len(onsets), len(durs))
            for i in range(n):
                w = str(words[i]).strip()
                s = float(onsets[i]) if onsets[i] is not None else np.nan
                d = float(durs[i]) if durs[i] is not None else np.nan
                if not (np.isfinite(s) and np.isfinite(d)):
                    continue
                rows.append((s, s + d, w))
        out = pd.DataFrame(rows, columns=["onset", "offset", "word"])
    else:
        # try best-effort: find two time columns + one text column
        time_cols = [c for c in df.columns if c in ("onset","offset","start","end","t0","t1","begin","finish","time_start","time_end")]
        text_cols = [c for c in df.columns if c in ("word","text","token","content","words_per_tr","text_per_tr")]
        if ("words_per_tr" in text_cols) and ("onsets_per_tr" in df.columns) and ("durations_per_tr" in df.columns):
            # route back to aggregated parser
            return read_transcript_tsv(path)  # recursion will hit the aggregated branch
        if len(time_cols) >= 2 and any(c in df.columns for c in ("word","text","token","content")):
            c0, c1 = time_cols[:2]
            txtcol = "word" if "word" in df.columns else ("text" if "text" in df.columns else ("token" if "token" in df.columns else "content"))
            out = df.rename(columns={c0:"onset", c1:"offset", txtcol:"word"})[["onset","offset","word"]].copy()
        else:
            raise RuntimeError(f"Unrecognized transcript schema in {path} (cols={list(df.columns)})")

    # clean
    out["onset"] = pd.to_numeric(out["onset"], errors="coerce")
    out["offset"] = pd.to_numeric(out["offset"], errors="coerce")
    out = out.dropna(subset=["onset","offset"])
    out = out[out["offset"] >= out["onset"]]
    out["word"] = out["word"].astype(str).str.strip()
    return out.reset_index(drop=True)

def main():
    grid = pd.read_parquet(GRID_2HZ)
    datasets = sorted(grid["dataset"].drop_duplicates().tolist())
    print(f"[INFO] Total datasets in grid: {len(datasets)}")

    all_rows, log_rows = [], []
    for ds in datasets:
        task = dataset_to_task(ds)
        if not task:
            log_rows.append([ds, "", "no_task_token"])
            print(f"[WARN] {ds}: cannot parse task token; skipping.")
            continue
        tsv_path = resolve_transcript_path(task)
        if tsv_path is None:
            log_rows.append([ds, "", "transcript_not_found"])
            print(f"[WARN] {ds}: transcript not found for task={task}")
            continue
        try:
            df = read_transcript_tsv(tsv_path)
        except Exception as e:
            log_rows.append([ds, str(tsv_path), f"read_failed: {e}"])
            print(f"[WARN] {ds}: failed to read transcript {tsv_path} -> {e}")
            continue

        df["dataset"] = ds
        df["subject"] = SUBJECT
        df = df[["dataset","subject","onset","offset","word"]]
        all_rows.append(df)
        log_rows.append([ds, str(tsv_path), "ok"])
        dump_path = OUT_DUMP_DIR / f"{ds}.tsv"
        df.to_csv(dump_path, sep="\t", index=False, encoding="utf-8")
        print(f"[OK] {ds}: {len(df)} words from {tsv_path.name}")

    big = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["dataset","subject","onset","offset","word"]
    )
    big.to_parquet(OUT_PARQUET, index=False)
    print(f"\n[DONE] {OUT_PARQUET} (rows={len(big)})")
    pd.DataFrame(log_rows, columns=["dataset","transcript_path","status"]).to_csv(OUT_LOG_CSV, index=False)
    print(f"[LOG ] {OUT_LOG_CSV}")

if __name__ == "__main__":
    main()