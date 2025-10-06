# -*- coding: utf-8 -*-
"""
快速探查 Algonauts 2025 transcripts 文件格式（friends/movie10）。
- 遍历 stimuli/transcripts/** 下的 tsv/csv/json
- 打印每个文件的列名/类型/样例
- 如发现 *_per_tr 的列（如 words_per_tr、onsets_per_tr、durations_per_tr、text_per_tr），
  尝试把前两行解析为 Python 列表并展示长度与部分内容（便于确认“逐 TR 列表”结构）
用法：
  python inspect_transcripts.py                # 随机取 10 个文件
  python inspect_transcripts.py --limit 5      # 取 5 个
  python inspect_transcripts.py --pattern s06  # 过滤文件名包含 s06 的
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, random, ast
import pandas as pd

DATA_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
TRANS_ROOTS = [
    DATA_ROOT / "stimuli" / "transcripts",
    DATA_ROOT / "stimuli" / "Transcripts",
]

def find_files(pattern: str|None, limit: int):
    all_files = []
    for root in TRANS_ROOTS:
        if root.exists():
            all_files += list(root.rglob("*"))
    files = [p for p in all_files if p.suffix.lower() in (".tsv", ".csv", ".json")]
    if pattern:
        pattern_low = pattern.lower()
        files = [p for p in files if pattern_low in str(p).lower()]
    # 随机取样，避免爆屏
    random.shuffle(files)
    return files[:limit]

def sniff_tsv_csv(fp: Path) -> pd.DataFrame:
    sep = "\t" if fp.suffix.lower() == ".tsv" else ","
    return pd.read_csv(fp, sep=sep)

def sniff_json(fp: Path) -> pd.DataFrame|dict|list:
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        data = json.loads(fp.read_text(encoding="latin-1"))
    return data

def try_parse_list_cell(x):
    """把像 '["a","b"]' 或 '[0.1, 0.2]' 这种字符串尝试转为 Python 列表；失败就原样返回。"""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                val = ast.literal_eval(s)
                return val
            except Exception:
                return x
    return x

def print_df_summary(df: pd.DataFrame, fp: Path):
    print(f"\n=== {fp} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:", dict(df.dtypes))

    head = df.head(2).copy()
    # 尝试解析 *_per_tr 列
    per_tr_cols = [c for c in head.columns if c.lower().endswith("_per_tr")]
    if per_tr_cols:
        print("detected *_per_tr columns:", per_tr_cols)
        for c in per_tr_cols:
            parsed = head[c].apply(try_parse_list_cell)
            lens = []
            previews = []
            for val in parsed.tolist():
                if isinstance(val, (list, tuple)):
                    lens.append(len(val))
                    previews.append(val[:5])
                else:
                    lens.append(None)
                    previews.append(str(val)[:80])
            print(f"  - {c}: lens_first_two={lens}, previews_first_two={previews}")
    else:
        print("no *_per_tr columns detected")

    # 打印前两行（截断）
    with pd.option_context('display.max_colwidth', 200):
        print("\nhead(2):")
        print(head)

def print_json_summary(obj, fp: Path):
    print(f"\n=== {fp} ===")
    if isinstance(obj, dict):
        print("json keys:", list(obj.keys()))
        # 常见：{"words":[{start:..., end:..., word:...}, ...]} 或 {"text_per_tr": [...], ...}
        for k in list(obj.keys())[:4]:
            v = obj[k]
            if isinstance(v, list):
                print(f"  - {k}: list(len={len(v)}) sample_first={v[:2]}")
            else:
                print(f"  - {k}: type={type(v).__name__}")
    elif isinstance(obj, list):
        print("top-level list, len:", len(obj))
        print("first elem type:", type(obj[0]).__name__ if obj else None)
        if obj and isinstance(obj[0], dict):
            keys = list(obj[0].keys())
            print("first elem keys:", keys)
            print("sample_first_two:", obj[:2])
        else:
            print("sample_first_two:", obj[:2])
    else:
        print("unknown json root type:", type(obj).__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str, default=None, help="只检查文件名里包含该子串的文件（如 s06）")
    ap.add_argument("--limit", type=int, default=10, help="最多检查多少个文件")
    args = ap.parse_args()

    files = find_files(args.pattern, args.limit)
    if not files:
        print("No transcript files found with given pattern.")
        return

    for fp in files:
        try:
            if fp.suffix.lower() in (".tsv", ".csv"):
                df = sniff_tsv_csv(fp)
                print_df_summary(df, fp)
            else:
                obj = sniff_json(fp)
                if isinstance(obj, (dict, list)):
                    print_json_summary(obj, fp)
                else:
                    print(f"\n=== {fp} ===")
                    print("Unsupported JSON structure")
        except Exception as e:
            print(f"\n=== {fp} ===")
            print("ERROR while reading:", e)

if __name__ == "__main__":
    main()
