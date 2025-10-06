# -*- coding: utf-8 -*-
"""
Extract 2Hz *timed* text embeddings (LLaMA) for sub-01, matching paper/source:
- For each word w, prepend the preceding k=1024 words (including w) -> contextualized embedding
- For each intermediate transformer layer l, average tokens overlapping w
- Build a 2Hz time grid: for each bin, SUM embeddings of words overlapping the bin
- Output per dataset: (T_2Hz, N_LAYERS, D_text), default D_text = model.hidden_size (3072)

Inputs:
  - pipeline_TRIBE/timelines/grid_2hz_sub-01.parquet      # 2Hz bins per dataset
  - pipeline_TRIBE/timelines/text_events_sub-01.parquet   # per-word rows

ENV (optional):
  ALGONAUTS_TEXT_MODEL      default "meta-llama/Llama-3.2-3B"
  ALGONAUTS_TEXT_REV        default None
  ALGONAUTS_TEXT_BATCH      default 8       # number of *words* per forward
  ALGONAUTS_TEXT_CTX_WORDS  default 1024    # preceding words count (including current)
  TEXT_SAVE_FP16            default 0       # 1 to save npy as float16 (accumulation stays float32)
  USE_DATASET_FILTER        default 1       # 1 to use split files; 0 to process all datasets
"""

from __future__ import annotations
from pathlib import Path
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# ───────── paths ─────────
PIPE_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
GRID_2HZ = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
TEXT_EVT = PIPE_ROOT / "timelines" / "text_events_sub-01.parquet"
OUT_DIR = PIPE_ROOT / "TRIBE_8features" / "text_2hz" / "sub-01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 可选分割（与视频一致）
SPLIT_FULL = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8_full_episodes.txt"
SPLIT_FALLBACK = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8.txt"

SUBJECT = "sub-01"
SAVE_FP16 = bool(int(os.environ.get("TEXT_SAVE_FP16", "0")))  # 存盘是否降到 FP16
USE_FILTER = bool(int(os.environ.get("USE_DATASET_FILTER", "1")))  # 是否使用数据集过滤
ACC_DTYPE = np.float32  # 累加用 FP32，防止溢出
SAVE_DTYPE = np.float32

# ───────── model & batching ─────────
TEXT_MODEL_ID = os.environ.get("ALGONAUTS_TEXT_MODEL", "meta-llama/Llama-3.2-3B")
TEXT_REVISION = os.environ.get("ALGONAUTS_TEXT_REV", None) or None
BATCH_WORDS = int(os.environ.get("ALGONAUTS_TEXT_BATCH", 8))
CONTEXT_WORDS = int(os.environ.get("ALGONAUTS_TEXT_CTX_WORDS", 1024))
TOKENS_CAP = None  # 可选：上限 token 数；None 表示不裁（交由模型 max_len 控制）


# ───────── utils ─────────
def tokenize_per_word(tokenizer, words: List[str]) -> List[List[int]]:
    toks = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        toks.append(ids)
    return toks


def build_batch_inputs(
        all_word_tokens: List[List[int]],
        positions: List[int],
        context_words: int,
        tokenizer,
        tokens_cap: Optional[int] = None,
):
    seqs = []
    spans: List[Tuple[int, int]] = []
    for i in positions:
        st = max(0, i - context_words + 1)
        toks = [tid for ws in all_word_tokens[st:i + 1] for tid in ws]
        last_len = len(all_word_tokens[i])
        seq_len = len(toks)
        last_s = seq_len - last_len
        last_e = seq_len

        if tokens_cap is not None and seq_len > tokens_cap:
            cut = seq_len - tokens_cap
            toks = toks[cut:]
            last_s -= cut
            last_e -= cut
            if last_s < 0:
                toks = all_word_tokens[i][-tokens_cap:] if tokens_cap else all_word_tokens[i]
                last_s, last_e = 0, len(toks)

        seqs.append(toks)
        spans.append((last_s, last_e))

    if not seqs:
        return None, None, []

    maxlen = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = []
    attn_mask = []
    for s in seqs:
        pad = maxlen - len(s)
        input_ids.append(s + [pad_id] * pad)
        attn_mask.append([1] * len(s) + [0] * pad)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attn_mask, dtype=torch.long),
        spans,
    )


def pool_word_from_hidden(hidden_states, spans: List[Tuple[int, int]]):
    """
    hidden_states: Tuple[Tensor], each [B, T, H]; index 0 is embeddings in HF.
    Output: [B, L, H] (L = #layers without embeddings)
    """
    hs_layers = hidden_states[1:]  # drop embeddings layer
    B, T, H = hs_layers[-1].shape
    L = len(hs_layers)

    outs = []
    for l in range(L):
        h = hs_layers[l]  # [B, T, H]
        pooled = []
        for b, (s, e) in enumerate(spans):
            s = max(0, min(int(s), T))
            e = max(0, min(int(e), T))
            if e <= s:
                pooled.append(torch.zeros(H, device=h.device, dtype=h.dtype))
            else:
                pooled.append(h[b, s:e, :].mean(dim=0))
        pooled = torch.stack(pooled, dim=0)  # [B, H]
        outs.append(pooled)
    return torch.stack(outs, dim=1)  # [B, L, H]


def find_overlapping_bins(bin_starts: np.ndarray, bin_ends: np.ndarray, w0: float, w1: float) -> np.ndarray:
    lo = np.maximum(bin_starts, w0)
    hi = np.minimum(bin_ends, w1)
    return np.nonzero(lo < hi)[0]


# ───────── main ─────────
def main():
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    assert TEXT_EVT.exists(), f"Missing text events parquet: {TEXT_EVT}"

    # 读入
    grid = pd.read_parquet(GRID_2HZ)
    txt = pd.read_parquet(TEXT_EVT)

    # 兼容列名：支持三种情况
    # 1. w_start/w_end -> onset/offset
    # 2. onset/duration -> onset/offset
    # 3. onset/offset (已经是正确格式)
    if {"w_start", "w_end"}.issubset(txt.columns):
        txt = txt.rename(columns={"w_start": "onset", "w_end": "offset"})
    elif {"onset", "duration"}.issubset(txt.columns) and "offset" not in txt.columns:
        txt["offset"] = txt["onset"].to_numpy(float) + np.maximum(0.0, txt["duration"].to_numpy(float))
    elif {"onset", "offset"}.issubset(txt.columns):
        # 已经有 onset 和 offset，无需处理
        pass
    else:
        raise SystemExit(
            f"Text events parquet needs (w_start,w_end), (onset,duration), or (onset,offset). Got: {list(txt.columns)}")

    # 确保 subject 列（若无则默认全是 sub-01）
    if "subject" not in grid.columns:
        grid["subject"] = SUBJECT
    if "subject" not in txt.columns:
        txt["subject"] = SUBJECT

    # 过滤被试
    grid = grid[grid["subject"] == SUBJECT].copy()
    txt = txt[txt["subject"] == SUBJECT].copy()

    # 根据 split 过滤（可选）
    if USE_FILTER:
        split_file = None
        if SPLIT_FULL.exists():
            split_file = SPLIT_FULL
        elif SPLIT_FALLBACK.exists():
            split_file = SPLIT_FALLBACK

        if split_file is not None:
            keep = {l.strip() for l in open(split_file, "r", encoding="utf-8") if l.strip()}
            grid = grid[grid["dataset"].isin(keep)].copy()
            print(f"[INFO] Filtering to {len(keep)} datasets from {split_file.name}")
        else:
            print(f"[INFO] USE_DATASET_FILTER=1 but no split files found, processing all datasets")
    else:
        print(f"[INFO] USE_DATASET_FILTER=0, processing all datasets without filtering")

    # ── 你要求的检查输出 ─────────────────────────────────────────────
    for ds in sorted(grid["dataset"].unique()):
        n = int((txt["dataset"] == ds).sum())
        print(f"[CHECK] {ds}: text_rows={n}")
    # ───────────────────────────────────────────────────────────────

    if grid.empty:
        raise SystemExit("Empty 2Hz grid after filtering.")
    if txt.empty:
        print("[WARN] No text rows for subject after filtering.")

    # 模型
    print(f"[INFO] Load text model: {TEXT_MODEL_ID} (rev={TEXT_REVISION})")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID, revision=TEXT_REVISION)
    model = AutoModel.from_pretrained(TEXT_MODEL_ID, revision=TEXT_REVISION, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    torch.set_grad_enabled(False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    H = int(getattr(model.config, "hidden_size"))
    print(f"[INFO] hidden_size(H) = {H}  (will output {H})")

    index_rows = []

    # 按 dataset 处理
    for ds, g in grid.groupby("dataset", sort=True):
        g = g.sort_values("bin_idx").reset_index(drop=True)
        bins_start = g["win_start"].to_numpy(dtype=float)
        bins_end = g["win_end"].to_numpy(dtype=float)
        T_bins = len(g)

        words_df = txt[txt["dataset"] == ds].copy()
        words_df = words_df.sort_values(["onset", "offset"]).reset_index(drop=True)

        # 探测层数（跑极小 batch）
        with torch.no_grad():
            probe_ids = torch.tensor([[tokenizer.pad_token_id]], device=device)
            probe_mask = torch.ones_like(probe_ids)
            pout = model(input_ids=probe_ids, attention_mask=probe_mask, output_hidden_states=True)
            L = len(pout.hidden_states) - 1  # drop embeddings

        out_arr = np.zeros((T_bins, L, H), dtype=ACC_DTYPE)

        if words_df.empty:
            # 无字幕：保留零阵
            out_path = OUT_DIR / f"{ds}.npy"
            to_save = out_arr.astype(SAVE_DTYPE)
            np.save(out_path, to_save)
            index_rows.append([ds, str(out_path), T_bins, L, H])
            print(f"[INFO] {ds}: no text; zeros -> {tuple(to_save.shape)}")
            continue

        words = words_df["word"].astype(str).tolist()
        w_on = words_df["onset"].to_numpy(dtype=float)
        w_off = words_df["offset"].to_numpy(dtype=float)

        word_toks = tokenize_per_word(tokenizer, words)
        positions = list(range(len(words)))

        for bstart in range(0, len(positions), BATCH_WORDS):
            batch_pos = positions[bstart:bstart + BATCH_WORDS]
            ids, mask, spans = build_batch_inputs(word_toks, batch_pos, CONTEXT_WORDS, tokenizer, TOKENS_CAP)
            if ids is None:
                continue
            ids = ids.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
                pooled = pool_word_from_hidden(out.hidden_states, spans)  # [B, L, H]
                pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
                pooled = pooled.float().cpu().numpy().astype(np.float32)  # 防溢出/一致精度

            # 时间对齐：把每个词加到所有重叠的 2Hz bin（论文=求和）
            for j, pos in enumerate(batch_pos):
                hit = find_overlapping_bins(bins_start, bins_end, float(w_on[pos]), float(w_off[pos]))
                if hit.size == 0:
                    continue
                out_arr[hit, :, :] += pooled[j]  # FP32 累加

            del out, pooled, ids, mask
            torch.cuda.empty_cache()

            if (bstart // max(1, BATCH_WORDS)) % 20 == 0:
                print(f"[{ds}] words {bstart + len(batch_pos)}/{len(words)}")

        # 落盘（可选降到 FP16）
        out_path = OUT_DIR / f"{ds}.npy"
        if SAVE_DTYPE == np.float16:
            np.clip(out_arr, -65504.0, 65504.0, out=out_arr)
        np.save(out_path, out_arr.astype(SAVE_DTYPE))
        index_rows.append([ds, str(out_path), T_bins, L, H])
        print(f"[OK] {ds}: saved {out_path.name} shape={tuple(out_arr.shape)}")

    # 写 index
    idx = pd.DataFrame(index_rows, columns=["dataset", "npy_path", "T_2Hz", "n_layers", "dim"])
    idx.to_csv(OUT_DIR / "index.csv", index=False)
    print("\n[DONE] text_2hz index.csv ->", OUT_DIR)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()