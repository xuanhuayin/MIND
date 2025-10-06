# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys, random, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---- repo root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- model ----
from algonauts2025.standalone.weighted_moe_decoder import FmriEncoder_MoE

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions)) or [L - 1]
    if idxs[-1] != L - 1: idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]; starts = [0] + bounds[:-1]
    groups = []
    for s, e in zip(starts, bounds):
        s = max(0, min(s, L)); e = max(0, min(e, L))
        if e <= s: s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(groups, axis=0)

def parse_layers_arg(layers_arg: str, probe_L: int):
    s = (layers_arg or "").strip().lower()
    if not s: return "indices", [probe_L - 1]
    if s == "all": return "indices", list(range(probe_L))
    if s.startswith("last"):
        try: k = int(s.replace("last", "")); 
        except: k = 1
        k = max(1, min(k, probe_L)); 
        return "indices", list(range(max(0, probe_L - k), probe_L))
    if s.startswith("idx:"):
        idxs = []
        for p in [p for p in s[4:].split(",") if p.strip()]:
            try:
                i = int(p); 
                if 0 <= i < probe_L: idxs.append(i)
            except: pass
        return "indices", sorted(set(idxs or [probe_L - 1]))
    try:
        fracs = [min(1.0, max(0.0, float(x))) for x in s.split(",") if x.strip() != ""]
        return "fractions", (fracs or [1.0])
    except:
        return "indices", [probe_L - 1]

class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]): self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v): self.data[k] = v.to(device, non_blocking=True)
        return self

# ---------------- dataset (windowed) ----------------
class WindowedDataset(Dataset):
    def __init__(self, ids: List[str], video_root: Path, text_root: Path, audio_root: Path,
                 anchor_fmri_root: Path, layers_arg: str, layer_agg: str,
                 window_tr: int, stride_tr: int, frames_per_tr: int):
        self.ids = ids
        self.video_root = Path(video_root); self.text_root = Path(text_root); self.audio_root = Path(audio_root)
        self.anchor_fmri_root = Path(anchor_fmri_root)
        self.N = int(window_tr); self.S = int(stride_tr); self.f = int(frames_per_tr)

        v0 = np.load(self.video_root / f"{ids[0]}.npy")
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions": self.fracs, self.sel_indices = [float(x) for x in payload], None
        else: self.fracs, self.sel_indices = None, [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v = np.load(self.video_root / f"{ds}.npy")
            T_frames = v.shape[0]; T_tr_feat = T_frames // self.f
            arr = load_fmri_flexible(self.anchor_fmri_root, ds)
            if 1000 in arr.shape: fmri = arr if arr.shape[0] == 1000 else arr.T
            else: fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            T_tr = min(T_tr_feat, fmri.shape[1])
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr: self._index.append((ds, st))

        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")
        t_LDT = self._load_feature_LDT(self.text_root  / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")
        v_GDT = self._maybe_pick_layers(v_LDT); t_GDT = self._maybe_pick_layers(t_LDT); a_GDT = self._maybe_pick_layers(a_LDT)
        self.G, self.Dv = v_GDT.shape[0], v_GDT.shape[1]; self.Dt, self.Da = t_GDT.shape[1], a_GDT.shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        arr = np.load(path_npy)
        if arr.ndim != 3: raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))

    def _maybe_pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L] or [L - 1]
            return lat_LDT[sel]
        if self.layer_agg in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
        sel = [min(L - 1, max(0, i)) for i in sel] or [L - 1]
        return lat_LDT[sel]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        win_frames = self.N * self.f; s_frame = start_tr * self.f; e_frame = s_frame + win_frames

        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_feature_LDT(root / f"{ds}.npy")
            lat_GDT = self._maybe_pick_layers(lat_LDT)
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]; s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        return {"video": feats["video"], "text": feats["text"], "audio": feats["audio"],
                "ds": ds, "start_tr": start_tr}

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    data: Dict[str, torch.Tensor] = {}
    for k in ["video","text","audio"]: data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]; data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)

# ---------------- metrics ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

def _rank1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort"); ranks = np.empty_like(x, dtype=np.float64); sx = x[order]
    n = x.size; i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]: j += 1
        avg = (i + j - 1) / 2.0 + 1.0; ranks[order[i:j]] = avg; i = j
    return ranks

@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    N, O = pred.shape
    rp = np.empty_like(pred, dtype=np.float64); rt = np.empty_like(true, dtype=np.float64)
    for o in range(O):
        rp[:, o] = _rank1d(pred[:, o]); rt[:, o] = _rank1d(true[:,  o])
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))

@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    return (1.0 - (ss_res / ss_tot)).astype(np.float32)

# ---------------- file resolver ----------------
_task_rx = re.compile(r"(task-[A-Za-z0-9]+(?:_[^.]*)?)", re.IGNORECASE)
def fmri_canonical(root: Path, ds: str) -> Path:
    p = Path(root) / f"{ds}.npy"
    if p.exists(): return p
    m = _task_rx.search(ds)
    if m:
        key = m.group(1)
        cands = sorted(Path(root).glob(f"*_{key}.npy")) + sorted(Path(root).glob(f"*{key}.npy"))
        if cands: return cands[0]
    parts = ds.split("_", 1)
    if len(parts) == 2:
        suf = parts[1]
        cands = sorted(Path(root).glob(f"*_{suf}.npy")) + sorted(Path(root).glob(f"*{suf}.npy"))
        if cands: return cands[0]
    raise FileNotFoundError(f"GT not found for '{ds}' under '{root}'")

def load_fmri_flexible(root: Path, ds: str) -> np.ndarray:
    return np.load(fmri_canonical(root, ds))

# ---------------- evaluate helpers ----------------
@torch.no_grad()
def reconstruct_episode_subject_conditioned(
    model: FmriEncoder_MoE,
    ds: str,
    video_root: Path, text_root: Path, audio_root: Path,
    fmri_roots_by_subject: Dict[int, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    anchor_root = list(fmri_roots_by_subject.values())[0]
    ds_tmp = WindowedDataset(
        ids=[ds],
        video_root=video_root, text_root=text_root, audio_root=audio_root,
        anchor_fmri_root=anchor_root,
        layers_arg=layers, layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type=='cuda'))

    T_ds = ds_tmp._episode_len_tr[ds]; O = 1000
    preds_sum = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(4)}
    preds_cnt = np.zeros((T_ds,), dtype=np.int32)

    model.eval()
    for batch in loader:
        batch = batch.to(device); st = int(batch.data["start_tr_list"][0])
        outs = {}
        for s in range(4):
            batch.data["subject_id"] = torch.full((1,), s, dtype=torch.long, device=device)
            y = model.forward(batch, pool_outputs=True)    # [B,O,N]
            outs[s] = y[0].permute(1, 0).detach().cpu().numpy()   # [N,O]
        N = list(outs.values())[0].shape[0]
        ed = min(st + N, T_ds); span = ed - st
        for s in range(4): preds_sum[s][st:ed] += outs[s][:span]
        preds_cnt[st:ed] += 1

    cnt = np.maximum(preds_cnt[:, None], 1)
    preds_by_sub = {s: (preds_sum[s] / cnt).astype(np.float32) for s in range(4)}

    gts_by_sub, available_subjects = {}, []
    for s, root in fmri_roots_by_subject.items():
        try:
            gt = load_fmri_flexible(root, ds)
            if 1000 in gt.shape: gt = gt if gt.shape[0] == 1000 else gt.T
            else: gt = gt.T if gt.shape[0] > gt.shape[1] else gt
            gts_by_sub[s] = gt[:, :T_ds].T.astype(np.float32)
            available_subjects.append(s)
        except FileNotFoundError:
            continue
    return preds_by_sub, gts_by_sub, available_subjects

@torch.no_grad()
def evaluate_episodes(
    model: FmriEncoder_MoE,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[int, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None, save_split_name: str = "val",
):
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(4)}
    agg_isg = {s: [] for s in range(4)}
    used_counts = {s: 0 for s in range(4)}
    per_episode_scores = {s: [] for s in range(4)}

    for ds in episodes:
        preds_by_sub, gts_by_sub, available = reconstruct_episode_subject_conditioned(
            model, ds,
            roots_feat["video"], roots_feat["text"], roots_feat["audio"],
            fmri_roots_by_subject,
            layers, layer_agg, window_tr, stride_tr, frames_per_tr, device
        )
        if not available: continue

        for s in available:
            pred, gt = preds_by_sub[s], gts_by_sub[s]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r); agg[s]["rho"].append(rho); agg[s]["r2"].append(r2)
            used_counts[s] += 1; per_episode_scores[s].append((ds, r))

        # ISG
        for s in available:
            r_list = []
            for t in available:
                if t == s: continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gts_by_sub[s]))))
            if r_list: agg_isg[s].append(float(np.mean(r_list)))

        if save_root is not None:
            subname = {0:"sub01", 1:"sub02", 2:"sub03", 3:"sub05"}
            for s in available:
                pdir = save_root / subname[s] / f"preds_{save_split_name}_episodes"
                gdir = save_root / subname[s] / f"preds_{save_split_name}_episodes_gt"
                pdir.mkdir(parents=True, exist_ok=True); gdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / f"{ds}_pred.npy", preds_by_sub[s])
                np.save(gdir / f"{ds}_gt.npy",   gts_by_sub[s])

    per_sub_means, isg_means = {}, {}
    for s in range(4):
        if used_counts[s] > 0:
            per_sub_means[s] = {"r": float(np.mean(agg[s]["r"])),
                                "rho": float(np.mean(agg[s]["rho"])),
                                "r2": float(np.mean(agg[s]["r2"]))}
            isg_means[s] = float(np.mean(agg_isg[s])) if agg_isg[s] else float("nan")
        else:
            per_sub_means[s] = {"r": float("nan"), "rho": float("nan"), "r2": float("nan")}
            isg_means[s] = float("nan")
    return per_sub_means, isg_means, used_counts, per_episode_scores

def pick_friends_episode(ids: List[str]) -> str:
    fs = [ds for ds in ids if "friends" in ds.lower()]
    return fs[0] if fs else ids[0]

# ================== 导出：friends_s01e02a 的 [1000,K] “权重均值” + 专家索引 ==================
@torch.no_grad()
def export_episode_voxel_topk_weight_means(
    model: FmriEncoder_MoE,
    episode: str,
    roots_feat: Dict[str, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
    subject_id: int,
    K: int,
    out_dir: Path
):
    """
    过程：
      1) 跑一遍 episode，得到每个窗口的  weights_final [1,N,E]、experts_out [1,N,E,O]、weights_pre [1,N,E]。
      2) 用贡献值（contrib = experts_out * weights_final）在“token维度求平均”后，按每个输出 o 选 Top-K 专家 → idx_topk[o,k]。
      3) 对这些专家，取“Top-K 之前的权重 weights_pre”在所有 token 上的均值 → 得到 [E]，再按 idx_topk 收集成 [O,K]。
    输出：
      weights_mean_[1000,K].npy （float32）
      experts_idx_[1000,K].npy  （int64）
    """
    video_root, text_root, audio_root = roots_feat["video"], roots_feat["text"], roots_feat["audio"]
    anchor_fmri_root = roots_feat["anchor_fmri_root"]

    ds_tmp = WindowedDataset(
        ids=[episode],
        video_root=video_root, text_root=text_root, audio_root=audio_root,
        anchor_fmri_root=anchor_fmri_root,
        layers_arg=layers, layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type=='cuda'))

    E = model.num_experts; O = model.n_outputs
    # 累加 token 维度的贡献（用于选 Top-K）
    contrib_sum_eo = torch.zeros(E, O, dtype=torch.float64)
    total_tokens = 0

    # 累加 “Top-K 之前”的权重（用于取均值）
    weights_pre_token_sum_e = torch.zeros(E, dtype=torch.float64)

    for batch in loader:
        batch = batch.to(device)
        batch.data["subject_id"] = torch.full((1,), subject_id, dtype=torch.long, device=device)
        # 取细节
        _, w_final_BNE, out_BNEO, w_pre_BNE = model.forward_with_details(batch, pool_outputs=True)  # [1,N,E], [1,N,E,O], [1,N,E]
        # 贡献：按 token 求和（或均值），用于挑 Top-K
        contrib = (out_BNEO * w_final_BNE.unsqueeze(-1)).sum(dim=1).squeeze(0).double().cpu()  # [E,O]
        contrib_sum_eo += contrib

        # 累加 Top-K 之前的权重（只在 token 上求和）
        weights_pre_token_sum_e += w_pre_BNE.sum(dim=1).squeeze(0).double().cpu()  # [E]
        total_tokens += w_pre_BNE.size(1)

    # 平均贡献（用于排序）；用绝对值避免正负抵消
    contrib_mean_eo = (contrib_sum_eo / max(1, total_tokens)).abs()  # [E,O]

    # 每个输出 o 选贡献最大的 Top-K 专家
    topk_vals, topk_idx = torch.topk(contrib_mean_eo.transpose(0,1), k=K, dim=1)  # [O,K] both
    experts_idx_OK = topk_idx.to(torch.int64).cpu().numpy()                        # [O,K]

    # 计算“Top-K 之前”的权重在 token 上的均值（对 episode 全部窗口）
    weights_pre_mean_e = (weights_pre_token_sum_e / max(1, total_tokens)).to(torch.float32)  # [E]

    # 按每个输出 o 的专家索引，收集对应的权重均值 → [O,K]
    weights_mean_OK = weights_pre_mean_e[torch.as_tensor(experts_idx_OK)].cpu().numpy().astype("float32")  # [O,K]

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{episode}_sub{subject_id+1:02d}_weights_mean_OK.npy", weights_mean_OK)
    np.save(out_dir / f"{episode}_sub{subject_id+1:02d}_experts_idx_OK.npy", experts_idx_OK)
    print(f"[EXPORT][{episode}] saved: {out_dir / (episode + f'_sub{subject_id+1:02d}_weights_mean_OK.npy')}")
    print(f"[EXPORT][{episode}] saved: {out_dir / (episode + f'_sub{subject_id+1:02d}_experts_idx_OK.npy')}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # splits
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--split_seed",  type=int,   default=33)
    ap.add_argument("--moe_dropout", type=float, default=0.1)

    # feature roots
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)

    # fMRI roots (multi-subject)
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # layer selection
    ap.add_argument("--layers", type=str, default="last41")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # windows
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)

    # optimization
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # model (MoE)
    ap.add_argument("--subject_embedding", action="store_true")
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_aux_weight", type=float, default=0.0)
    ap.add_argument("--moe_combine_mode", type=str, default="router_x_learned",
                    choices=["router","learned","router_x_learned"])
    ap.add_argument("--moe_subject_expert_bias", action="store_true")

    # grad ckpt / SWA
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--disable_swa", action="store_true")

    # output & logs
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "MoE_IMAGEBIND"))
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "MoE_IMAGEBIND"))

    # 导出 friends_s01e02a 的 [1000,K]
    ap.add_argument("--export_voxel_topk_episode", type=str, default="friends_s01e02a")
    ap.add_argument("--export_voxel_topk_subject", type=int, default=0)  # 0→sub01
    ap.add_argument("--export_voxel_topk_k", type=int, default=None)     # 不填则用 moe_top_k

    # misc
    ap.add_argument("--seed", type=int, default=33)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] Using {device}")

    out_dir = Path(args.out_dir); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "experts").mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    tb_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] Logging to: {tb_dir}")

    fmri_roots = {
        0: Path(args.fmri_root_sub1),
        1: Path(args.fmri_root_sub2),
        2: Path(args.fmri_root_sub3),
        3: Path(args.fmri_root_sub5),
    }

    # 数据集拆分
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.seed); rnd.shuffle(all_ids)
        k = max(1, min(len(all_ids)-1, int(round(len(all_ids) * args.split_ratio))))
        train_ids, val_ids = all_ids[:k], all_ids[k:]
        print(f"[SPLIT] Using --all_list, split to train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide --all_list or both --train_list/--val_list")
        train_ids = read_ids(args.train_list); val_ids = read_ids(args.val_list)
        print(f"[SPLIT] Using provided lists: train={len(train_ids)} val={len(val_ids)}")

    layer_agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    train_set = WindowedDataset(train_ids, Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                                fmri_roots[0], args.layers, layer_agg,
                                args.window_tr, args.stride_tr, args.frames_per_tr)
    val_set_for_loss = WindowedDataset(val_ids if len(val_ids)>0 else train_ids[:1],
                                       Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                                       fmri_roots[0], args.layers, layer_agg,
                                       args.window_tr, args.stride_tr, args.frames_per_tr)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader_for_loss = DataLoader(val_set_for_loss, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    feat_dims = {"video": (train_set.G, train_set.Dv), "text": (train_set.G, train_set.Dt), "audio": (train_set.G, train_set.Da)}
    model = FmriEncoder_MoE(feature_dims=feat_dims, n_outputs=1000, n_output_timesteps=args.window_tr,
                            n_subjects=4, num_experts=args.moe_num_experts, top_k=args.moe_top_k,
                            feature_aggregation="cat", layer_aggregation="cat",
                            subject_embedding=args.subject_embedding, moe_dropout=args.moe_dropout,
                            combine_mode=args.moe_combine_mode, subject_expert_bias=args.moe_subject_expert_bias).to(device)
    print(f"[MODEL] experts={args.moe_num_experts} topk={args.moe_top_k} combine_mode={args.moe_combine_mode} "
          f"subj_embed={args.subject_embedding} subj_bias={args.moe_subject_expert_bias}")

    # grad ckpt（可选）
    if args.grad_ckpt:
        try:
            import torch.utils.checkpoint as ckpt
            if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and isinstance(model.encoder.layers, torch.nn.ModuleList):
                for blk in model.encoder.layers:
                    fwd = blk.forward
                    def wrapper(*x, _f=fwd, **kw):
                        return ckpt.checkpoint(_f, *x, use_reentrant=False, **kw)
                    blk.forward = wrapper
                print("[CKPT] Enabled on encoder layers.")
            else:
                print("[CKPT][WARN] encoder.layers not found; skip.")
        except Exception as e:
            print(f"[CKPT][WARN] enabling failed: {e}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps_per_epoch*args.epochs,
                           pct_start=args.warmup_pct, anneal_strategy="cos")

    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = (not args.disable_swa) and (swa_start_epoch < args.epochs)
    swa_model = AveragedModel(model) if use_swa else None

    num_E = args.moe_num_experts
    expert_weight_sum = np.zeros((4, num_E), dtype=np.float64)
    expert_weight_cnt = np.zeros((4,), dtype=np.int64)
    experts_dir = out_dir / "experts"; experts_dir.mkdir(parents=True, exist_ok=True)

    best_key = float("-inf")
    fmri_cache: Dict[Tuple[int,str], np.ndarray] = {}
    global_step = 0

    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # --------------------- Train Loop ---------------------
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            loss_terms = []; aux_terms = []
            for s in range(4):
                batch.data["subject_id"] = torch.full((batch.data["video"].size(0),), s, dtype=torch.long, device=device)
                y = model.forward(batch, pool_outputs=True)         # [B,O,N]

                # >>> 改成统计 Top-K 之前（weights_pre）的平均 <<<
                w_pre_avg = model.get_last_weight_pre_avg()
                if w_pre_avg is not None:
                    expert_weight_sum[s] += w_pre_avg.numpy(); expert_weight_cnt[s] += 1

                B, O, N = y.shape
                ds_list = batch.data["ds_list"]; st_list = batch.data["start_tr_list"]
                for i in range(B):
                    ds = ds_list[i]; st = int(st_list[i]); ed = st + N
                    try:
                        key = (s, ds)
                        if key not in fmri_cache:
                            gt_all = load_fmri_flexible(fmri_roots[s], ds)
                            if 1000 in gt_all.shape: gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
                            else: gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
                            fmri_cache[key] = gt_all
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed: continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)  # [O,N]
                        loss_terms.append(criterion(y[i], gt_win))
                    except FileNotFoundError:
                        continue

                if args.moe_aux_weight > 0 and getattr(model, "last_aux_loss", None) is not None:
                    aux_terms.append(model.last_aux_loss)

            if not loss_terms: continue
            loss = torch.stack(loss_terms).mean()
            if aux_terms: loss = loss + float(args.moe_aux_weight) * torch.stack(aux_terms).mean()

            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", float(loss.item()), global_step); global_step += 1

            if use_swa and epoch >= swa_start_epoch: swa_model.update_parameters(model)

        train_loss = running / max(1, len(train_loader))
        writer.add_scalar("loss/train_epoch", float(train_loss), epoch)

        # -------- Val (window loss, anchor=subject 0) --------
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_for_loss:
                batch = batch.to(device)
                batch.data["subject_id"] = torch.zeros((batch.data["video"].size(0),), dtype=torch.long, device=device)
                y = model.forward(batch, pool_outputs=True)  # [B,O,N]
                B, O, N = y.shape
                for i in range(B):
                    ds = batch.data["ds_list"][i]; st = int(batch.data["start_tr_list"][i]); ed = st + N
                    try:
                        key = (0, ds)
                        if key not in fmri_cache:
                            gt_all = load_fmri_flexible(fmri_roots[0], ds)
                            if 1000 in gt_all.shape: gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
                            else: gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
                            fmri_cache[key] = gt_all
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed: continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)
                        val_loss += nn.MSELoss()(y[i], gt_win).item()
                    except FileNotFoundError:
                        continue
        val_loss /= max(1, len(val_loader_for_loss))
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # -------- Evaluate on full validation episodes --------
        roots_feat = {"video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root)}
        per_sub_means, isg_means, used_counts, per_episode_scores = evaluate_episodes(
            model=model, episodes=val_ids, roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=layer_agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )

        # Train-probe（Friends）
        probe_means, probe_isg, probe_used, _ = evaluate_episodes(
            model=model, episodes=[train_probe_ds], roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=layer_agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="trainprobe"
        )

        # -------- Print & TB --------
        acc_key = []
        log_line = [f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  |  VAL(all episodes)"]
        for s in range(4):
            r = per_sub_means.get(s, {}).get("r", float("nan"))
            rho = per_sub_means.get(s, {}).get("rho", float("nan"))
            r2 = per_sub_means.get(s, {}).get("r2", float("nan"))
            isg = isg_means.get(s, float("nan"))
            n_used = used_counts.get(s, 0)
            if not np.isnan(r): acc_key.append(r)
            writer.add_scalar(f"val/sub{s+1:02d}_pearson_mean",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_spearman_mean", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_r2_mean",       0.0 if np.isnan(r2) else r2,  epoch)
            if not np.isnan(isg): writer.add_scalar(f"val/sub{s+1:02d}_ISG_pearson", isg, epoch)
            log_line.append(f"S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")
        val_key = float(np.mean(acc_key)) if acc_key else float("-inf")
        writer.add_scalar("val/mean_pearson", 0.0 if np.isnan(val_key) else val_key, epoch)

        log_line.append(" | TRAIN-PROBE(Friends)")
        for s in range(4):
            r = probe_means.get(s, {}).get("r", float("nan"))
            rho = probe_means.get(s, {}).get("rho", float("nan"))
            r2 = probe_means.get(s, {}).get("r2", float("nan"))
            isg = probe_isg.get(s, float("nan"))
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_pearson",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_spearman", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_r2",       0.0 if np.isnan(r2) else r2,  epoch)
            log_line.append(f"S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}")
        print("  ".join(log_line))

        # ===== 专家权重统计输出（控制台 / TensorBoard / 文件）=====
        txt_path = experts_dir / f"epoch_{epoch:03d}.txt"
        with open(txt_path, "w", encoding="utf-8") as ftxt:
            for s in range(4):
                if expert_weight_cnt[s] > 0:
                    w_epoch = (expert_weight_sum[s] / max(1, expert_weight_cnt[s]))  # [E]
                    for e in range(num_E):
                        writer.add_scalar(f"experts/epoch_avg/subject{s+1:02d}/E{e}", float(w_epoch[e]), epoch)
                    order = np.argsort(-w_epoch); pairs = [f"E{int(e)}={w_epoch[e]:.3f}" for e in order]
                    print(f"[EXPERT][Epoch {epoch}] Subject S{s+1:02d} expert weights: " + ", ".join(pairs))
                    ftxt.write(f"Subject S{s+1:02d} (avg over batches & windows)\n")
                    ftxt.write("raw: " + ", ".join([f"E{e}={w_epoch[e]:.6f}" for e in range(num_E)]) + "\n")
                    ftxt.write("sorted: " + ", ".join(pairs) + "\n\n")
                else:
                    print(f"[EXPERT][Epoch {epoch}] Subject S{s+1:02d} no gating stats collected this epoch.")
                    ftxt.write(f"Subject S{s+1:02d}: no gating stats collected.\n\n")
        expert_weight_sum[:] = 0.0; expert_weight_cnt[:] = 0

        if val_key > best_key:
            best_key = val_key
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            try: torch.save(model, out_dir / "checkpoints" / "best_full.pt")
            except Exception as e: print(f"[SAVE][WARN] torch.save(model) failed: {e}")

    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best VAL mean Pearson: {best_key:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")

    # ====== 训练结束后：导出 friends_s01e02a 的 [1000, K] ======
    try:
        best_path = out_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device))
            model.to(device).eval()
            print(f"[EXPORT] Loaded best checkpoint: {best_path}")
    except Exception as e:
        print(f"[EXPORT][WARN] failed to load best.pt: {e}")

    roots_feat = {
        "video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root),
        "anchor_fmri_root": Path(args.fmri_root_sub1)  # 仅用于窗口切分
    }
    episode = args.export_voxel_topk_episode or "friends_s01e02a"
    subject_id = int(args.export_voxel_topk_subject)
    K = int(args.export_voxel_topk_k) if args.export_voxel_topk_k is not None else int(args.moe_top_k)

    export_dir = out_dir / "experts" / "episode_voxel_topk"
    export_episode_voxel_topk_weight_means(
        model=model, episode=episode, roots_feat=roots_feat,
        layers=args.layers, layer_agg=layer_agg,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
        device=device, subject_id=subject_id, K=K, out_dir=export_dir
    )
    print(f"Export dir: {export_dir}")

if __name__ == "__main__":
    main()
