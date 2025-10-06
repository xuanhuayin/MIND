# -*- coding: utf-8 -*-
# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/train_moetransformer_TRIBE.py
from __future__ import annotations
import argparse, os, sys, re, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

# ---- project root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ==== 关键：导入独立 MoE 模型 ====
from algonauts2025.standalone.moe_transformer_encoder import (
    FmriEncoder_MoETransformer, FmriEncoderMoEConfig
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed(seed: int = 33):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

# ---------- feature layer helpers ----------
def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions))
    if not idxs: idxs = [L - 1]
    if idxs[-1] != L - 1: idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]
    starts = [0] + bounds[:-1]; ends = bounds
    groups = []
    for s, e in zip(starts, ends):
        s = max(0, min(s, L)); e = max(0, min(e, L))
        if e <= s: s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(groups, axis=0)

def parse_layers_arg(layers_arg: str, probe_L: int):
    s = (layers_arg or "").strip().lower()
    if not s: return ("indices", [probe_L - 1])
    if s == "all": return ("indices", list(range(probe_L)))
    if s.startswith("last"):
        try: k = int(s.replace("last",""))
        except Exception: k = 1
        k = max(1, min(k, probe_L))
        return ("indices", list(range(max(0, probe_L-k), probe_L)))
    if s.startswith("idx:"):
        parts = [p for p in s[4:].split(",") if p.strip()]
        idxs = []
        for p in parts:
            try:
                i = int(p)
                if 0 <= i < probe_L: idxs.append(i)
            except Exception: pass
        if not idxs: idxs = [probe_L - 1]
        return ("indices", sorted(set(idxs)))
    try:
        fracs = [float(x) for x in s.split(",") if x.strip()!=""]
        fracs = [min(1.0, max(0.0, f)) for f in fracs]
        if not fracs: fracs = [1.0]
        return ("fractions", fracs)
    except Exception:
        return ("indices", [probe_L - 1])

# ---------------- canonical file resolver ----------------
_task_rx = re.compile(r"(task-[A-Za-z0-9]+(?:_[^.]*)?)", re.IGNORECASE)
_ses_rx = re.compile(r"ses-(\d+)", re.IGNORECASE)
def task_key_from_name(name: str) -> Optional[str]:
    m = _task_rx.search(name); return m.group(1).lower() if m else None
def pick_maxses(paths: List[Path]) -> Optional[Path]:
    if not paths: return None
    best, best_ses = None, -1
    for p in paths:
        m = _ses_rx.search(p.name); ses = int(m.group(1)) if m else -1
        if ses > best_ses: best_ses, best = ses, p
    return best
def build_task_map(root: Path) -> Dict[str, Path]:
    root = Path(root); files = sorted(root.glob("*.npy"))
    buckets: Dict[str, List[Path]] = {}
    for p in files:
        tk = task_key_from_name(p.name)
        if tk is None: continue
        buckets.setdefault(tk, []).append(p)
    out: Dict[str, Path] = {}
    for tk, lst in buckets.items(): out[tk] = pick_maxses(lst)
    return out

# ---------------- dataset (multi-subject) ----------------
SUBS = ["sub1", "sub2", "sub3", "sub5"]

class WindowedDatasetMS(Dataset):
    def __init__(self, ids: List[str], video_root: Path, text_root: Path, audio_root: Path,
                 fmri_roots: Dict[str, Path], layers: str, layer_agg: str,
                 window_tr: int, stride_tr: int, frames_per_tr: int):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root  = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_roots = {k: Path(v) for k, v in fmri_roots.items()}
        self.N = int(window_tr); self.S = int(stride_tr); self.f = int(frames_per_tr)

        self.task_maps: Dict[str, Dict[str, Path]] = {}
        for s in SUBS:
            root = self.fmri_roots.get(s, None)
            if root is None or (not root.exists()):
                self.task_maps[s] = {}
            else:
                self.task_maps[s] = build_task_map(root)

        probe_key = ids[0]
        v0 = np.load(self.video_root / f"{probe_key}.npy")  # [T,L,D]
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers, probe_L)
        if self.layer_mode == "fractions":
            self.fracs = [float(x) for x in payload]; self.sel_indices = None
        else:
            self.fracs = None; self.sel_indices = [int(i) for i in payload]
        self.layer_agg = (layer_agg or "none").lower()

        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v_path = self.video_root / f"{ds}.npy"
            if not v_path.exists(): continue
            v = np.load(v_path)
            T_frames = v.shape[0]; T_tr_feat = T_frames // self.f

            tk = task_key_from_name(ds)
            sub_T = []
            for s in SUBS:
                p = self.task_maps.get(s, {}).get(tk, None)
                if p is None: continue
                arr = np.load(p)
                if 1000 in arr.shape: fmri = arr if arr.shape[0] == 1000 else arr.T
                else: fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
                if fmri.shape[0] != 1000: continue
                sub_T.append(fmri.shape[1])
            if not sub_T: continue
            T_tr = min(T_tr_feat, min(sub_T))
            self._episode_len_tr[ds] = T_tr
            for start_tr in range(0, T_tr - self.N + 1, self.S):
                self._index.append((ds, start_tr))

        first_ds, _ = self._index[0]
        def _pick_layers(path):
            arr = np.load(path); arr = np.transpose(arr, (1,2,0))
            if self.layer_mode == "indices":
                sel = [i for i in self.sel_indices if 0<=i<arr.shape[0]] or [arr.shape[0]-1]
                return arr[sel]
            if self.layer_agg in ("group_mean","groupmean"):
                return group_mean_layers(arr, self.fracs)
            sel = sorted(set(int(round(f*(arr.shape[0]-1))) for f in self.fracs)) or [arr.shape[0]-1]
            sel = [min(arr.shape[0]-1, max(0,i)) for i in sel]
            return arr[sel]
        v_GDT = _pick_layers(self.video_root / f"{first_ds}.npy")
        t_GDT = _pick_layers(self.text_root  / f"{first_ds}.npy")
        a_GDT = _pick_layers(self.audio_root / f"{first_ds}.npy")
        self.Gv, self.Dv = v_GDT.shape[0], v_GDT.shape[1]
        self.Gt, self.Dt = t_GDT.shape[0], t_GDT.shape[1]
        self.Ga, self.Da = a_GDT.shape[0], a_GDT.shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        arr = np.load(path_npy)
        if arr.ndim != 3: raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))  # [L,D,T]

    def _maybe_pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L] or [L - 1]
            return lat_LDT[sel]
        if self.layer_agg in ("group_mean","groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs)) or [L - 1]
        sel = [min(L - 1, max(0, i)) for i in sel]
        return lat_LDT[sel]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        win_frames = self.N * self.f
        s_frame = start_tr * self.f; e_frame = s_frame + win_frames
        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_GDT = self._maybe_pick_layers(self._load_feature_LDT(root / f"{ds}.npy"))
            if e_frame > lat_GDT.shape[-1]: e_frame = lat_GDT.shape[-1]; s_frame = e_frame - win_frames
            feats[name] = torch.from_numpy(lat_GDT[..., s_frame:e_frame].astype(np.float32))

        # fmri: [4,1000,N] + mask
        fmri_4 = np.zeros((4, 1000, self.N), dtype=np.float32)
        mask_4 = np.zeros((4,), dtype=np.float32)
        tk = task_key_from_name(ds)
        for si, s in enumerate(SUBS):
            p = build_task_map(self.fmri_roots[s]).get(tk, None)
            if p is None: continue
            arr = np.load(p)
            if 1000 in arr.shape: fmri = arr if arr.shape[0] == 1000 else arr.T
            else: fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            if fmri.shape[0] != 1000: continue
            Y = fmri[:, start_tr:start_tr + self.N]
            if Y.shape[1] == self.N:
                fmri_4[si] = Y.astype(np.float32); mask_4[si] = 1.0

        return {"video": feats["video"], "text": feats["text"], "audio": feats["audio"],
                "fmri": torch.from_numpy(fmri_4), "mask": torch.from_numpy(mask_4),
                "ds": ds, "start_tr": int(start_tr)}

def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    data = {}
    for k in ["video","text","audio"]:
        data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["fmri"]  = torch.stack([b["fmri"] for b in batch], dim=0)
    data["mask"]  = torch.stack([b["mask"] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [b["start_tr"] for b in batch]
    return Batch(data)

class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]): self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v): self.data[k] = v.to(device, non_blocking=True)
        return self

# ---------------- adapters ----------------
def compute_hidden_tr(model: nn.Module, batch: Batch) -> torch.Tensor:
    x = model.aggregate_features(batch)               # [B,T2,H]
    x = model.transformer_forward(x, subject_id=None) # [B,T2,H]
    x = x.transpose(1, 2)                             # [B,H,T2]
    x_tr = model.pooler(x).transpose(1, 2)            # [B,N,H]
    return x_tr

def predict_all_heads_from_hidden(model: nn.Module, x_tr: torch.Tensor) -> torch.Tensor:
    B, N, _ = x_tr.shape; device = x_tr.device
    ys = []
    for sid in (0,1,2,3):
        sid_vec = torch.full((B,), sid, dtype=torch.long, device=device)
        y = model.pred_head(x_tr, sid_vec)  # [B,N,O]
        ys.append(y.unsqueeze(1))
    return torch.cat(ys, dim=1)             # [B,4,N,O]

# ---------------- metrics ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True); true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0); den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)
def _rank1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort"); ranks = np.empty_like(x, dtype=np.float64); sx = x[order]; n = x.size; i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]: j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0; i = j
    return ranks
@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    N, O = pred.shape; rp = np.empty_like(pred, dtype=np.float64); rt = np.empty_like(true, dtype=np.float64)
    for o in range(O): rp[:, o] = _rank1d(pred[:, o]); rt[:, o] = _rank1d(true[:,  o])
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))
@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0); ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    return (1.0 - (ss_res / ss_tot)).astype(np.float32)
@torch.no_grad()
def compute_metrics(preds_np: np.ndarray, trues_np: np.ndarray) -> Dict[str, float]:
    if preds_np.shape[0] == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "r2": float("nan")}
    pear = voxelwise_pearson(preds_np, trues_np)
    spear = voxelwise_spearman(preds_np, trues_np)
    r2v = voxelwise_r2(preds_np, trues_np)
    return {"pearson": float(np.nanmean(pear)),
            "spearman": float(np.nanmean(spear)),
            "r2": float(np.nanmean(r2v))}

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # lists & roots
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # layers
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
    ap.add_argument("--modality_dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-2)

    # MoE args（含均衡项）
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_expert_layers", type=int, default=1, choices=[1,2])
    ap.add_argument("--moe_hidden_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.1)
    ap.add_argument("--moe_aux_weight", type=float, default=0.0)

    # misc
    ap.add_argument("--subject_embedding", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--disable_swa", action="store_true")  # 兼容保留，无实际用

    # logging / out
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "MoE"))
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "MoE"))
    ap.add_argument("--seed", type=int, default=33)

    args = ap.parse_args()
    set_seed(args.seed)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0"); pin_mem = True
        print(f"[DEV] Using CUDA device: {device}")
    else:
        device = torch.device("cpu"); pin_mem = False
        print("[DEV] Using CPU")

    # dirs
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tb_dir = Path(args.log_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] Logging to: {tb_dir}")

    # split
    if args.all_list.strip():
        all_ids = read_ids(args.all_list.strip())
        rnd = random.Random(args.seed); rnd.shuffle(all_ids)
        n_train = max(1, int(round(len(all_ids) * 0.9)))
        train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]
        print(f"[SPLIT] Using --all_list: train={len(train_ids)} val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise ValueError("Provide --all_list or both --train_list/--val_list")
        train_ids = read_ids(args.train_list); val_ids = read_ids(args.val_list)
        print(f"[SPLIT] Using provided lists: train={len(train_ids)} val={len(val_ids)}")

    # fmri roots
    fmri_roots = {
        "sub1": Path(args.fmri_root_sub1),
        "sub2": Path(args.fmri_root_sub2),
        "sub3": Path(args.fmri_root_sub3),
        "sub5": Path(args.fmri_root_sub5),
    }

    # dataset/loader
    def build_loader(ids: List[str], shuffle: bool):
        ds = WindowedDatasetMS(
            ids=ids,
            video_root=Path(args.video_root),
            text_root =Path(args.text_root),
            audio_root=Path(args.audio_root),
            fmri_roots=fmri_roots,
            layers=args.layers,
            layer_agg=args.layer_aggregation,
            window_tr=args.window_tr,
            stride_tr=args.stride_tr,
            frames_per_tr=args.frames_per_tr,
        )
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=pin_mem, drop_last=False)
        return ds, ld

    train_set, train_loader = build_loader(train_ids, shuffle=True)
    val_set,   val_loader   = build_loader(val_ids,   shuffle=False)

    # model
    feat_dims = {"video": (train_set.Gv, train_set.Dv),
                 "text" : (train_set.Gt, train_set.Dt),
                 "audio": (train_set.Ga, train_set.Da)}
    n_outputs = 1000

    cfg = FmriEncoderMoEConfig(
        n_subjects=4,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=args.subject_embedding,
        modality_dropout=args.modality_dropout,

        # transformer
        hidden=3072, transformer_depth=8, n_heads=8,
        attn_dropout=0.0, resid_dropout=0.0, layer_dropout=0.0,

        # MoE
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_expert_layers=args.moe_expert_layers,
        moe_hidden_mult=args.moe_hidden_mult,
        moe_dropout=args.moe_dropout,
    )

    model = FmriEncoder_MoETransformer(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=args.window_tr,
        config=cfg,
    ).to(device)

    # 位置编码长度对齐到 window_tr * frames_per_tr
    with torch.no_grad():
        want = args.window_tr * args.frames_per_tr
        cur  = model.time_pos_embed.shape[1]
        if cur != want:
            pos = model.time_pos_embed
            pos = torch.nn.functional.interpolate(pos.transpose(1,2), size=want, mode="linear", align_corners=False).transpose(1,2)
            model.time_pos_embed = nn.Parameter(pos)

    # grad ckpt（逐层包）
    if args.grad_ckpt:
        try:
            import torch.utils.checkpoint as ckpt
            if hasattr(model.encoder, "layers") and isinstance(model.encoder.layers, torch.nn.ModuleList):
                for blk in model.encoder.layers:
                    fwd = blk.forward
                    def wrapper(*x, _f=fwd, **kw): return ckpt.checkpoint(_f, *x, use_reentrant=False, **kw)
                    blk.forward = wrapper
                print("[CKPT] Enabled on encoder layers.")
        except Exception as e:
            print(f"[CKPT][WARN] enabling failed: {e}")

    # optim / sched
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader)); total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_pct, anneal_strategy="cos")

    best_val_mean_pearson = float("-inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            x_tr = compute_hidden_tr(model, batch)              # [B,N,H]
            y_all = predict_all_heads_from_hidden(model, x_tr)  # [B,4,N,O]
            y_all = y_all.permute(0,1,3,2)                      # [B,4,O,N]

            fmri = batch.data["fmri"]                           # [B,4,O,N]
            mask = batch.data["mask"]                           # [B,4]

            diff = y_all - fmri
            mse = (diff**2)
            mask_ = mask[:, :, None, None]
            mse = mse * mask_
            denom = (mask_.sum() * float(n_outputs) * float(args.window_tr)).clamp(min=1.0)
            loss = mse.sum() / denom

            # === MoE 均衡正则 ===
            if args.moe_aux_weight > 0:
                aux = model.moe_aux_loss()
                loss = loss + args.moe_aux_weight * aux

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_epoch += loss.item() * y_all.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            SummaryWriter.add_scalar(writer, "loss/train_step", float(loss.item()), global_step)
            if args.moe_aux_weight > 0:
                SummaryWriter.add_scalar(writer, "loss/moe_aux", float(aux.item()), global_step)
            global_step += 1

        train_loss_epoch /= max(1, len(train_set))
        SummaryWriter.add_scalar(writer, "loss/train_epoch", float(train_loss_epoch), epoch)

        # ---- Val ----
        model.eval()
        preds_cat = {s: [] for s in SUBS}; trues_cat = {s: [] for s in SUBS}
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val", leave=False)
            for batch in pbar_v:
                batch = batch.to(device)
                x_tr = compute_hidden_tr(model, batch)
                y_all = predict_all_heads_from_hidden(model, x_tr)  # [B,4,N,O]
                fmri = batch.data["fmri"]                           # [B,4,O,N]
                mask = batch.data["mask"]                           # [B,4]
                B, S, N, O = y_all.shape
                for si, s in enumerate(SUBS):
                    if mask[:,si].sum().item() == 0: continue
                    valid_b = (mask[:,si] > 0.5).nonzero(as_tuple=False).flatten()
                    if valid_b.numel() == 0: continue
                    yp = y_all[valid_b, si].reshape(-1, O).detach().cpu().numpy()
                    yt = fmri [valid_b, si].permute(0,2,1).reshape(-1, O).detach().cpu().numpy()
                    preds_cat[s].append(yp); trues_cat[s].append(yt)

        sub_ps = []
        for s in SUBS:
            if preds_cat[s]:
                preds_np = np.concatenate(preds_cat[s], axis=0)
                trues_np = np.concatenate(trues_cat[s], axis=0)
                m = compute_metrics(preds_np, trues_np)
            else:
                m = {"pearson": float("nan"), "spearman": float("nan"), "r2": float("nan")}
            SummaryWriter.add_scalar(writer, f"val/{s}_pearson",  m["pearson"],  epoch)
            SummaryWriter.add_scalar(writer, f"val/{s}_spearman", m["spearman"], epoch)
            SummaryWriter.add_scalar(writer, f"val/{s}_r2",       m["r2"],       epoch)
            if not math.isnan(m["pearson"]): sub_ps.append(m["pearson"])
        mean_val_pearson = float(np.nanmean(sub_ps)) if sub_ps else float("nan")
        SummaryWriter.add_scalar(writer, "val/mean_pearson", mean_val_pearson, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} | VAL mean Pearson={mean_val_pearson:.6f}")

        # save best
        if not math.isnan(mean_val_pearson) and mean_val_pearson > best_val_mean_pearson:
            best_val_mean_pearson = mean_val_pearson
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            try:
                torch.save(model, out_dir / "checkpoints" / "best_full.pt")
            except Exception as e:
                print(f"[SAVE][WARN] full model save failed: {e}")

        writer.flush()

    writer.close()
    print("\n[Done]")
    print(f"Best VAL mean Pearson: {best_val_mean_pearson:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    print(f"TensorBoard: {tb_dir}")

if __name__ == "__main__":
    main()