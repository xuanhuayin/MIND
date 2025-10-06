# -*- coding: utf-8 -*-
"""
Standalone windowed training using *ImageBind* TRIBE_8features (video/text/audio) @ 2Hz for sub-01.

This version:
- Default epochs=25.
- Support --all_list to do in-script split (90% train / 10% val by default).
- Per-epoch metrics: Pearson (r), Spearman (ρ), R² (coefficient of determination).
- Model selection by VAL Pearson only.
- Visualization updates are decoupled:
    * Best VAL -> save ckpt + update ONLY best-VAL episode visualization.
    * Best TRAIN-PROBE -> update ONLY train-probe visualization.
- fMRI filename exactly matches dataset id, e.g., 'ses-001_task-bourne01.npy'
"""

from __future__ import annotations
import argparse, os, sys, re, subprocess, random
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

# ---- project root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- minimal fmri model (unchanged interface) ----
from algonauts2025.standalone.fmri_model_min import FmriEncoder, FmriEncoderConfig


# ---------------- utils ----------------
def set_seed(seed: int = 33):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]


def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """
    lat_LDT: [L, D, T]  -> group by fractional boundaries over L, output [G, D, T]
    Fractions are interpreted as right-edge indices ~ round(f * (L-1)).
    """
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions))
    if not idxs:
        idxs = [L - 1]
    if idxs[-1] != L - 1:
        idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]  # right-open
    starts = [0] + bounds[:-1]
    ends = bounds
    groups = []
    for s, e in zip(starts, ends):
        s = max(0, min(s, L))
        e = max(0, min(e, L))
        if e <= s:
            s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))  # [D, T]
    return np.stack(groups, axis=0)  # [G, D, T]


def parse_layers_arg(layers_arg: str, probe_L: int):
    """
    Parse --layers into a mode + payload.
      - "all"                 -> ("indices", [0..L-1])
      - "lastK"               -> ("indices", [L-K..L-1])
      - "idx:0,1,5"           -> ("indices", [0,1,5])
      - "0.5,0.75,1.0"        -> ("fractions", [0.5,0.75,1.0])
      - "" / invalid          -> last layer by default
    """
    s = (layers_arg or "").strip().lower()
    if not s:
        return "indices", [probe_L - 1]

    if s == "all":
        return "indices", list(range(probe_L))

    if s.startswith("last"):
        try:
            k = int(s.replace("last", ""))
        except Exception:
            k = 1
        k = max(1, min(k, probe_L))
        return "indices", list(range(max(0, probe_L - k), probe_L))

    if s.startswith("idx:"):
        parts = [p for p in s[4:].split(",") if p.strip()]
        idxs = []
        for p in parts:
            try:
                i = int(p)
                if 0 <= i < probe_L:
                    idxs.append(i)
            except Exception:
                pass
        if not idxs:
            idxs = [probe_L - 1]
        return "indices", sorted(set(idxs))

    # fractions
    try:
        fracs = [float(x) for x in s.split(",") if x.strip() != ""]
        fracs = [min(1.0, max(0.0, f)) for f in fracs]
        if not fracs:
            fracs = [1.0]
        return "fractions", fracs
    except Exception:
        return "indices", [probe_L - 1]


# ---------------- tiny batch container ----------------
class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


# ---------------- dataset (windowed) ----------------
class WindowedDataset(Dataset):
    """
    Sample = one window (N TRs) from an episode:
      - Inputs (video/text/audio): [G, D, N * frames_per_tr]  # frame-level
      - Target (fmri):             [1000, N]                  # TR-level
    """
    def __init__(
        self,
        ids: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        fmri_root: Path,
        fractions: List[float],
        layer_agg: str,
        window_tr: int,
        stride_tr: int,
        frames_per_tr: int,
        layers_arg: str = "",
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_root = Path(fmri_root)
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)

        # probe L to parse --layers
        probe_key = ids[0]
        v0 = np.load(self.video_root / f"{probe_key}.npy")  # [T, L, D]
        probe_L = v0.shape[1]

        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs = [float(x) for x in payload]
            self.sel_indices = None
        else:
            self.fracs = None
            self.sel_indices = [int(i) for i in payload]  # absolute indices

        self.layer_agg = layer_agg.lower()

        # build window index
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            key = ds

            v_path = self.video_root / f"{key}.npy"
            if not v_path.exists():
                raise FileNotFoundError(f"Missing video feature: {v_path}")
            v = np.load(v_path)  # [T, L, D]
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f

            fmri_path = self.fmri_root / f"{ds}.npy"
            if not fmri_path.exists():
                raise FileNotFoundError(f"Missing fmri npy: {fmri_path}")
            arr = np.load(fmri_path)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T  # [1000, T_tr]
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
                print(f"[WARN] fmri {arr.shape} no 1000-dim axis, used heuristic.")
            O, T_tr_fmri = fmri.shape
            assert O == 1000, f"Expect O=1000, got {O} for {fmri_path}"

            T_tr = min(T_tr_feat, T_tr_fmri)
            self._episode_len_tr[ds] = T_tr

            for start_tr in range(0, T_tr - self.N + 1, self.S):
                self._index.append((ds, start_tr))

        # infer G, D* for shapes (from first sample)
        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")
        v_GDT = self._maybe_pick_layers(v_LDT)
        self.G = v_GDT.shape[0]
        self.Dv = v_GDT.shape[1]
        t_LDT = self._load_feature_LDT(self.text_root / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")
        self.Dt = self._maybe_pick_layers(t_LDT).shape[1]
        self.Da = self._maybe_pick_layers(a_LDT).shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        """Convert cached [T, L, D] -> [L, D, T]"""
        arr = np.load(path_npy)
        if arr.ndim != 3:
            raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))

    def _maybe_pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        """
        Return [G, D, T] according to --layers.
        - fractions + group_mean: grouped mean over slices
        - fractions + none      : pick nearest layers by fraction
        - indices               : direct gather by absolute indices
        """
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L]
            if not sel: sel = [L - 1]
            return lat_LDT[sel]

        # fractions
        if self.layer_agg in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        else:
            sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
            sel = [min(L - 1, max(0, i)) for i in sel]
            if not sel: sel = [L - 1]
            return lat_LDT[sel]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        key = ds

        win_frames = self.N * self.f
        s_frame = start_tr * self.f
        e_frame = s_frame + win_frames

        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_feature_LDT(root / f"{key}.npy")     # [L, D, T]
            lat_GDT = self._maybe_pick_layers(lat_LDT)                # [G, D, T]
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]                       # [G, D, win_frames]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        fmri_path = self.fmri_root / f"{ds}.npy"
        arr = np.load(fmri_path)
        if 1000 in arr.shape:
            fmri = arr if arr.shape[0] == 1000 else arr.T  # [1000, T_tr]
        else:
            fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            print(f"[WARN] fmri {arr.shape} no 1000-dim axis, used heuristic.")
        Y = fmri[:, start_tr:start_tr + self.N]  # [1000, N]

        return {
            "video": feats["video"],
            "text" : feats["text" ],
            "audio": feats["audio"],
            "fmri" : torch.from_numpy(Y.astype(np.float32)),
            "subject_id": torch.tensor(0, dtype=torch.long),
            "ds": ds,
            "start_tr": start_tr,
        }


# ---------------- collate ----------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = ["video","text","audio","fmri","subject_id"]
    data: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k == "subject_id":
            data[k] = torch.stack([b[k] for b in batch], dim=0)  # [B]
        else:
            data[k] = torch.stack([b[k] for b in batch], dim=0)  # video/text/audio: [B,G,D,win_frames]; fmri: [B,1000,N]
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)


# ---------------- metrics ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred,true: [N_total, O] -> per-voxel Pearson r in [O]."""
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    """Average ranks (1..N) with ties handled by mean; stable mergesort keeps order of equals."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sx = x[order]
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        # average rank (1-based)
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks

@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Spearman ρ per voxel via Pearson on rank-transformed data. Shapes [N_total, O]."""
    N, O = pred.shape
    rp = np.empty_like(pred, dtype=np.float64)
    rt = np.empty_like(true, dtype=np.float64)
    # column-wise ranks
    for o in range(O):
        rp[:, o] = _rankdata_1d(pred[:, o])
        rt[:, o] = _rankdata_1d(true[:,  o])
    # convert to Pearson
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))

@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """R² per voxel. Shapes [N_total, O]."""
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    r2 = 1.0 - (ss_res / ss_tot)
    return r2.astype(np.float32)


# ---------------- helper: eval one episode (full reconstruction) ----------------
@torch.no_grad()
def eval_full_episode(
    model: nn.Module,
    ds: str,
    video_root: Path,
    text_root: Path,
    audio_root: Path,
    fmri_root: Path,
    layers_arg: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pred_full[T,1000], gt_full[T,1000]) on CPU np.float32.
    """
    dataset = WindowedDataset(
        ids=[ds],
        video_root=video_root,
        text_root=text_root,
        audio_root=audio_root,
        fmri_root=fmri_root,
        fractions=[1.0],
        layer_agg=layer_agg,
        window_tr=window_tr,
        stride_tr=stride_tr,
        frames_per_tr=frames_per_tr,
        layers_arg=layers_arg,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    T_ds = dataset._episode_len_tr[ds]
    n_outputs = 1000
    acc = np.zeros((T_ds, n_outputs), dtype=np.float32)
    cnt = np.zeros((T_ds, ), dtype=np.int32)

    # full GT
    fmri_path = fmri_root / f"{ds}.npy"
    gt_full = np.load(fmri_path)
    if 1000 in gt_full.shape:
        gt_full = gt_full if gt_full.shape[0] == 1000 else gt_full.T  # [1000,T]
    else:
        gt_full = gt_full.T if gt_full.shape[0] > gt_full.shape[1] else gt_full
    gt_full = gt_full[:, :T_ds].T.astype(np.float32)  # [T,1000]

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_pred = model(batch)   # [1,1000,N]
        start_tr = int(batch.data["start_tr_list"][0])
        N = y_pred.shape[-1]
        yp = y_pred[0].permute(1,0).detach().cpu().numpy()  # [N,1000]
        end_tr = min(start_tr + N, T_ds)
        acc[start_tr:end_tr] += yp[:end_tr-start_tr, :]
        cnt[start_tr:end_tr] += 1

    cnt = np.maximum(cnt[:,None], 1)
    pred_full = acc / cnt  # [T,1000]
    return pred_full.astype(np.float32), gt_full.astype(np.float32)


# ---------------- helper: call your visualization script ----------------
def _run_vis_cli(
    script: str,
    gt: str,
    pred: str,
    atlas: str,
    outdir: str,
    subject: str,
    modality: str,
    align: str,
    delay: int,
):
    if not script or not Path(script).exists():
        print(f"[VIS][WARN] Visualization script not found: {script}")
        return
    cmd = [
        sys.executable, str(script),
        "--gt", str(gt),
        "--pred", str(pred),
        "--atlas", str(atlas),
        "--outdir", str(outdir),
        "--subject", str(subject),
        "--modality", str(modality),
        "--align", str(align),
        "--delay", str(int(delay)),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[VIS][WARN] Visualization script failed: {e}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # splits
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="",
                    help="If provided, ignore --train_list/--val_list and do in-script split.")
    ap.add_argument("--split_ratio", type=float, default=0.9, help="Train ratio when using --all_list.")
    ap.add_argument("--split_seed",  type=int,   default=33,  help="Split seed when using --all_list.")

    # TRIBE_8features
    ap.add_argument("--video_root", type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "TRIBE_8features" / "video_2hz" / "sub-01"))
    ap.add_argument("--text_root",  type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "TRIBE_8features" / "text_2hz" / "sub-01"))
    ap.add_argument("--audio_root", type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "TRIBE_8features" / "audio_2hz" / "sub-01"))
    # fmri
    ap.add_argument("--fmri_root",  type=str, required=True)

    # layer selection / aggregation
    ap.add_argument("--layers", type=str, default="last41",
                    help="last41 | all | idx:0,1,5 | 0.5,0.75,1.0 (with --layer_aggregation group_mean)")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # windows (match paper)
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)  # 2Hz → 3 frames per TR

    # optimization
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--modality_dropout", type=float, default=0.2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # misc
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "standalone_windows_imagebind"))
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "standalone_imagebind"))

    # visualization & extra
    ap.add_argument("--vis_script", type=str,
                    default=str(PROJ / "vis" / "plot_pred_on_brain.py"))
    ap.add_argument("--atlas_path", type=str,
                    default=str(PROJ / "download" / "algonauts_2025.competitors" / "fmri" / "sub-01" / "atlas" /
                                "sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"))
    ap.add_argument("--subject_code", type=str, default="01")

    ap.add_argument("--vis_align", type=str, default="truncate", choices=["truncate","resample"])
    ap.add_argument("--vis_delay", type=int, default=0)

    # 指定用哪个 episode 做“训练集整段评估”
    ap.add_argument("--vis_train_ds_override", type=str, default="",
                    help="若为空将自动取训练集中第一个；必须是 train_list 里的一个 dataset id")
    # 指定验证集里优先可视化哪个 episode（可选）
    ap.add_argument("--vis_val_ds_override", type=str, default="",
                    help="如果提供，且该 episode 在本轮有重建结果，则优先用它做 VAL 可视化")

    ap.add_argument("--print_vis_shapes", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_windows").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_episodes_gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_train_episode").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_train_episode_gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis_val_best").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis_trainprobe_best").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # layer aggregation flag
    agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    # datasets (support --all_list split)
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.split_seed)
        rnd.shuffle(all_ids)
        k = int(round(len(all_ids) * args.split_ratio))
        k = max(1, min(len(all_ids)-1, k))
        train_ids = all_ids[:k]
        val_ids   = all_ids[k:]
        print(f"[SPLIT] from --all_list={args.all_list}: "
              f"train={len(train_ids)} ({args.split_ratio:.2f}), val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide both --train_list and --val_list, or use --all_list to split automatically.")
        train_ids = read_ids(args.train_list)
        val_ids   = read_ids(args.val_list)

    # choose default train-probe if not provided or invalid
    if not args.vis_train_ds_override or (args.vis_train_ds_override not in set(train_ids)):
        args.vis_train_ds_override = train_ids[0]
        print(f"[INFO] --vis_train_ds_override not valid/provided, use default train id: {args.vis_train_ds_override}")

    # parse fractions for dataset (not used if indices-mode)
    try:
        fractions_guess = [float(x) for x in args.layers.split(",") if x.strip()]
    except Exception:
        fractions_guess = [1.0]

    train_set = WindowedDataset(
        ids=train_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =Path(args.fmri_root),
        fractions=fractions_guess,
        layer_agg=agg,
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers,
    )
    val_set = WindowedDataset(
        ids=val_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =Path(args.fmri_root),
        fractions=fractions_guess,
        layer_agg=agg,
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False
    )

    # —— build model —— #
    G, Dv, Dt, Da = train_set.G, train_set.Dv, train_set.Dt, train_set.Da
    feat_dims = {"video": (G, Dv), "text": (G, Dt), "audio": (G, Da)}
    n_outputs = 1000  # fixed voxels

    cfg = FmriEncoderConfig(
        n_subjects=1,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=False,
        modality_dropout=args.modality_dropout,
    )
    model = FmriEncoder(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=args.window_tr,
        config=cfg,
    ).to(device)

    # optimizer / sched
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_pct, anneal_strategy="cos"
    )

    # SWA
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = swa_start_epoch < args.epochs
    swa_model = AveragedModel(model) if use_swa else None

    # ====== 独立最优指标（按 VAL Pearson 选最佳） ======
    best_val = float("-inf")
    best_trainprobe = float("-inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # --------------- Train ---------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            y_pred = model(batch)            # [B,1000,N]
            y_true = batch.data["fmri"]      # [B,1000,N]
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * y_pred.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss /= len(train_set)
        writer.add_scalar("loss/train_epoch", float(train_loss), epoch)

        # --------------- Val (reconstruct full episodes) -----------------
        model.eval()
        val_loss = 0.0
        preds_all = []
        trues_all = []

        recon_pred: Dict[str, np.ndarray] = {}
        recon_cnt : Dict[str, np.ndarray] = {}
        T_ds_map  : Dict[str, int]        = val_set._episode_len_tr.copy()

        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val  ", leave=False)
            for batch in pbar_v:
                batch = batch.to(device)
                y_pred = model(batch)            # [B,1000,N]
                y_true = batch.data["fmri"]      # [B,1000,N]
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * y_pred.size(0)

                # collect (for global metrics on windows)
                yp = y_pred.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()  # [B*N,1000]
                yt = y_true.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()  # [B*N,1000]
                preds_all.append(yp)
                trues_all.append(yt)

                # save to reconstruct episodes
                ds_list = batch.data["ds_list"]
                start_list = batch.data["start_tr_list"]
                B = y_pred.shape[0]
                for i in range(B):
                    ds = ds_list[i]
                    st = int(start_list[i])
                    npy_win = y_pred[i].permute(1,0).cpu().numpy()  # [N,1000]

                    T_ds = T_ds_map[ds]
                    if ds not in recon_pred:
                        recon_pred[ds] = np.zeros((T_ds, n_outputs), dtype=np.float32)
                        recon_cnt [ds] = np.zeros((T_ds, ), dtype=np.int32)
                    end_tr = min(st + args.window_tr, T_ds)
                    recon_pred[ds][st:end_tr] += npy_win[:end_tr-st, :]
                    recon_cnt [ds][st:end_tr] += 1

        val_loss /= len(val_set)
        preds_np = np.concatenate(preds_all, axis=0)
        trues_np = np.concatenate(trues_all, axis=0)

        pearson_vec = voxelwise_pearson(preds_np, trues_np)   # [1000]
        spearman_vec = voxelwise_spearman(preds_np, trues_np) # [1000]
        r2_vec       = voxelwise_r2(preds_np, trues_np)       # [1000]

        val_pearson_mean  = float(np.nanmean(pearson_vec))
        val_spearman_mean = float(np.nanmean(spearman_vec))
        val_r2_mean       = float(np.nanmean(r2_vec))

        # merge episodes & save GT/PRED for each
        best_ds = None
        best_len = -1
        for ds, acc in recon_pred.items():
            cnt = np.maximum(recon_cnt[ds][:,None], 1)
            merged = acc / cnt
            np.save(out_dir / "preds_val_episodes" / f"{ds}_pred.npy", merged.astype(np.float32))

            # also save GT for the same T
            gt_all = np.load(Path(args.fmri_root) / f"{ds}.npy")
            if 1000 in gt_all.shape:
                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
            else:
                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
            T = merged.shape[0]
            np.save(out_dir / "preds_val_episodes_gt" / f"{ds}_gt.npy", gt_all[:, :T].T.astype(np.float32))

            # choose the longest episode as default best for visualization (unless override available)
            if T > best_len:
                best_len = T
                best_ds = ds

        # --------------- Train-probe FULL EPISODE metric -----------------
        train_probe_ds = args.vis_train_ds_override
        pred_full_tr, gt_full_tr = eval_full_episode(
            model=model,
            ds=train_probe_ds,
            video_root=Path(args.video_root),
            text_root =Path(args.text_root),
            audio_root=Path(args.audio_root),
            fmri_root =Path(args.fmri_root),
            layers_arg=args.layers,
            layer_agg=agg,
            window_tr=args.window_tr,
            stride_tr=args.stride_tr,
            frames_per_tr=args.frames_per_tr,
            device=device,
        )
        # save for possible visualization
        np.save(out_dir / "preds_train_episode"    / f"{train_probe_ds}_pred.npy", pred_full_tr)
        np.save(out_dir / "preds_train_episode_gt" / f"{train_probe_ds}_gt.npy",  gt_full_tr)

        # train-probe metrics (full episode)
        tp_r   = voxelwise_pearson(pred_full_tr, gt_full_tr)
        tp_rho = voxelwise_spearman(pred_full_tr, gt_full_tr)
        tp_r2  = voxelwise_r2(pred_full_tr, gt_full_tr)

        trainprobe_pearson_mean  = float(np.nanmean(tp_r))
        trainprobe_spearman_mean = float(np.nanmean(tp_rho))
        trainprobe_r2_mean       = float(np.nanmean(tp_r2))

        # --------------- Logging & print -----------------
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)
        writer.add_scalar("metric/val_pearson_mean",  val_pearson_mean, epoch)
        writer.add_scalar("metric/val_spearman_mean", val_spearman_mean, epoch)
        writer.add_scalar("metric/val_r2_mean",       val_r2_mean, epoch)

        writer.add_scalar("metric/trainprobe_pearson_mean",  trainprobe_pearson_mean, epoch)
        writer.add_scalar("metric/trainprobe_spearman_mean", trainprobe_spearman_mean, epoch)
        writer.add_scalar("metric/trainprobe_r2_mean",       trainprobe_r2_mean, epoch)
        writer.flush()

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"VAL: r={val_pearson_mean:.6f}, ρ={val_spearman_mean:.6f}, R²={val_r2_mean:.6f}  "
            f"TRAIN-PROBE({train_probe_ds}): r={trainprobe_pearson_mean:.6f}, ρ={trainprobe_spearman_mean:.6f}, R²={trainprobe_r2_mean:.6f}"
        )

        # --------------- Save best & visualize (FULL EPISODES ONLY, decoupled) ---------------

        # A) VAL 指标（Pearson）创新高：更新 VAL 可视化 + 保存 checkpoint
        if val_pearson_mean > best_val:
            best_val = val_pearson_mean

            # prefer override ds if provided & available this epoch
            best_val_ds_for_vis = None
            # only use override if the ds appeared in recon this epoch
            if args.vis_val_ds_override and (args.vis_val_ds_override in recon_pred):
                best_val_ds_for_vis = args.vis_val_ds_override
            else:
                best_val_ds_for_vis = best_ds

            # 保存基于 VAL 的最佳 checkpoint
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            torch.save(model, out_dir / "checkpoints" / "best_full.pt")

            # ---- 仅更新：VAL 整段可视化 ----
            if best_val_ds_for_vis is not None:
                ds = best_val_ds_for_vis
                ep_gt_path   = out_dir / "preds_val_episodes_gt" / f"{ds}_gt.npy"
                ep_pred_path = out_dir / "preds_val_episodes"    / f"{ds}_pred.npy"

                if args.print_vis_shapes:
                    try:
                        _gt  = np.load(ep_gt_path)
                        _pr  = np.load(ep_pred_path)
                        print(f"[VIS] VAL FULL ds={ds} GT shape={_gt.shape}  PRED shape={_pr.shape}")
                    except Exception as e:
                        print(f"[VIS][WARN] 读取 VAL FULL 可视化数组失败: {e}")

                _run_vis_cli(
                    script=args.vis_script,
                    gt=str(ep_gt_path),
                    pred=str(ep_pred_path),
                    atlas=args.atlas_path,
                    outdir=str(out_dir / "vis_val_best"),
                    subject=args.subject_code,
                    modality=f"val_best_{ds}",
                    align=args.vis_align,
                    delay=args.vis_delay,
                )

        # B) TRAIN-PROBE 指标（Pearson）创新高：仅更新 TRAIN-PROBE 可视化
        if trainprobe_pearson_mean > best_trainprobe:
            best_trainprobe = trainprobe_pearson_mean

            tp_gt_path   = out_dir / "preds_train_episode_gt" / f"{train_probe_ds}_gt.npy"
            tp_pred_path = out_dir / "preds_train_episode"    / f"{train_probe_ds}_pred.npy"

            if args.print_vis_shapes:
                try:
                    _gtp = np.load(tp_gt_path)
                    _prp = np.load(tp_pred_path)
                    print(f"[VIS] TRAIN-PROBE FULL ds={train_probe_ds} GT shape={_gtp.shape}  PRED shape={_prp.shape}")
                except Exception as e:
                    print(f"[VIS][WARN] 读取 TRAIN-PROBE FULL 可视化数组失败: {e}")

            _run_vis_cli(
                script=args.vis_script,
                gt=str(tp_gt_path),
                pred=str(tp_pred_path),
                atlas=args.atlas_path,
                outdir=str(out_dir / "vis_trainprobe_best"),
                subject=args.subject_code,
                modality=f"trainprobe_{train_probe_ds}",
                align=args.vis_align,
                delay=args.vis_delay,
            )

    # SWA finalize (optional)
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best val pearson (mean over voxels): {best_val:.6f}")
    print(f"Best train-probe pearson (mean over voxels): {best_trainprobe:.6f}")
    print(f"Checkpoints dir: {out_dir / 'checkpoints'}")
    print(f"Val episodes dir: {out_dir / 'preds_val_episodes'}")
    print(f"Train-probe episode dir: {out_dir / 'preds_train_episode'}")


if __name__ == "__main__":
    main()