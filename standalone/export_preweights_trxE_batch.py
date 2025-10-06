# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from algonauts2025.standalone.weighted_moe_decoder import FmriEncoder_MoE

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ---------- 小数据集（单 episode 切窗口） ----------
def _load_feature_LDT(path_npy: Path) -> np.ndarray:
    arr = np.load(path_npy)
    if arr.ndim != 3: raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
    return np.transpose(arr, (1, 2, 0))  # [L,D,T]

def _group_mean_layers(lat_LDT: np.ndarray, fracs: List[float]) -> np.ndarray:
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f*(L-1))) for f in fracs)) or [L-1]
    if idxs[-1] != L-1: idxs[-1] = L-1
    bounds = [i+1 for i in idxs]; starts = [0]+bounds[:-1]
    out = []
    for s,e in zip(starts, bounds):
        s = max(0,min(L,s)); e=max(0,min(L,e))
        if e<=s: s,e = L-1,L
        out.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(out, axis=0)

def _parse_layers_arg(s: str, probe_L: int):
    s = (s or "").strip().lower()
    if not s: return "indices", [probe_L-1]
    if s == "all": return "indices", list(range(probe_L))
    if s.startswith("last"):
        try: k=int(s.replace("last",""))
        except: k=1
        k=max(1,min(k,probe_L))
        return "indices", list(range(probe_L-k, probe_L))
    try:
        fracs=[min(1.0,max(0.0,float(x))) for x in s.split(",") if x.strip()!=""]
        return "fractions", (fracs or [1.0])
    except:
        return "indices", [probe_L-1]

class _EpisodeWindows(Dataset):
    def __init__(self, episode: str, video_root: Path, text_root: Path, audio_root: Path,
                 layers_arg: str, layer_agg: str, window_tr: int, stride_tr: int, frames_per_tr: int):
        self.ds = episode
        self.video_root, self.text_root, self.audio_root = map(Path, (video_root,text_root,audio_root))
        self.N, self.S, self.f = int(window_tr), int(stride_tr), int(frames_per_tr)

        v0 = np.load(self.video_root / f"{self.ds}.npy")
        probe_L = v0.shape[1]
        self.mode, payload = _parse_layers_arg(layers_arg, probe_L)
        if self.mode == "fractions": self.fracs, self.sel = [float(x) for x in payload], None
        else: self.fracs, self.sel = None, [int(i) for i in payload]
        self.layer_agg = (layer_agg or "group_mean").lower()

        T_frames = v0.shape[0]; self.T_tr = T_frames // self.f
        self.index = [st for st in range(0, max(1, self.T_tr-self.N+1), self.S) if st+self.N <= self.T_tr]

        LDT_v = _load_feature_LDT(self.video_root / f"{self.ds}.npy")
        LDT_t = _load_feature_LDT(self.text_root  / f"{self.ds}.npy")
        LDT_a = _load_feature_LDT(self.audio_root / f"{self.ds}.npy")
        v = self._pick(LDT_v); t = self._pick(LDT_t); a = self._pick(LDT_a)
        self.G, self.Dv, self.Dt, self.Da = v.shape[0], v.shape[1], t.shape[1], a.shape[1]

    def _pick(self, LDT: np.ndarray) -> np.ndarray:
        L = LDT.shape[0]
        if self.mode == "indices":
            sel = [i for i in self.sel if 0<=i<L] or [L-1]
            return LDT[sel]
        if self.layer_agg in ("group_mean","groupmean"):
            return _group_mean_layers(LDT, self.fracs)
        sel = sorted(set(int(round(f*(L-1))) for f in self.fracs)) or [L-1]
        return LDT[sel]

    def __len__(self): return len(self.index)

    def __getitem__(self, i: int):
        st = self.index[i]; s_frame = st*self.f; e_frame = s_frame + self.N*self.f
        feats = {}
        for name, root in (("video",self.video_root),("text",self.text_root),("audio",self.audio_root)):
            LDT = _load_feature_LDT(root / f"{self.ds}.npy")
            GDT = self._pick(LDT)
            if e_frame > GDT.shape[-1]:
                e_frame = GDT.shape[-1]; s_frame = e_frame - self.N*self.f
            feats[name] = torch.from_numpy(GDT[..., s_frame:e_frame].astype(np.float32))
        return {"video":feats["video"], "text":feats["text"], "audio":feats["audio"],
                "ds": self.ds, "start_tr": int(st)}

def _collate(batch):
    out={}
    for k in ("video","text","audio"):
        out[k]=torch.stack([b[k] for b in batch], dim=0)
    out["ds_list"]=[b["ds"] for b in batch]
    out["start_tr_list"]=[b["start_tr"] for b in batch]
    class B: 
        def __init__(self,d): self.data=d
        def to(self,dev):
            for k,v in self.data.items():
                if torch.is_tensor(v): self.data[k]=v.to(dev, non_blocking=True)
            return self
    return B(out)

@torch.no_grad()
def export_trxe_for_episode(model: FmriEncoder_MoE, episode: str,
                            video_root: Path, text_root: Path, audio_root: Path,
                            layers: str, layer_agg: str, window_tr: int, stride_tr: int, frames_per_tr: int,
                            device: torch.device, subjects: List[int], out_dir: Path):
    ds = _EpisodeWindows(episode, video_root, text_root, audio_root, layers, layer_agg, window_tr, stride_tr, frames_per_tr)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate,
                        pin_memory=(device.type=='cuda'))

    E = model.num_experts
    T_total = ds.T_tr  # 理论 TR
    for sid in subjects:
        sum_TRxE = np.zeros((T_total, E), dtype=np.float64)
        hit = np.zeros((T_total,), dtype=np.int32)
        for batch in loader:
            batch = batch.to(device)
            batch.data["subject_id"] = torch.full((1,), sid, dtype=torch.long, device=device)
            _, _, _, w_pre = model.forward_with_details(batch, pool_outputs=True)  # [1,N,E]
            w_np = w_pre.squeeze(0).cpu().numpy()  # [N,E]
            st = int(batch.data["start_tr_list"][0]); ed = st + w_np.shape[0]
            sum_TRxE[st:ed] += w_np; hit[st:ed] += 1
        hit = np.maximum(hit, 1)
        TRxE = (sum_TRxE / hit[:,None]).astype(np.float32)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{episode}_sub{sid+1:02d}_preweights_TRxE.npy", TRxE)
        # 同时存每集的专家均值（用于热图）
        np.save(out_dir / f"{episode}_sub{sid+1:02d}_preweights_meanE.npy", TRxE.mean(axis=0).astype(np.float32))
        print(f"[SAVE] {episode} sub{sid+1:02d}: TRxE {TRxE.shape}, meanE {TRxE.mean(axis=0).shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_list", required=True, help="txt，每行一个 episode id（不带 .npy）")
    ap.add_argument("--video_root", required=True); ap.add_argument("--text_root", required=True); ap.add_argument("--audio_root", required=True)
    ap.add_argument("--layers", default="0.6,0.8,1.0"); ap.add_argument("--layer_aggregation", default="group_mean")
    ap.add_argument("--window_tr", type=int, default=100); ap.add_argument("--stride_tr", type=int, default=50); ap.add_argument("--frames_per_tr", type=int, default=3)
    ap.add_argument("--moe_num_experts", type=int, required=True); ap.add_argument("--moe_top_k", type=int, required=True)
    ap.add_argument("--moe_combine_mode", choices=["router","learned","router_x_learned"], required=True)
    ap.add_argument("--moe_subject_expert_bias", action="store_true"); ap.add_argument("--subject_embedding", action="store_true")
    ap.add_argument("--moe_dropout", type=float, default=0.1)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subjects", type=int, nargs="+", default=[0,1,2,3], help="0→sub01, 1→sub02, 2→sub03, 3→sub05")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = [ln.strip() for ln in open(args.episodes_list, "r", encoding="utf-8") if ln.strip()]

    # 先用第一集确定特征维度
    ds0 = _EpisodeWindows(episodes[0], Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                          args.layers, args.layer_aggregation, args.window_tr, args.stride_tr, args.frames_per_tr)
    feat_dims = {"video": (ds0.G, ds0.Dv), "text": (ds0.G, ds0.Dt), "audio": (ds0.G, ds0.Da)}
    model = FmriEncoder_MoE(feat_dims, n_outputs=1000, n_output_timesteps=args.window_tr, n_subjects=4,
                            num_experts=args.moe_num_experts, top_k=args.moe_top_k,
                            feature_aggregation="cat", layer_aggregation="cat",
                            subject_embedding=args.subject_embedding, moe_dropout=args.moe_dropout,
                            combine_mode=args.moe_combine_mode, subject_expert_bias=args.moe_subject_expert_bias).to(device)
    sd = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(sd, strict=True); model.eval()
    print(f"[LOAD] {args.checkpoint}")

    out_dir = Path(args.out_dir)
    for ep in episodes:
        export_trxe_for_episode(model, ep, Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                                args.layers, args.layer_aggregation, args.window_tr, args.stride_tr, args.frames_per_tr,
                                device, args.subjects, out_dir)

if __name__ == "__main__":
    main()