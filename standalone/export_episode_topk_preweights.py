# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---- repo root (按需修改或保持与训练一致) ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# 直接复用你训练用的 MoE
from algonauts2025.standalone.weighted_moe_decoder import FmriEncoder_MoE

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# --------- 数据集（只支持单 episode 的 window 化）---------
def _load_feature_LDT(path_npy: Path) -> np.ndarray:
    arr = np.load(path_npy)
    if arr.ndim != 3:
        raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
    # 返回 [L,D,T]
    return np.transpose(arr, (1, 2, 0))

def _group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions)) or [L - 1]
    if idxs[-1] != L - 1: idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]
    starts = [0] + bounds[:-1]
    groups = []
    for s, e in zip(starts, bounds):
        s = max(0, min(s, L)); e = max(0, min(e, L))
        if e <= s: s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(groups, axis=0)

def _parse_layers_arg(layers_arg: str, probe_L: int):
    s = (layers_arg or "").strip().lower()
    if not s:
        return "indices", [probe_L - 1]
    if s == "all":
        return "indices", list(range(probe_L))
    if s.startswith("last"):
        try: k = int(s.replace("last", ""))
        except: k = 1
        k = max(1, min(k, probe_L))
        return "indices", list(range(max(0, probe_L - k), probe_L))
    if s.startswith("idx:"):
        idxs = []
        for p in [p for p in s[4:].split(",") if p.strip()]:
            try:
                i = int(p)
                if 0 <= i < probe_L: idxs.append(i)
            except: pass
        return "indices", sorted(set(idxs or [probe_L - 1]))
    try:
        fracs = [min(1.0, max(0.0, float(x))) for x in s.split(",") if x.strip() != ""]
        return "fractions", (fracs or [1.0])
    except:
        return "indices", [probe_L - 1]

class _EpisodeWindows(Dataset):
    def __init__(self, episode_id: str, video_root: Path, text_root: Path, audio_root: Path,
                 layers_arg: str, layer_agg: str, window_tr: int, stride_tr: int, frames_per_tr: int):
        self.ds = episode_id
        self.video_root, self.text_root, self.audio_root = map(Path, (video_root, text_root, audio_root))
        self.N = int(window_tr); self.S = int(stride_tr); self.f = int(frames_per_tr)

        # 读取层数
        v0 = np.load(self.video_root / f"{self.ds}.npy")
        probe_L = v0.shape[1]
        self.layer_mode, payload = _parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs, self.sel_indices = [float(x) for x in payload], None
        else:
            self.fracs, self.sel_indices = None, [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        # 序列长度（按特征确定）
        T_frames = v0.shape[0]
        self.T_tr = T_frames // self.f

        # 切窗口
        self.index: List[int] = []
        for st in range(0, max(1, self.T_tr - self.N + 1), self.S):
            if st + self.N <= self.T_tr:
                self.index.append(st)

        # 维度信息
        v_LDT = self._load_LDT(self.video_root / f"{self.ds}.npy")
        t_LDT = self._load_LDT(self.text_root  / f"{self.ds}.npy")
        a_LDT = self._load_LDT(self.audio_root / f"{self.ds}.npy")
        v_GDT = self._pick_layers(v_LDT); t_GDT = self._pick_layers(t_LDT); a_GDT = self._pick_layers(a_LDT)
        self.G, self.Dv = v_GDT.shape[0], v_GDT.shape[1]
        self.Dt, self.Da = t_GDT.shape[1], a_GDT.shape[1]

    def __len__(self): return len(self.index)

    def _load_LDT(self, path: Path) -> np.ndarray: return _load_feature_LDT(path)

    def _pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L] or [L - 1]
            return lat_LDT[sel]
        if self.layer_agg in ("group_mean", "groupmean"):
            return _group_mean_layers(lat_LDT, self.fracs)
        sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
        sel = [min(L - 1, max(0, i)) for i in sel] or [L - 1]
        return lat_LDT[sel]

    def __getitem__(self, idx: int):
        start_tr = self.index[idx]
        s_frame = start_tr * self.f
        e_frame = s_frame + self.N * self.f
        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_LDT(root / f"{self.ds}.npy")
            lat_GDT = self._pick_layers(lat_LDT)
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]; s_frame = e_frame - self.N * self.f
            lat = lat_GDT[..., s_frame:e_frame]
            feats[name] = torch.from_numpy(lat.astype(np.float32))
        return {"video": feats["video"], "text": feats["text"], "audio": feats["audio"],
                "ds": self.ds, "start_tr": int(start_tr)}

def _collate(batch):
    out = {}
    for k in ["video","text","audio"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    out["ds_list"] = [b["ds"] for b in batch]
    out["start_tr_list"] = [b["start_tr"] for b in batch]
    class _B:
        def __init__(self, d): self.data=d
        def to(self, device):
            for kk,v in self.data.items():
                if torch.is_tensor(v): self.data[kk]=v.to(device, non_blocking=True)
            return self
    return _B(out)

@torch.no_grad()
def export_episode_all_experts_preweight_OE(
    model: FmriEncoder_MoE,
    episode: str,
    video_root: Path, text_root: Path, audio_root: Path,
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device, subject_id: int, out_dir: Path
):
    """
    方案A：导出 Top-K 之前的平均权重，包含全部专家。
    输出：
      weights_mean_OE.npy: [O, E] = [1000, num_experts]，每行相同（因为权重与输出通道无关）
      experts_idx_OE.npy:  [O, E] = 对应专家索引 0..E-1（每行相同）
    """
    ds_tmp = _EpisodeWindows(
        episode, Path(video_root), Path(text_root), Path(audio_root),
        layers, layer_agg, window_tr, stride_tr, frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=_collate, pin_memory=(device.type=='cuda'))

    E = model.num_experts
    O = model.n_outputs

    # 累加 Top-K 之前的权重（仅 token 维）
    weights_pre_token_sum_e = torch.zeros(E, dtype=torch.float64)
    total_tokens = 0

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        batch.data["subject_id"] = torch.full((1,), subject_id, dtype=torch.long, device=device)
        # 取细节：我们只用 w_pre（Top-K 之前的权重）
        # forward_with_details: returns (y, w_final, experts_out, w_pre)
        _, _, _, w_pre_BNE = model.forward_with_details(batch, pool_outputs=True)  # [1,N,E]
        weights_pre_token_sum_e += w_pre_BNE.sum(dim=1).squeeze(0).double().cpu()  # [E]
        total_tokens += w_pre_BNE.size(1)

    # token 上的平均 → [E]
    weights_pre_mean_e = (weights_pre_token_sum_e / max(1, total_tokens)).to(torch.float32)  # [E]

    # 扩成 [O, E]（每个 voxel 行相同）
    weights_mean_OE = weights_pre_mean_e.view(1, E).repeat(O, 1).cpu().numpy().astype("float32")

    # 专家索引矩阵 [O, E]
    experts_idx_OE = np.tile(np.arange(E, dtype=np.int64), (O, 1))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{episode}_sub{subject_id+1:02d}_weights_mean_OE.npy", weights_mean_OE)
    np.save(out_dir / f"{episode}_sub{subject_id+1:02d}_experts_idx_OE.npy",  experts_idx_OE)
    print(f"[OK] saved: {out_dir / f'{episode}_sub{subject_id+1:02d}_weights_mean_OE.npy'}")
    print(f"[OK] saved: {out_dir / f'{episode}_sub{subject_id+1:02d}_experts_idx_OE.npy'}")
    print(f"[INFO] weights rows are identical by design (weights depend on tokens, not output voxels).")

def main():
    ap = argparse.ArgumentParser()
    # feature roots
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--text_root",  required=True)
    ap.add_argument("--audio_root", required=True)

    # episode/id（不带 .npy）
    ap.add_argument("--episode_id", required=True)

    # model arch（必须与训练一致）
    ap.add_argument("--moe_num_experts", type=int, required=True)
    ap.add_argument("--moe_top_k", type=int, required=True)  # 虽然本方案不用 top-k，但为保持与训练一致仍要求传入
    ap.add_argument("--moe_combine_mode", choices=["router","learned","router_x_learned"], required=True)
    ap.add_argument("--moe_subject_expert_bias", action="store_true")
    ap.add_argument("--subject_embedding", action="store_true")
    ap.add_argument("--moe_dropout", type=float, default=0.1)

    # layers & window（需与训练保持一致）
    ap.add_argument("--layers", default="0.6,0.8,1.0")
    ap.add_argument("--layer_aggregation", default="group_mean")
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)

    # ckpt / export config
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--subject_id", type=int, default=0)  # 0->sub01
    ap.add_argument("--out_dir", required=True)

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 先用 episode 的窗口数据来推断 feature 维度，搭模型
    ds_tmp = _EpisodeWindows(
        episode_id=args.episode_id,
        video_root=Path(args.video_root), text_root=Path(args.text_root), audio_root=Path(args.audio_root),
        layers_arg=args.layers, layer_agg=args.layer_aggregation,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr
    )
    feat_dims = {
        "video": (ds_tmp.G, ds_tmp.Dv),
        "text":  (ds_tmp.G, ds_tmp.Dt),
        "audio": (ds_tmp.G, ds_tmp.Da),
    }

    model = FmriEncoder_MoE(
        feature_dims=feat_dims, n_outputs=1000, n_output_timesteps=args.window_tr,
        n_subjects=4, num_experts=args.moe_num_experts, top_k=args.moe_top_k,
        feature_aggregation="cat", layer_aggregation="cat",
        subject_embedding=args.subject_embedding, moe_dropout=args.moe_dropout,
        combine_mode=args.moe_combine_mode, subject_expert_bias=args.moe_subject_expert_bias
    ).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    print(f"[LOAD] loaded checkpoint: {ckpt_path}")

    export_episode_all_experts_preweight_OE(
        model=model, episode=args.episode_id,
        video_root=Path(args.video_root), text_root=Path(args.text_root), audio_root=Path(args.audio_root),
        layers=args.layers, layer_agg=args.layer_aggregation,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
        device=device, subject_id=int(args.subject_id),
        out_dir=Path(args.out_dir)
    )

if __name__ == "__main__":
    main()
