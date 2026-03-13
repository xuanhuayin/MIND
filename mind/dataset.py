# -*- coding: utf-8 -*-
"""
Windowed dataset for training / evaluation.

Each sample is a fixed-length window of 2 Hz multimodal features
(video, text, audio) aligned to fMRI TRs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import (
    group_mean_layers,
    load_fmri_flexible,
    orient_fmri,
    parse_layers_arg,
)


# --------------------------------------------------------------------------- #
class Batch:
    """Thin wrapper around a dict so the model can do ``batch.data[key]``."""

    def __init__(self, data: Dict[str, object]):
        self.data = data

    def to(self, device: torch.device):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


# --------------------------------------------------------------------------- #
class WindowedDataset(Dataset):
    """
    Yields fixed-length windows of multimodal features.

    Handles both ``[T, L, D]`` (multi-layer) and ``[T, D]`` (single-layer)
    feature files on disk.
    """

    def __init__(
        self,
        ids: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        anchor_fmri_root: Path,
        layers_arg: str,
        layer_agg: str,
        window_tr: int,
        stride_tr: int,
        frames_per_tr: int,
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.anchor_fmri_root = Path(anchor_fmri_root)
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)

        # probe layer count from the first video file
        v0_LDT = self._load_feature_LDT(self.video_root / f"{ids[0]}.npy")
        probe_L = v0_LDT.shape[0]

        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs: list[float] | None = [float(x) for x in payload]
            self.sel_indices: list[int] | None = None
        else:
            self.fracs = None
            self.sel_indices = [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        # build window index
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v = np.load(self.video_root / f"{ds}.npy")
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f
            arr = load_fmri_flexible(self.anchor_fmri_root, ds)
            fmri = orient_fmri(arr)
            T_tr = min(T_tr_feat, fmri.shape[1])
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr:
                    self._index.append((ds, st))

        # resolve G (groups) and per-modality dims from first sample
        first_ds = self._index[0][0]
        v_GDT = self._pick(self._load_feature_LDT(self.video_root / f"{first_ds}.npy"))
        t_GDT = self._pick(self._load_feature_LDT(self.text_root / f"{first_ds}.npy"))
        a_GDT = self._pick(self._load_feature_LDT(self.audio_root / f"{first_ds}.npy"))
        self.G = v_GDT.shape[0]
        self.Dv = v_GDT.shape[1]
        self.Dt = t_GDT.shape[1]
        self.Da = a_GDT.shape[1]

    def __len__(self) -> int:
        return len(self._index)

    # ----- feature loading helpers -----
    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        """Load feature file and return [L, D, T]."""
        arr = np.load(path_npy)
        if arr.ndim == 3:
            return np.transpose(arr, (1, 2, 0))        # [T,L,D] -> [L,D,T]
        elif arr.ndim == 2:
            return np.transpose(arr[:, np.newaxis, :], (1, 2, 0))  # [T,D] -> [1,D,T]
        else:
            raise ValueError(f"Expect [T,L,D] or [T,D], got {arr.shape}: {path_npy}")

    def _pick(self, lat_LDT: np.ndarray) -> np.ndarray:
        """Select / group layers according to config."""
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L] or [L - 1]
            return lat_LDT[sel]
        if self.layer_agg in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
        sel = [min(L - 1, max(0, i)) for i in sel] or [L - 1]
        return lat_LDT[sel]

    def _align_G(self, x_GDT: np.ndarray) -> np.ndarray:
        """Repeat or truncate along G to match self.G."""
        G = x_GDT.shape[0]
        if G == self.G:
            return x_GDT
        if G == 1 and self.G > 1:
            return np.repeat(x_GDT, repeats=self.G, axis=0)
        return x_GDT[: self.G]

    def __getitem__(self, i: int) -> Dict[str, object]:
        ds, start_tr = self._index[i]
        win_frames = self.N * self.f
        s_frame = start_tr * self.f
        e_frame = s_frame + win_frames

        feats = {}
        for name, root in (
            ("video", self.video_root),
            ("text", self.text_root),
            ("audio", self.audio_root),
        ):
            lat_LDT = self._load_feature_LDT(root / f"{ds}.npy")
            lat_GDT = self._align_G(self._pick(lat_LDT))

            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            feats[name] = torch.from_numpy(lat_GDT[..., s_frame:e_frame].astype(np.float32))

        return {
            "video": feats["video"],
            "text": feats["text"],
            "audio": feats["audio"],
            "ds": ds,
            "start_tr": start_tr,
        }


# --------------------------------------------------------------------------- #
def collate_fn(batch: List[Dict]) -> Batch:
    data: Dict[str, object] = {}
    for k in ("video", "text", "audio"):
        data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)
