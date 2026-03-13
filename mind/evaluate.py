# -*- coding: utf-8 -*-
"""
Episode-level evaluation and expert-weight export.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Batch, WindowedDataset, collate_fn
from .metrics import voxelwise_pearson, voxelwise_r2, voxelwise_spearman
from .model import FmriEncoder_MoE
from .utils import load_fmri_flexible, orient_fmri


# --------------------------------------------------------------------------- #
#  Reconstruct a full episode with per-subject predictions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_episode(
    model: FmriEncoder_MoE,
    ds: str,
    video_root: Path,
    text_root: Path,
    audio_root: Path,
    fmri_roots_by_subject: Dict[int, Path],
    layers: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    Run sliding-window inference on one episode for all subjects.

    Returns:
        preds_by_sub  -- {subject_idx: [T, O]}
        gts_by_sub    -- {subject_idx: [T, O]}  (only for subjects with GT)
        available_subjects -- list of subject indices that have GT
    """
    anchor_root = list(fmri_roots_by_subject.values())[0]
    ds_tmp = WindowedDataset(
        ids=[ds],
        video_root=video_root,
        text_root=text_root,
        audio_root=audio_root,
        anchor_fmri_root=anchor_root,
        layers_arg=layers,
        layer_agg=layer_agg,
        window_tr=window_tr,
        stride_tr=stride_tr,
        frames_per_tr=frames_per_tr,
    )
    loader = DataLoader(
        ds_tmp, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    T_ds = ds_tmp._episode_len_tr[ds]
    O = model.n_outputs
    n_subjects = len(fmri_roots_by_subject)
    preds_sum = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(n_subjects)}
    preds_cnt = np.zeros((T_ds,), dtype=np.int32)

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        st = int(batch.data["start_tr_list"][0])
        outs = {}
        for s in range(n_subjects):
            batch.data["subject_id"] = torch.full((1,), s, dtype=torch.long, device=device)
            y = model.forward(batch, pool_outputs=True)  # [1, O, N]
            outs[s] = y[0].permute(1, 0).detach().cpu().numpy()  # [N, O]
        N = list(outs.values())[0].shape[0]
        ed = min(st + N, T_ds)
        span = ed - st
        for s in range(n_subjects):
            preds_sum[s][st:ed] += outs[s][:span]
        preds_cnt[st:ed] += 1

    cnt = np.maximum(preds_cnt[:, None], 1)
    preds_by_sub = {s: (preds_sum[s] / cnt).astype(np.float32) for s in range(n_subjects)}

    gts_by_sub: Dict[int, np.ndarray] = {}
    available_subjects: List[int] = []
    for s, root in fmri_roots_by_subject.items():
        try:
            gt = orient_fmri(load_fmri_flexible(root, ds))
            gts_by_sub[s] = gt[:, :T_ds].T.astype(np.float32)
            available_subjects.append(s)
        except FileNotFoundError:
            continue
    return preds_by_sub, gts_by_sub, available_subjects


# --------------------------------------------------------------------------- #
#  Evaluate over a list of episodes
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_episodes(
    model: FmriEncoder_MoE,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[int, Path],
    layers: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None,
    save_split_name: str = "val",
    n_subjects: int = 4,
):
    """
    Returns:
        per_sub_means   -- {s: {"r", "rho", "r2"}}
        isg_means       -- {s: float}            (Inter-Subject Generalization)
        used_counts     -- {s: int}
        per_episode_scores -- {s: [(ds, r), ...]}
    """
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(n_subjects)}
    agg_isg: Dict[int, list] = {s: [] for s in range(n_subjects)}
    used_counts = {s: 0 for s in range(n_subjects)}
    per_episode_scores: Dict[int, list] = {s: [] for s in range(n_subjects)}

    for ds in episodes:
        preds_by_sub, gts_by_sub, available = reconstruct_episode(
            model, ds,
            roots_feat["video"], roots_feat["text"], roots_feat["audio"],
            fmri_roots_by_subject,
            layers, layer_agg, window_tr, stride_tr, frames_per_tr, device,
        )
        if not available:
            continue

        for s in available:
            pred, gt = preds_by_sub[s], gts_by_sub[s]
            r = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2 = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r)
            agg[s]["rho"].append(rho)
            agg[s]["r2"].append(r2)
            used_counts[s] += 1
            per_episode_scores[s].append((ds, r))

        # ISG: use predictions of subject t to predict GT of subject s
        for s in available:
            r_list = []
            for t in available:
                if t == s:
                    continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gts_by_sub[s]))))
            if r_list:
                agg_isg[s].append(float(np.mean(r_list)))

        if save_root is not None:
            subname = {0: "sub01", 1: "sub02", 2: "sub03", 3: "sub05"}
            for s in available:
                pdir = save_root / subname[s] / f"preds_{save_split_name}_episodes"
                gdir = save_root / subname[s] / f"preds_{save_split_name}_episodes_gt"
                pdir.mkdir(parents=True, exist_ok=True)
                gdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / f"{ds}_pred.npy", preds_by_sub[s])
                np.save(gdir / f"{ds}_gt.npy", gts_by_sub[s])

    per_sub_means: Dict[int, dict] = {}
    isg_means: Dict[int, float] = {}
    for s in range(n_subjects):
        if used_counts[s] > 0:
            per_sub_means[s] = {
                "r": float(np.mean(agg[s]["r"])),
                "rho": float(np.mean(agg[s]["rho"])),
                "r2": float(np.mean(agg[s]["r2"])),
            }
            isg_means[s] = float(np.mean(agg_isg[s])) if agg_isg[s] else float("nan")
        else:
            per_sub_means[s] = {"r": float("nan"), "rho": float("nan"), "r2": float("nan")}
            isg_means[s] = float("nan")
    return per_sub_means, isg_means, used_counts, per_episode_scores


# --------------------------------------------------------------------------- #
#  Export expert routing weights for interpretability (Fig. 4 in paper)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def export_episode_voxel_topk_weight_means(
    model: FmriEncoder_MoE,
    episode: str,
    roots_feat: Dict[str, Path],
    layers: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
    subject_id: int,
    K: int,
    out_dir: Path,
):
    """
    Export per-parcel Top-K expert weights for a given episode & subject.

    Saves:
        {episode}_sub{id}_weights_mean_OK.npy   [O, K] float32
        {episode}_sub{id}_experts_idx_OK.npy     [O, K] int64
    """
    ds_tmp = WindowedDataset(
        ids=[episode],
        video_root=roots_feat["video"],
        text_root=roots_feat["text"],
        audio_root=roots_feat["audio"],
        anchor_fmri_root=roots_feat["anchor_fmri_root"],
        layers_arg=layers,
        layer_agg=layer_agg,
        window_tr=window_tr,
        stride_tr=stride_tr,
        frames_per_tr=frames_per_tr,
    )
    loader = DataLoader(
        ds_tmp, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )

    E = model.num_experts
    O = model.n_outputs
    contrib_sum_eo = torch.zeros(E, O, dtype=torch.float64)
    weights_pre_token_sum_e = torch.zeros(E, dtype=torch.float64)
    total_tokens = 0

    for batch in loader:
        batch = batch.to(device)
        batch.data["subject_id"] = torch.full((1,), subject_id, dtype=torch.long, device=device)
        _, w_final, out_BNEO, w_pre = model.forward_with_details(batch, pool_outputs=True)
        contrib = (out_BNEO * w_final.unsqueeze(-1)).sum(dim=1).squeeze(0).double().cpu()
        contrib_sum_eo += contrib
        weights_pre_token_sum_e += w_pre.sum(dim=1).squeeze(0).double().cpu()
        total_tokens += w_pre.size(1)

    contrib_mean_eo = (contrib_sum_eo / max(1, total_tokens)).abs()
    _, topk_idx = torch.topk(contrib_mean_eo.T, k=K, dim=1)
    experts_idx_OK = topk_idx.to(torch.int64).cpu().numpy()

    weights_pre_mean_e = (weights_pre_token_sum_e / max(1, total_tokens)).to(torch.float32)
    weights_mean_OK = weights_pre_mean_e[torch.as_tensor(experts_idx_OK)].cpu().numpy().astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{episode}_sub{subject_id + 1:02d}"
    np.save(out_dir / f"{tag}_weights_mean_OK.npy", weights_mean_OK)
    np.save(out_dir / f"{tag}_experts_idx_OK.npy", experts_idx_OK)
    print(f"[EXPORT] {tag}: saved to {out_dir}")


def pick_friends_episode(ids: List[str]) -> str:
    """Pick the first Friends episode from the list (for train-time probing)."""
    fs = [ds for ds in ids if "friends" in ds.lower() or ds.startswith("ses-") and "s0" in ds]
    return fs[0] if fs else ids[0]
