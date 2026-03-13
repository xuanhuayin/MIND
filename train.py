# -*- coding: utf-8 -*-
"""
Unified training script for MIND (MoE + SADGate) on the Algonauts 2025 benchmark.

Supports all three backbones (TRIBE / ImageBind / Qwen2.5-Omni) via command-line
arguments that specify the feature directories.

Usage examples -- see scripts/ directory.
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mind.dataset import Batch, WindowedDataset, collate_fn
from mind.evaluate import (
    evaluate_episodes,
    export_episode_voxel_topk_weight_means,
    pick_friends_episode,
)
from mind.model import FmriEncoder_MoE
from mind.utils import load_fmri_flexible, orient_fmri, read_ids, set_seed

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ========================================================================== #
#  Argument parser
# ========================================================================== #
def parse_args():
    ap = argparse.ArgumentParser(
        description="Train MIND decoder on Algonauts 2025"
    )

    # --- data splits ---
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list", type=str, default="")
    ap.add_argument("--all_list", type=str, default="")
    ap.add_argument("--split_ratio", type=float, default=0.9)

    # --- feature roots (backbone-dependent) ---
    ap.add_argument("--video_root", type=str, required=True,
                    help="Dir with {episode}.npy video features at 2 Hz")
    ap.add_argument("--text_root", type=str, required=True,
                    help="Dir with {episode}.npy text features at 2 Hz")
    ap.add_argument("--audio_root", type=str, required=True,
                    help="Dir with {episode}.npy audio features at 2 Hz")

    # --- fMRI roots (per-subject) ---
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # --- layer selection ---
    ap.add_argument("--layers", type=str, default="last1",
                    help="Layer selection: 'last4', 'all', 'idx:0,3,7', '0.25,0.75'")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean", "none"])

    # --- output parcels ---
    ap.add_argument("--n_outputs", type=int, default=1000,
                    help="Number of output parcels (1000 for Schaefer atlas)")

    # --- windowing ---
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)

    # --- optimization ---
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # --- model (MoE / SADGate) ---
    ap.add_argument("--subject_embedding", action="store_true")
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_aux_weight", type=float, default=0.01)
    ap.add_argument("--moe_dropout", type=float, default=0.1)
    ap.add_argument("--moe_combine_mode", type=str, default="router_x_learned",
                    choices=["router", "learned", "router_x_learned"])
    ap.add_argument("--moe_subject_expert_bias", action="store_true")

    # --- training options ---
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--disable_swa", action="store_true")

    # --- output ---
    ap.add_argument("--out_dir", type=str, default="outputs/MIND")
    ap.add_argument("--log_dir", type=str, default="logs/MIND")

    # --- export ---
    ap.add_argument("--export_voxel_topk_episode", type=str, default="")
    ap.add_argument("--export_voxel_topk_subject", type=int, default=0)
    ap.add_argument("--export_voxel_topk_k", type=int, default=None)

    ap.add_argument("--seed", type=int, default=33)
    return ap.parse_args()


# ========================================================================== #
#  Main
# ========================================================================== #
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] {device}")

    # ---- directories ----
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "experts").mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    tb_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] {tb_dir}")

    # ---- fMRI roots ----
    fmri_roots: Dict[int, Path] = {
        0: Path(args.fmri_root_sub1),
        1: Path(args.fmri_root_sub2),
        2: Path(args.fmri_root_sub3),
        3: Path(args.fmri_root_sub5),
    }
    n_subjects = len(fmri_roots)

    # ---- episode splits ----
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.seed)
        rnd.shuffle(all_ids)
        k = max(1, min(len(all_ids) - 1, int(round(len(all_ids) * args.split_ratio))))
        train_ids, val_ids = all_ids[:k], all_ids[k:]
        print(f"[SPLIT] auto: train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide --all_list OR both --train_list and --val_list")
        train_ids = read_ids(args.train_list)
        val_ids = read_ids(args.val_list)
        print(f"[SPLIT] lists: train={len(train_ids)}  val={len(val_ids)}")

    layer_agg = "group_mean" if args.layer_aggregation != "none" else "none"

    # ---- datasets ----
    train_set = WindowedDataset(
        train_ids, Path(args.video_root), Path(args.text_root), Path(args.audio_root),
        fmri_roots[0], args.layers, layer_agg,
        args.window_tr, args.stride_tr, args.frames_per_tr,
    )
    val_set = WindowedDataset(
        val_ids if val_ids else train_ids[:1],
        Path(args.video_root), Path(args.text_root), Path(args.audio_root),
        fmri_roots[0], args.layers, layer_agg,
        args.window_tr, args.stride_tr, args.frames_per_tr,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- model ----
    feat_dims = {
        "video": (train_set.G, train_set.Dv),
        "text": (train_set.G, train_set.Dt),
        "audio": (train_set.G, train_set.Da),
    }
    model = FmriEncoder_MoE(
        feature_dims=feat_dims,
        n_outputs=args.n_outputs,
        n_output_timesteps=args.window_tr,
        n_subjects=n_subjects,
        num_experts=args.moe_num_experts,
        top_k=args.moe_top_k,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=args.subject_embedding,
        moe_dropout=args.moe_dropout,
        combine_mode=args.moe_combine_mode,
        subject_expert_bias=args.moe_subject_expert_bias,
    ).to(device)
    print(f"[MODEL] experts={args.moe_num_experts}  top_k={args.moe_top_k}  "
          f"combine={args.moe_combine_mode}  subj_embed={args.subject_embedding}  "
          f"subj_bias={args.moe_subject_expert_bias}")

    # ---- optional gradient checkpointing ----
    if args.grad_ckpt:
        _enable_grad_ckpt(model)

    # ---- optimizer & scheduler ----
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), eps=1e-8,
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=steps_per_epoch * args.epochs,
        pct_start=args.warmup_pct, anneal_strategy="cos",
    )

    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = (not args.disable_swa) and (swa_start_epoch < args.epochs)
    swa_model = AveragedModel(model) if use_swa else None

    # ---- bookkeeping ----
    num_E = args.moe_num_experts
    expert_weight_sum = np.zeros((n_subjects, num_E), dtype=np.float64)
    expert_weight_cnt = np.zeros((n_subjects,), dtype=np.int64)
    experts_dir = out_dir / "experts"

    best_key = float("-inf")
    fmri_cache: Dict[Tuple[int, str], np.ndarray] = {}
    global_step = 0
    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[PROBE] {train_probe_ds}")

    # ================================================================== #
    #  Training loop
    # ================================================================== #
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            batch = batch.to(device)
            loss_terms: List[torch.Tensor] = []
            aux_terms: List[torch.Tensor] = []

            for s in range(n_subjects):
                batch.data["subject_id"] = torch.full(
                    (batch.data["video"].size(0),), s, dtype=torch.long, device=device,
                )
                y = model.forward(batch, pool_outputs=True)  # [B, O, N]

                # track expert weights
                w_pre = model.get_last_weight_pre_avg()
                if w_pre is not None:
                    expert_weight_sum[s] += w_pre.numpy()
                    expert_weight_cnt[s] += 1

                B, O, N = y.shape
                ds_list = batch.data["ds_list"]
                st_list = batch.data["start_tr_list"]
                for i in range(B):
                    ds = ds_list[i]
                    st = int(st_list[i])
                    ed = st + N
                    try:
                        key = (s, ds)
                        if key not in fmri_cache:
                            gt_all = orient_fmri(load_fmri_flexible(fmri_roots[s], ds))
                            fmri_cache[key] = gt_all
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)
                        loss_terms.append(criterion(y[i], gt_win))
                    except FileNotFoundError:
                        continue

                if args.moe_aux_weight > 0 and getattr(model, "last_aux_loss", None) is not None:
                    aux_terms.append(model.last_aux_loss)

            if not loss_terms:
                continue
            loss = torch.stack(loss_terms).mean()
            if aux_terms:
                loss = loss + args.moe_aux_weight * torch.stack(aux_terms).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", loss.item(), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss = running / max(1, len(train_loader))
        writer.add_scalar("loss/train_epoch", train_loss, epoch)

        # ---- validation (window loss, anchor = subject 0) ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch.data["subject_id"] = torch.zeros(
                    (batch.data["video"].size(0),), dtype=torch.long, device=device,
                )
                y = model.forward(batch, pool_outputs=True)
                B, O, N = y.shape
                for i in range(B):
                    ds = batch.data["ds_list"][i]
                    st = int(batch.data["start_tr_list"][i])
                    ed = st + N
                    try:
                        key = (0, ds)
                        if key not in fmri_cache:
                            fmri_cache[key] = orient_fmri(load_fmri_flexible(fmri_roots[0], ds))
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)
                        val_loss += nn.MSELoss()(y[i], gt_win).item()
                    except FileNotFoundError:
                        continue
        val_loss /= max(1, len(val_loader))
        writer.add_scalar("loss/val_epoch", val_loss, epoch)

        # ---- full episode evaluation ----
        roots_feat = {
            "video": Path(args.video_root),
            "text": Path(args.text_root),
            "audio": Path(args.audio_root),
        }
        per_sub, isg, used, ep_scores = evaluate_episodes(
            model, val_ids, roots_feat, fmri_roots,
            args.layers, layer_agg,
            args.window_tr, args.stride_tr, args.frames_per_tr,
            device, save_root=out_dir, save_split_name="val",
            n_subjects=n_subjects,
        )

        # train-probe (Friends)
        probe_sub, probe_isg, _, _ = evaluate_episodes(
            model, [train_probe_ds], roots_feat, fmri_roots,
            args.layers, layer_agg,
            args.window_tr, args.stride_tr, args.frames_per_tr,
            device, save_root=out_dir, save_split_name="trainprobe",
            n_subjects=n_subjects,
        )

        # ---- logging ----
        acc_key = []
        parts = [f"Epoch {epoch}: train={train_loss:.6f}  val={val_loss:.6f}"]
        for s in range(n_subjects):
            r = per_sub.get(s, {}).get("r", float("nan"))
            rho = per_sub.get(s, {}).get("rho", float("nan"))
            r2 = per_sub.get(s, {}).get("r2", float("nan"))
            isg_v = isg.get(s, float("nan"))
            if not np.isnan(r):
                acc_key.append(r)
            writer.add_scalar(f"val/S{s+1:02d}_r", 0.0 if np.isnan(r) else r, epoch)
            writer.add_scalar(f"val/S{s+1:02d}_rho", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"val/S{s+1:02d}_r2", 0.0 if np.isnan(r2) else r2, epoch)
            if not np.isnan(isg_v):
                writer.add_scalar(f"val/S{s+1:02d}_ISG", isg_v, epoch)
            parts.append(f"S{s+1:02d} r={r:.4f} rho={rho:.4f} R2={r2:.4f} ISG={isg_v:.4f}")

        val_key = float(np.mean(acc_key)) if acc_key else float("-inf")
        writer.add_scalar("val/mean_r", 0.0 if np.isnan(val_key) else val_key, epoch)
        print("  |  ".join(parts))

        # expert weight stats
        _log_expert_weights(
            expert_weight_sum, expert_weight_cnt, num_E, n_subjects,
            epoch, writer, experts_dir,
        )
        expert_weight_sum[:] = 0.0
        expert_weight_cnt[:] = 0

        # save best
        if val_key > best_key:
            best_key = val_key
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")

    # ---- SWA ----
    if use_swa:
        print("Updating BN for SWA ...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print(f"\nDone.  Best val mean Pearson r = {best_key:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")

    # ---- optional post-training export ----
    if args.export_voxel_topk_episode:
        best_path = out_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            model.to(device).eval()
        roots_feat["anchor_fmri_root"] = Path(args.fmri_root_sub1)
        K = args.export_voxel_topk_k if args.export_voxel_topk_k else args.moe_top_k
        export_episode_voxel_topk_weight_means(
            model, args.export_voxel_topk_episode, roots_feat,
            args.layers, layer_agg,
            args.window_tr, args.stride_tr, args.frames_per_tr,
            device, args.export_voxel_topk_subject, K,
            out_dir / "experts" / "episode_voxel_topk",
        )


# ========================================================================== #
#  Helpers
# ========================================================================== #
def _enable_grad_ckpt(model):
    try:
        import torch.utils.checkpoint as ckpt
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            for blk in model.encoder.layers:
                fwd = blk.forward

                def wrapper(*x, _f=fwd, **kw):
                    return ckpt.checkpoint(_f, *x, use_reentrant=False, **kw)

                blk.forward = wrapper
            print("[CKPT] gradient checkpointing enabled")
        else:
            print("[CKPT] encoder.layers not found; skipped")
    except Exception as e:
        print(f"[CKPT] failed: {e}")


def _log_expert_weights(
    weight_sum, weight_cnt, num_E, n_subjects, epoch, writer, experts_dir,
):
    txt_path = experts_dir / f"epoch_{epoch:03d}.txt"
    with open(txt_path, "w", encoding="utf-8") as ftxt:
        for s in range(n_subjects):
            if weight_cnt[s] > 0:
                w = weight_sum[s] / max(1, weight_cnt[s])
                for e in range(num_E):
                    writer.add_scalar(f"experts/S{s+1:02d}/E{e}", float(w[e]), epoch)
                order = np.argsort(-w)
                pairs = [f"E{int(e)}={w[e]:.3f}" for e in order]
                ftxt.write(f"S{s+1:02d}: {', '.join(pairs)}\n")
            else:
                ftxt.write(f"S{s+1:02d}: no data\n")


if __name__ == "__main__":
    main()
