#!/usr/bin/env bash
# Train MIND with Qwen2.5-Omni backbone features.
# Qwen features: fused multimodal embeddings.
# Note: For Qwen we point all three modality roots at the same fused feature dir.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-.}"
QWEN_FEAT="${DATA_ROOT}/pipeline_QWEN/features/multimodal_2hz/sub-01"

python train.py \
    --all_list  splits/all_episodes.txt \
    --video_root "${QWEN_FEAT}" \
    --text_root  "${QWEN_FEAT}" \
    --audio_root "${QWEN_FEAT}" \
    --fmri_root_sub1 "${DATA_ROOT}/fmri_data/sub-01" \
    --fmri_root_sub2 "${DATA_ROOT}/fmri_data/sub2" \
    --fmri_root_sub3 "${DATA_ROOT}/fmri_data/sub3" \
    --fmri_root_sub5 "${DATA_ROOT}/fmri_data/sub5" \
    --layers 0.6,0.8,1.0 \
    --layer_aggregation group_mean \
    --frames_per_tr 3 \
    --window_tr 100 --stride_tr 50 \
    --epochs 25 --batch_size 1 --lr 1e-3 --num_workers 0 \
    --moe_num_experts 6 --moe_top_k 2 --moe_dropout 0.1 \
    --moe_combine_mode router_x_learned \
    --moe_subject_expert_bias \
    --moe_aux_weight 0.01 \
    --subject_embedding \
    --out_dir outputs/MIND_Qwen \
    --log_dir logs/MIND_Qwen \
    "$@"
