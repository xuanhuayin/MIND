# -*- coding: utf-8 -*-
"""
AFIRE / MIND  --  model definitions.

* ``FmriEncoder_MoE``  : MIND decoder  (MoE + SADGate)   -- Table 1 "w. MIND"
* ``FmriEncoder``       : MLP baseline  (subject-conditional linear) -- Table 1 "Baseline"
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from .modules import build_transformer_encoder, build_projector


# ========================================================================== #
#  Expert MLP (used inside FmriEncoder_MoE)
# ========================================================================== #
class _ExpertMLP(nn.Module):
    """Single expert decoder head."""

    def __init__(self, in_dim: int, out_dim: int, layers: int = 1,
                 hidden_mult: float = 4.0, dropout: float = 0.1):
        super().__init__()
        layers = max(1, int(layers))
        if layers == 1:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
        else:
            h = int(round(in_dim * hidden_mult))
            blocks = [nn.Dropout(dropout), nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(layers - 2):
                blocks += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
            blocks += [nn.Linear(h, out_dim)]
            self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ========================================================================== #
#  MIND  (MoE decoder + SADGate)
# ========================================================================== #
class FmriEncoder_MoE(nn.Module):
    """
    Multimodal -> Transformer encoder -> (pool to TR) -> MoE decode.

    ``combine_mode`` controls expert weighting (SADGate variants):
      - ``"router"``:           token-dependent softmax routing only
      - ``"learned"``:          global learnable weights (+ optional subject bias)
      - ``"router_x_learned"``: product of router * learned  (recommended)
    """

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        n_subjects: int | None = None,
        num_experts: int = 4,
        top_k: int = 1,
        feature_aggregation: str = "cat",
        layer_aggregation: str = "cat",
        subject_embedding: bool = False,
        moe_dropout: float = 0.1,
        expert_layers: int = 1,
        expert_hidden_mult: float = 4.0,
        combine_mode: str = "router",
        subject_expert_bias: bool = False,
    ):
        super().__init__()
        assert feature_aggregation in ("cat", "sum")
        assert layer_aggregation in ("cat", "mean")
        assert combine_mode in ("router", "learned", "router_x_learned")

        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation
        self.combine_mode = combine_mode

        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_layers = int(max(1, expert_layers))
        self.expert_hidden_mult = float(expert_hidden_mult)

        hidden = 3072
        self.hidden = hidden
        self.n_outputs = int(n_outputs)

        # -- per-modality projectors --
        self.projectors = nn.ModuleDict()
        num_modalities = len(feature_dims)
        for modality, (num_layers, feat_dim) in feature_dims.items():
            in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
            out_dim = hidden if feature_aggregation == "sum" else (hidden // num_modalities)
            self.projectors[modality] = build_projector(in_dim, out_dim)
        # pad last projector so concatenation equals `hidden`
        if feature_aggregation == "cat":
            used = (hidden // num_modalities) * num_modalities
            if used != hidden:
                last = list(self.projectors.keys())[-1]
                num_layers, feat_dim = feature_dims[last]
                in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
                want = hidden - (hidden // num_modalities) * (num_modalities - 1)
                self.projectors[last] = build_projector(in_dim, want)

        # -- positional / subject embeddings --
        max_T2 = max(n_output_timesteps * 2, 1024)
        self.time_pos_embed = nn.Parameter(torch.randn(1, max_T2, hidden))
        if subject_embedding and self.n_subjects:
            self.subject_embed = nn.Embedding(self.n_subjects, hidden)

        # -- backbone transformer --
        self.encoder = build_transformer_encoder(
            dim=hidden, depth=8, heads=8,
            attn_dropout=0.0, ff_dropout=0.0,
        )

        # -- 2 Hz -> TR pooling --
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # -- MoE: router + experts --
        self.router = nn.Linear(hidden, self.num_experts)
        self.experts = nn.ModuleList([
            _ExpertMLP(hidden, self.n_outputs,
                       layers=self.expert_layers,
                       hidden_mult=self.expert_hidden_mult,
                       dropout=moe_dropout)
            for _ in range(self.num_experts)
        ])

        # learnable global expert logits (Subject Prior Router)
        if self.combine_mode in ("learned", "router_x_learned"):
            self.expert_logit = nn.Parameter(torch.zeros(self.num_experts))
        if subject_expert_bias and self.n_subjects:
            self.subject_expert_bias = nn.Embedding(self.n_subjects, self.num_experts)
        else:
            self.subject_expert_bias = None

        self.last_aux_loss: torch.Tensor | None = None
        self._debug_last_weights_avg: torch.Tensor | None = None
        self._debug_last_weights_pre_avg: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def _aggregate_features(self, batch) -> torch.Tensor:
        tensors = []
        for modality, (num_layers, feat_dim) in self.feature_dims.items():
            data = batch.data[modality].to(torch.float32)
            if data.ndim == 3:
                data = data.unsqueeze(1)
            if self.layer_aggregation == "mean":
                data = data.mean(dim=1)
            else:
                data = rearrange(data, "b l d t -> b (l d) t")
            data = data.transpose(1, 2)
            proj = self.projectors[modality](data)
            tensors.append(proj)
        return torch.cat(tensors, dim=-1) if self.feature_aggregation == "cat" else sum(tensors)

    def _compute_router_probs(self, x_routed: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.router(x_routed), dim=-1)

    def _get_learned_weights(self, B: int, N: int, subject_id):
        base = self.expert_logit
        if self.subject_expert_bias is not None and subject_id is not None:
            bias = self.subject_expert_bias(subject_id)
            w = F.softmax(base.unsqueeze(0) + bias, dim=-1).unsqueeze(1).expand(B, N, self.num_experts)
        else:
            w = F.softmax(base, dim=-1).view(1, 1, -1).expand(B, N, self.num_experts)
        return w

    # ------------------------------------------------------------------ #
    def _route_and_decode_with_experts(self, x_tr, subject_id):
        """Returns (y, weights_final, experts_out, weights_pre)."""
        B, N, H = x_tr.shape

        if hasattr(self, "subject_embed") and subject_id is not None:
            x_routed = x_tr + self.subject_embed(subject_id).unsqueeze(1)
        else:
            x_routed = x_tr

        x_flat = x_tr.reshape(-1, H)
        experts_out = torch.stack(
            [e(x_flat).reshape(B, N, self.n_outputs) for e in self.experts], dim=2
        )  # [B, N, E, O]

        self.last_aux_loss = None

        if self.combine_mode == "router":
            probs_full = self._compute_router_probs(x_routed)
            weights_pre = probs_full
            weights_final = self._topk_sparse(probs_full)
            self._compute_load_balance_loss(probs_full)

        elif self.combine_mode == "learned":
            weights_pre = self._get_learned_weights(B, N, subject_id)
            weights_final = weights_pre

        else:  # router_x_learned
            probs_full = self._compute_router_probs(x_routed)
            learned_w = self._get_learned_weights(B, N, subject_id)
            mix = probs_full * learned_w
            weights_pre = mix / (mix.sum(dim=-1, keepdim=True) + 1e-8)
            weights_final = self._topk_sparse(mix)
            self._compute_load_balance_loss(probs_full)

        y = torch.einsum("bneo,bne->bno", experts_out, weights_final)

        with torch.no_grad():
            self._debug_last_weights_avg = weights_final.mean(dim=(0, 1)).detach().cpu()
            self._debug_last_weights_pre_avg = weights_pre.mean(dim=(0, 1)).detach().cpu()

        return y, weights_final, experts_out, weights_pre

    def _topk_sparse(self, probs: torch.Tensor) -> torch.Tensor:
        if self.top_k is not None and self.top_k < self.num_experts:
            topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
            out = torch.zeros_like(probs)
            out.scatter_(dim=-1, index=topk_idx, src=topk_probs)
            return out
        return probs

    def _compute_load_balance_loss(self, probs_full: torch.Tensor):
        all_probs = probs_full.reshape(-1, self.num_experts)
        importance = all_probs.mean(dim=0)
        if self.top_k == 1:
            top1 = torch.argmax(probs_full, dim=-1)
            load = F.one_hot(top1.view(-1), num_classes=self.num_experts).float().mean(dim=0)
        else:
            load = all_probs.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(load * importance)

    def _route_and_decode(self, x_tr, subject_id):
        y, w_final, _, _ = self._route_and_decode_with_experts(x_tr, subject_id)
        return y, w_final

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def forward(self, batch, pool_outputs: bool = True):
        x = self._aggregate_features(batch)
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} exceeds time_pos_embed length {self.time_pos_embed.size(1)}")
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)

        subj = batch.data.get("subject_id", None) if hasattr(batch, "data") else None

        if pool_outputs:
            x_tr = self.pooler(x.transpose(1, 2)).transpose(1, 2)
            y_bn, _ = self._route_and_decode(x_tr, subj)
            return y_bn.transpose(1, 2)                         # [B, O, N]
        else:
            y_bt, _ = self._route_and_decode(x, subj)
            return y_bt.transpose(1, 2)                         # [B, O, T2]

    def forward_with_details(self, batch, pool_outputs: bool = True):
        x = self._aggregate_features(batch)
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} exceeds time_pos_embed length {self.time_pos_embed.size(1)}")
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)

        subj = batch.data.get("subject_id", None) if hasattr(batch, "data") else None

        if pool_outputs:
            x_tr = self.pooler(x.transpose(1, 2)).transpose(1, 2)
            y, w_final, exp_out, w_pre = self._route_and_decode_with_experts(x_tr, subj)
            return y.transpose(1, 2), w_final, exp_out, w_pre
        else:
            y, w_final, exp_out, w_pre = self._route_and_decode_with_experts(x, subj)
            return y.transpose(1, 2), w_final, exp_out, w_pre

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def get_expert_weights(self, subject_id=None):
        if getattr(self, "combine_mode", "router") == "router":
            return None
        if subject_id is None:
            return F.softmax(self.expert_logit, dim=-1).cpu()
        if getattr(self, "subject_expert_bias", None) is None:
            return F.softmax(self.expert_logit, dim=-1).expand(subject_id.shape[0], -1).cpu()
        bias = self.subject_expert_bias(subject_id)
        return F.softmax(self.expert_logit.unsqueeze(0) + bias, dim=-1).cpu()

    @torch.no_grad()
    def get_last_weight_avg(self):
        return getattr(self, "_debug_last_weights_avg", None)

    @torch.no_grad()
    def get_last_weight_pre_avg(self):
        return getattr(self, "_debug_last_weights_pre_avg", None)


# ========================================================================== #
#  Baseline: subject-conditional linear decoder  (Table 1 "Baseline")
# ========================================================================== #
class SubjectConditionalLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int | None):
        super().__init__()
        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None
        if self.n_subjects is None:
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            self.weight = nn.Parameter(torch.empty(self.n_subjects, out_channels, in_channels))
            self.bias = nn.Parameter(torch.empty(self.n_subjects, out_channels))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            bound = 1 / (in_channels ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, subject_id=None):
        if self.n_subjects is None:
            return self.lin(x)
        if subject_id is None:
            raise ValueError("subject_id required when n_subjects > 1")
        W = self.weight[subject_id]
        b = self.bias[subject_id]
        return torch.einsum("bnh,boh->bno", x, W) + b.unsqueeze(1)


class FmriEncoder(nn.Module):
    """Simple baseline encoder: projectors -> Transformer -> pool -> subject-conditional linear."""

    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        n_subjects: int | None = None,
        feature_aggregation: str = "cat",
        layer_aggregation: str = "cat",
        subject_embedding: bool = False,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation
        self.n_subjects = n_subjects

        hidden = 3072
        self.hidden = hidden

        self.projectors = nn.ModuleDict()
        for modality, (num_layers, feat_dim) in feature_dims.items():
            in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
            out_dim = hidden if feature_aggregation == "sum" else (hidden // len(feature_dims))
            self.projectors[modality] = build_projector(in_dim, out_dim)

        self.time_pos_embed = nn.Parameter(torch.randn(1, 1024, hidden))
        if subject_embedding and n_subjects:
            self.subject_embed = nn.Embedding(n_subjects, hidden)

        self.encoder = build_transformer_encoder(
            dim=hidden, depth=8, heads=8,
            attn_dropout=0.0, ff_dropout=0.0,
        )

        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        self.pred_head = SubjectConditionalLinear(hidden, n_outputs, n_subjects)

    def forward(self, batch, pool_outputs: bool = True):
        # aggregate modalities
        tensors = []
        for modality in self.feature_dims:
            data = batch.data[modality].to(torch.float32)
            if data.ndim == 3:
                data = data.unsqueeze(1)
            if self.layer_aggregation == "mean":
                data = data.mean(dim=1)
            else:
                data = rearrange(data, "b l d t -> b (l d) t")
            data = data.transpose(1, 2)
            tensors.append(self.projectors[modality](data))

        x = torch.cat(tensors, dim=-1) if self.feature_aggregation == "cat" else sum(tensors)
        x = x + self.time_pos_embed[:, :x.size(1)]

        subject_id = batch.data.get("subject_id", None)
        if hasattr(self, "subject_embed") and subject_id is not None:
            x = x + self.subject_embed(subject_id).unsqueeze(1)

        x = self.encoder(x)

        if pool_outputs:
            x_tr = self.pooler(x.transpose(1, 2)).transpose(1, 2)
            y = self.pred_head(x_tr, subject_id)
            return y.transpose(1, 2)
        else:
            y = self.pred_head(x, subject_id)
            return y.transpose(1, 2)
