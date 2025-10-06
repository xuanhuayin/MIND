# -*- coding: utf-8 -*-
# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/moe_transformer_encoder.py
from __future__ import annotations
import typing as tp
import math
import pydantic
import torch
from torch import nn
from einops import rearrange

# --------------------------
# Configs
# --------------------------
class FmriEncoderMoEConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder_MoETransformer"] = "FmriEncoder_MoETransformer"

    # dataset/model glue
    n_subjects: int | None = 4
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    # backbone
    hidden: int = 3072
    transformer_depth: int = 8
    n_heads: int = 8
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    layer_dropout: float = 0.0

    # MoE
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_expert_layers: int = 1
    moe_hidden_mult: float = 4.0
    moe_dropout: float = 0.1
    moe_token_chunk: int = 8192
    moe_ffn_where: str = "last1"      # e.g. last1 / last2 / idx:3,5
    moe_share_dense: bool = False     # 共享稠密专家（默认 False）
    moe_share_alpha: float = 1.0      # 共享专家缩放

# --------------------------
# Subject-conditional head
# --------------------------
class SubjectConditionalLinear(nn.Module):
    """
    与 TRIBE 描述一致的 subject-conditional linear。
    """
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int | None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None

        if self.n_subjects is None:
            self.lin = nn.Linear(in_channels, out_channels, bias=True)
        else:
            S = self.n_subjects
            O, H = out_channels, in_channels
            self.weight = nn.Parameter(torch.empty(S, O, H))
            self.bias = nn.Parameter(torch.empty(S, O))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            bound = 1 / (H ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        if self.n_subjects is None:
            return self.lin(x)
        if subject_id is None:
            raise ValueError("SubjectConditionalLinear: subject_id is required when n_subjects>1.")
        W = self.weight[subject_id]  # [B,O,H]
        b = self.bias[subject_id]    # [B,O]
        y = torch.einsum("bnh,boh->bno", x, W) + b.unsqueeze(1)
        return y

# --------------------------
# Expert MLP (可共享/不共享)
# --------------------------
class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, mult: float, dropout: float, n_layers: int = 1,
                 shared_dense: bool = False, alpha: float = 1.0):
        super().__init__()
        d_ff = int(round(d_model * mult))
        layers = []
        if n_layers <= 1:
            layers += [nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                       nn.Linear(d_ff, d_model), nn.Dropout(dropout)]
        else:
            layers += [nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(d_ff, d_ff), nn.GELU(), nn.Dropout(dropout)]
            layers += [nn.Linear(d_ff, d_model), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

        # 共享稠密专家：把最后一层权重缩放（简单做法）
        self.shared_dense = shared_dense
        self.alpha = alpha

    def forward(self, x):
        y = self.net(x)
        if self.shared_dense and self.alpha != 1.0:
            y = y * self.alpha
        return y

# --------------------------
# Token-level MoE (top-1 / top-k)
# 不做 detach、不用 no_grad；保持可反传。
# --------------------------
class TokenMoE(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 mult: float, dropout: float, expert_layers: int,
                 token_chunk: int = 8192,
                 shared_dense: bool = False, share_alpha: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = max(1, int(top_k))
        self.token_chunk = int(token_chunk)

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, mult, dropout, n_layers=expert_layers,
                      shared_dense=shared_dense, alpha=share_alpha)
            for _ in range(n_experts)
        ])

        # forward 过程中写入，供 gather_moe_losses() 读取
        self._aux_loss: torch.Tensor | None = None
        self._z_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor:
        if self._aux_loss is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        return self._aux_loss

    @property
    def z_loss(self) -> torch.Tensor:
        if self._z_loss is None:
            return torch.tensor(0.0, device=self.gate.weight.device)
        return self._z_loss

    def _dispatch_top1(self, x_flat, logits, probs):
        BT, D = x_flat.shape
        top1 = logits.argmax(dim=-1)  # [BT]
        y_flat = torch.zeros_like(x_flat)
        # 简单可靠的 expert-by-expert 分发
        for e in range(self.n_experts):
            idx = (top1 == e).nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            y_e = self.experts[e](x_flat[idx])     # [n_e, D]
            y_flat[idx] = y_e
        return y_flat

    def _dispatch_topk(self, x_flat, probs):
        # 混合：对每个 token 取 top-k 专家并加权相加
        BT, D = x_flat.shape
        K = self.top_k
        topk_vals, topk_idx = probs.topk(K, dim=-1)  # [BT,K]
        y_accum = torch.zeros(BT, D, device=x_flat.device, dtype=x_flat.dtype)
        for kk in range(K):
            idx_e = topk_idx[:, kk]          # [BT]
            w_e   = topk_vals[:, kk]         # [BT]
            for e in range(self.n_experts):
                sel = (idx_e == e).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    continue
                y_e = self.experts[e](x_flat[sel])     # [n_e, D]
                y_accum[sel] += y_e * w_e[sel].unsqueeze(1)
        return y_accum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,D] -> [B,T,D]
        """
        B, T, D = x.shape
        x_flat = x.reshape(B*T, D)                    # [BT,D]
        logits = self.gate(x_flat)                    # [BT,E]
        probs  = torch.softmax(logits, dim=-1)        # [BT,E]

        # Load-balance（Switch 风格）：让各 expert 的平均概率接近均匀
        # 这部分需要对 gate 的梯度，不能 detach
        mean_prob = probs.mean(dim=0)                 # [E]
        aux = (mean_prob * mean_prob * self.n_experts).sum()  # scalar
        zloss = (logits ** 2).mean() * 1e-2

        if self.top_k == 1:
            y_flat = self._dispatch_top1(x_flat, logits, probs)
        else:
            y_flat = self._dispatch_topk(x_flat, probs)

        self._aux_loss = aux
        self._z_loss = zloss
        return y_flat.view(B, T, D)

# --------------------------
# 简单 Transformer Encoder Block（支持 MoE FFN）
# --------------------------
class MHABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, resid_dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x):
        h = self.ln(x)
        y, _ = self.mha(h, h, h, need_weights=False)
        return x + self.drop(y)

class FFNBlock(nn.Module):
    def __init__(self, d_model: int, mult: float, dropout: float):
        super().__init__()
        d_ff = int(round(d_model * mult))
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x):
        h = self.ln(x)
        return x + self.ff(h)

class MoEFFNBlock(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, mult: float, dropout: float,
                 expert_layers: int, token_chunk: int, shared_dense: bool, share_alpha: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.moe = TokenMoE(d_model, n_experts, top_k, mult, dropout,
                            expert_layers, token_chunk, shared_dense, share_alpha)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln(x)
        y = self.moe(h)
        return x + self.drop(y)

class Encoder(nn.Module):
    def __init__(self, d_model: int, depth: int, n_heads: int,
                 attn_dropout: float, resid_dropout: float,
                 moe_slots: set[int],
                 moe_kwargs: dict):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(MHABlock(d_model, n_heads, attn_dropout, resid_dropout))
            if i in moe_slots:
                layers.append(MoEFFNBlock(**moe_kwargs))
            else:
                layers.append(FFNBlock(d_model, mult=4.0, dropout=resid_dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x

# --------------------------
# 主模型
# --------------------------
class FmriEncoder_MoETransformer(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderMoEConfig,
        n_input_timesteps_2hz: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        H = config.hidden

        # --- 每模态投影 ---
        self.projectors = nn.ModuleDict()
        n_mods = len(feature_dims)
        for modality, tup in feature_dims.items():
            num_layers, feat_dim = tup
            in_dim = (feat_dim * num_layers if config.layer_aggregation == "cat" else feat_dim)
            out_dim = (H // n_mods) if config.feature_aggregation == "cat" else H
            self.projectors[modality] = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Linear(out_dim, out_dim),
            )

        # --- 模态融合 ---
        self.combiner = nn.Identity()

        # --- 时间位置编码 ---
        pe_len = n_input_timesteps_2hz if n_input_timesteps_2hz is not None else 1024
        self.time_pos_embed = nn.Parameter(torch.randn(1, pe_len, H) * 0.01)

        # --- 可选 subject embedding ---
        if config.subject_embedding and (config.n_subjects is not None):
            self.subject_embed = nn.Embedding(config.n_subjects, H)

        # --- Encoder （插入 MoE）---
        moe_slots = self._parse_moe_slots(config.transformer_depth, config.moe_ffn_where)
        moe_kwargs = dict(
            d_model=H,
            n_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            mult=config.moe_hidden_mult,
            dropout=config.moe_dropout,
            expert_layers=config.moe_expert_layers,
            token_chunk=config.moe_token_chunk,
            shared_dense=config.moe_share_dense,
            share_alpha=config.moe_share_alpha,
        )
        self.encoder = Encoder(
            d_model=H, depth=config.transformer_depth, n_heads=config.n_heads,
            attn_dropout=config.attn_dropout, resid_dropout=config.resid_dropout,
            moe_slots=moe_slots, moe_kwargs=moe_kwargs
        )

        # --- 池化到 TR ---
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # --- subject-conditional head ---
        self.pred_head = SubjectConditionalLinear(H, n_outputs, config.n_subjects)

    # ------ utils ------
    @staticmethod
    def _parse_moe_slots(depth: int, where: str) -> set[int]:
        where = (where or "").strip().lower()
        slots: set[int] = set()
        if not where:
            return slots
        if where.startswith("last"):
            try:
                k = int(where.replace("last", ""))
            except Exception:
                k = 1
            for i in range(depth-1, max(-1, depth-k-1), -1):
                slots.add(i)
            return slots
        if where.startswith("idx:"):
            parts = [p for p in where[4:].split(",") if p.strip()]
            for p in parts:
                try:
                    i = int(p)
                    if 0 <= i < depth:
                        slots.add(i)
                except Exception:
                    pass
            return slots
        # 默认 last1
        slots.add(depth-1)
        return slots

    # ------ data glue ------
    def aggregate_features(self, batch):
        tensors = []
        x0 = next(v for k, v in batch.data.items() if k in self.feature_dims)
        B, T2 = x0.shape[0], x0.shape[-1]

        drop = []
        for m in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                drop.append(m)
        if len(drop) == len(self.feature_dims):
            drop = drop[:-1]

        for modality in self.feature_dims.keys():
            data = batch.data[modality]              # [B,G,D,T]
            data = data.to(torch.float32)
            if data.ndim == 3:
                data = data.unsqueeze(1)
            if self.config.layer_aggregation == "mean":
                data = data.mean(dim=1)             # [B,D,T]
            else:
                data = rearrange(data, "b l d t -> b (l d) t")
            data = data.transpose(1, 2)             # [B,T2,D*]
            proj = self.projectors[modality](data)  # [B,T2,Hm]
            if modality in drop:
                proj = torch.zeros_like(proj)
            tensors.append(proj)

        if self.config.feature_aggregation == "cat":
            x = torch.cat(tensors, dim=-1)          # [B,T2,H]
        else:
            x = sum(tensors)                        # [B,T2,H]
        return x

    def transformer_forward(self, x, subject_id=None):
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed") and (subject_id is not None):
            x = x + self.subject_embed(subject_id).unsqueeze(1)
        x = self.encoder(x)                         # [B,T2,H]
        return x

    # ------ public API used by train script ------
    def forward_features(self, batch) -> torch.Tensor:
        """
        返回 [B, N_TR, H]（池化后的时序表征），保留梯度。
        """
        x = self.aggregate_features(batch)                  # [B,T2,H]
        x = self.transformer_forward(x, batch.data.get("subject_id", None))
        x = x.transpose(1, 2)                               # [B,H,T2]
        x_tr = self.pooler(x).transpose(1, 2)               # [B,N,H]
        return x_tr

    @torch.no_grad()
    def decode_single_subject(self, x_tr: torch.Tensor, sid: int) -> torch.Tensor:
        """
        仅用于快速可视化的无梯度路径；训练里不要用这个！
        """
        B, N, _ = x_tr.shape
        sid_vec = torch.full((B,), sid, dtype=torch.long, device=x_tr.device)
        y = self.pred_head(x_tr, sid_vec)                   # [B,N,O]
        return y

    def decode_all_subjects(self, x_tr: torch.Tensor) -> torch.Tensor:
        """
        训练/验证用，有梯度：返回 [B,4,N,O]
        """
        B, N, _ = x_tr.shape
        outs = []
        for sid in (0,1,2,3):
            sid_vec = torch.full((B,), sid, dtype=torch.long, device=x_tr.device)
            y = self.pred_head(x_tr, sid_vec)               # [B,N,O]
            outs.append(y.unsqueeze(1))                     # [B,1,N,O]
        return torch.cat(outs, dim=1)                       # [B,4,N,O]

    def gather_moe_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        汇总 Encoder 中所有 MoE 的 aux / z；保持张量、可反传。
        """
        aux = torch.tensor(0.0, device=self.time_pos_embed.device)
        z = torch.tensor(0.0, device=self.time_pos_embed.device)
        for m in getattr(self.encoder, "layers", []):
            if isinstance(m, MoEFFNBlock):
                aux = aux + m.moe.aux_loss
                z = z + m.moe.z_loss
        return aux, z

    # 兼容 .forward（按 2Hz 输出）
    def forward(self, batch, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)
        x = self.transformer_forward(x, batch.data.get("subject_id", None))   # [B,T2,H]
        if pool_outputs:
            x_ = x.transpose(1, 2)
            x_tr = self.pooler(x_).transpose(1, 2)                            # [B,N,H]
            B, N, _ = x_tr.shape
            sid = batch.data.get("subject_id", None)
            y = self.pred_head(x_tr, sid)                                     # [B,N,O]
            return y.transpose(1, 2)                                          # [B,O,N]
        else:
            B, T2, _ = x.shape
            sid = batch.data.get("subject_id", None)
            y2 = self.pred_head(x, sid)                                       # [B,T2,O]
            return y2.transpose(1, 2)                                         # [B,O,T2]