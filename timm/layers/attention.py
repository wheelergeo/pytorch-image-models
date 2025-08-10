from typing import Final, Optional, Type, Callable

import math
import os
import torch
from torch import nn as nn
from torch.nn import functional as F

from ._fx import register_notrace_function
from .config import use_fused_attn
from .pos_embed_sincos import apply_rot_embed_cat


@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    # Token Merging
    def bipartite_soft_matching(self, k: torch.Tensor, r: int):
        """
        Input is k from attention, size [batch, tokens, channels].
        """
        # k: [batch_size, num_patches + 1, embed_dim]
        # 对 k 特征维作归一化处理，使其只表示方向而不表示长度，利于相似性计算
        k = k / k.norm(dim=-1, keepdim=True)
        # a: [batch_size, num_patches / 2 + 1, embed_dim]
        # b: [batch_size, num_patches / 2, embed_dim]
        a, b = k[..., ::2, :], k[..., 1::2, :]
        # 避免 r 超过当前层可以合并的最大 token 数
        if a.shape[-2] == 1:
            return None
        r = min(r, b.shape[-2])

        # dot product to get similarity scores
        # scores: [batch_size, num_patches / 2 + 1, num_patches / 2]
        scores = a @ b.transpose(-1, -2)
        # don't merge cls token（以下注释仅假设省略 cls，实际存在）
        # scores[batch_size, num_patches / 2, num_patches / 2]
        scores[..., 0, :] = -math.inf
        
        # node_max: [batch_size, num_patches / 2]，元素值为当前行的最大值
        # node_idx: [batch_size, num_patches / 2]，元素值为当前行最大值的索引
        node_max, node_idx = scores.max(dim=-1)
        # edge_idx: [batch_size, num_patches / 2, 1]
        # argsort 先记录按照当前行号记录当前行索引，后根据降序排列调整分量顺序，分量值替换为之前记录的行索引
        # 通过 [..., None] 将其变为三维张量，方便后续广播操作
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        
        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        # src_idx: [batch_size, r, 1]
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        # src_idx 满足除 dim 维，其余维长度与 node_idx 相同，dim维作为取值的索引
        # gather() 从 node_idx 中取出 src_idx 对应的索引值，得到 src_idx 的实际索引
        # dist_idx: [batch_size, r, 1]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        
        # 由于倒数第二维记录的就是排序前的行数，根据该值进行sort可以恢复为原来的相对位置
        unm_idx = unm_idx.sort(dim=-2)[0]  # Sort cls token back to idx 0
        
        # 上面内容仅用于计算索引，下面定义 merge 函数用于实际的合并操作
        # 通过闭包操作，使得 merge 函数可以访问 unm_idx, src_idx, dst_idx
        def merge(x: torch.Tensor, s_in: torch.Tensor = None):
            """
            input x is of shape [batch, tokens, channels].
            input s_in is of shape [batch, tokens]. 
            """
            src_x, dst_x = x[..., ::2, :], x[..., 1::2, :]
            n, t1, c = src_x.shape
            
            # unm 表示仅取出 src 中无需合并的部分
            # unm: [batch_size, num_patches / 2 + 1 - r, embed_dim]
            unm_x = src_x.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
            
            # src 表示取出 src 中需要合并的部分
            # src: [batch_size, r, embed_dim]
            src_x = src_x.gather(dim=-2, index=src_idx.expand(n, r, c))
            
            # 从 dst 中根据 dst_index 在 dim 维度上索引与 src 进行每个元素相加操作
            # dst: [batch_size, r, embed_dim]
            dst_x = dst_x.scatter_add(dim=-2, index=dst_idx.expand(n, r, c), src=src_x)
            
            
            # 按照 unm, dst 的顺序进行拼接（无需保留原输入 token 顺序）
            if s_in is None:
                return torch.cat([unm_x, dst_x], dim=-2), s_in
            
            src_s, dst_s = s_in[..., ::2], s_in[..., 1::2]
            unm_s = src_s.gather(dim=-1, index=unm_idx.squeeze(-1))
            src_s = src_s.gather(dim=-1, index=src_idx.squeeze(-1))
            dst_s = dst_s.scatter_add(dim=-1, index=dst_idx.squeeze(-1), src=src_s)
            
            return torch.cat([unm_x, dst_x], dim=-2), torch.cat([unm_s, dst_s], dim=-1)
        return merge

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # (batch_size, num_patches + 1, embedding_dim)
        B, N, C = x.shape
        # reshape (batch_size, num_patches + 1, 3, num_heads, head_dim)
        # permute (3, batch_size, num_heads, num_patches + 1, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # q, k, v (batch_size, num_heads, num_patches + 1, head_dim)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Token Merging
        self.merge_fn: Callable = None
        if os.environ.get("TOME_R") not in ("0", None, ""):
            self.merge_fn = self.bipartite_soft_matching(k.mean(dim=1), 
                                                         int(os.environ.get("TOME_R")))
        return x


class AttentionRope(nn.Module):
    """ A Self Attention module with ROPE support.

    Includes options for:
     * QK normalization option
     * Attention output (scale) normalization
     * Fused or unfused QKV projection support
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            num_prefix_tokens: int = 1,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            attn_head_dim: Optional[int] = None,
            norm_layer: Type[nn.Module] = None,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
    ):
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to add a bias term to the query, key, and value projections
            num_prefix_tokens: Number of reg/cls tokens at the beginning of the sequence that
                should not have position embeddings applied
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for the output projection
            attn_head_dim: Dimension of each attention head (if None, computed as dim // num_heads)
            norm_layer: Normalization layer constructor to use for QK and scale normalization
            qk_norm: Enable normalization of query (Q) and key (K) vectors with norm_layer
            scale_norm: Enable normalization (scaling) of attention output with norm_layer
        """
        super().__init__()
        if scale_norm or qk_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        attn_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.fused_attn = use_fused_attn()

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.qkv = None
            self.q_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, attn_dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x,
            rope: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass for the attention module.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            rope: Rotary position embeddings tensor for position-aware attention
            attn_mask: Optional attention mask to apply during attention computation

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        B, N, C = x.shape

        if self.qkv is not None:
            qkv = self.qkv(x)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # B, num_heads, N, C
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)

            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
