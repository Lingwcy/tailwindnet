# Copyright (c) SouthWest Minzu University | ZhongZhiTechLab. All rights reserved.
# Author: Chongyang Wang
from __future__ import annotations

from inspect import isfunction
from typing import List, Sequence, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, einsum, nn
from einops import rearrange, repeat
from mmengine.model import BaseModule
from mmseg.registry import MODELS


def exists(val) -> bool:
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor: 
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    class _DWConv(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
            b, n, c = x.shape
            h, w = hw_shape
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = self.dwconv(x)
            return rearrange(x, 'b c h w -> b (h w) c')

    def __init__(self, embed_dims: int, hidden_dims: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dims, hidden_dims)
        self.dwconv = self._DWConv(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, embed_dims)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(self.dwconv(x, hw_shape))
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        q = rearrange(self.q(x), 'b n (head dim) -> b head n dim', head=self.num_heads)

        if self.sr_ratio > 1:
            h, w = hw_shape
            x_ = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x_ = self.sr(x_)
            x_ = rearrange(x_, 'b c h w -> b (h w) c')
            x_ = self.norm(x_)
        else:
            x_ = x

        k, v = rearrange(
            self.kv(x_),
            'b n (kv head dim) -> kv b head n dim',
            kv=2,
            head=self.num_heads,
        )

        attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(out, 'b head n dim -> b n (head dim)')
        x = self.proj(x)
        return x

"""
    层级空间降采样 Transformer 块
    Hierarchical Spatial-Reduction Transformer Block (HSRT Block)
    We build the backbone with a series of Hierarchical Spatial-Reduction Transformer (HSRT) blocks, 
    which combine spatially-reduced multi-head self-attention with lightweight depthwise-convolutional feed-forward networks.
"""
class HSRTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 4)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x), hw_shape))
        return x

# 下采样模块
# 实现多尺度特征提取能力，通过卷积操作将输入特征图的空间分辨率降低，同时增加通道数
"""
    At each stage, an overlapping convolutional patch embedding downsamples the feature maps and increases the channel dimension, 
    forming a multi-scale token pyramid.
"""
class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dims, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, (h, w)


class FusionReconstructionHead(nn.Module):
    """Reconstruct a fused RGB-like image from stage-1 fused features.

    This head upsamples the low-resolution fused feature map back to the
    original input resolution and maps channels to a small number of
    output channels (e.g. 3 for visualization).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int = 3,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor, out_hw: Tuple[int, int]) -> Tensor:
        """x: B x C x H/4 x W/4, out_hw: (H, W)."""
        h, w = out_hw
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        x = self.blocks(x)
        x = self.proj_out(x)
        if self.use_sigmoid:
            x = x.sigmoid()
        return x

"""
    标准SE变体
    Lightweight Global Channel Attention (LGCA) Block | 轻量全局通道注意力模块
    We attach a Lightweight Global Channel Attention (LGCA) block after the fusion module to rescale feature channels according to global context.
    全局平均池化 + 两层 MLP + Sigmoid，标准 SE 变体。
    用在融合后的特征上，强化有用通道。
"""
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = hw_shape
        feat = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        attn = self.avg_pool(feat).view(b, c)
        attn = self.fc(attn).view(b, c, 1, 1)
        feat = feat * attn
        return rearrange(feat, 'b c h w -> b (h w) c')

# 逐深度卷积
class DWConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, padding, groups=dim)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = hw_shape
        feat = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        feat = self.dwconv(feat)
        return rearrange(feat, 'b c h w -> b (h w) c')

# 逐点卷积
class PWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = hw_shape
        feat = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        feat = self.bn(self.pwconv(feat))
        return rearrange(feat, 'b c h w -> b (h w) c')

# MixFFN 的强扩展版 
# 原理念（线性 + 卷积混合 + 线性）
# 多尺度 DWConv
"""
The feed-forward subnetwork is implemented as a multi-branch depthwise convolutional MLP, 
consisting of parallel depthwise convolutions with kernel sizes 3, 5 and 7, sandwiched between pointwise convolutions
"""
class MixFFN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.pwconv1 = PWConv(channels)
        self.dwconv3 = DWConv(channels, 3)
        self.dwconv5 = DWConv(channels, 5)
        self.dwconv7 = DWConv(channels, 7)
        self.pwconv2 = PWConv(channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, hw_shape)
        x3 = self.dwconv3(x, hw_shape)
        x5 = self.dwconv5(x, hw_shape)
        x7 = self.dwconv7(x, hw_shape)
        x = self.pwconv2(x + x3 + x5 + x7, hw_shape)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

# 交叉注意力模块
class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

# 简单融合
class SimpleFusion(nn.Module):
    """Fallback fusion that averages RGB and NIR features."""

    def forward(self, rgb_feat: Tensor, nir_feat: Tensor) -> Tensor:
        return 0.5 * (rgb_feat + nir_feat)

"""
    Gated Cross-Modal Fusion and Alignment (GCMFA) Block
    门控跨模态融合与对齐模块
    We propose a Gated Cross-Modal Fusion and Alignment (GCMFA) block, 
    which performs bidirectional RGB–NIR cross-attention at a low spatial resolution, 
    generates a shared representation, and writes it back into each modality via learned gates, 
    coupled with a cosine-similarity alignment regularizer.
"""
# 跨模态融合块
class GCMFA(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int,
        attn_heads: int,
        attn_dim_head: int,
        dropout: float,
        sr_ratio: int,
        alignment_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.attn_heads = attn_heads
        self.attn_dim_head = attn_dim_head
        self.sr_ratio = max(sr_ratio, 1)

        self.rgb_norm = nn.LayerNorm(channels)
        self.nir_norm = nn.LayerNorm(channels)

        self.rgb_to_nir = CrossAttention(
            query_dim=channels,
            context_dim=channels,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=dropout,
        )
        self.nir_to_rgb = CrossAttention(
            query_dim=channels,
            context_dim=channels,
            heads=attn_heads,
            dim_head=attn_dim_head,
            dropout=dropout,
        )

        self.fuse_linear = nn.Linear(channels * 2, channels)
        self.shared_act = nn.GELU()
        self.gate_linear = nn.Linear(channels, 2)
        self.merge_linear = nn.Linear(channels * 2, channels)
        self.mix_ffn = MixFFN(channels)
        self.mid_norm = nn.LayerNorm(channels)
        self.channel_attn = ChannelAttentionBlock(channels, reduction)
        self.out_norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = nn.AvgPool2d(self.sr_ratio, self.sr_ratio) if self.sr_ratio > 1 else None
        self.alignment_loss_weight = float(alignment_loss_weight)
        self._alignment_loss = None
        self.register_buffer('zero_tensor', torch.tensor(0.0), persistent=False)

    def forward(self, rgb_feat: Tensor, nir_feat: Tensor) -> Tensor:
        b, c, h, w = rgb_feat.shape
        # 会先对 rgb_feat 和 nir_feat 做一次平均池化，
        # 得到低分辨率版本 rgb_low 和 nir_low。这样可以在更小的空间网格上执行后续的交叉注意力。
        # 如果 sr_ratio == 1，说明不需要降采样，就直接使用原始特征，
        if self.sr_ratio > 1:
            rgb_low = self.pool(rgb_feat)
            nir_low = self.pool(nir_feat)
        else:
            rgb_low = rgb_feat
            nir_low = nir_feat

        low_h, low_w = rgb_low.shape[2:]
        rgb_seq = rearrange(rgb_low, 'b c h w -> b (h w) c')
        nir_seq = rearrange(nir_low, 'b c h w -> b (h w) c')

        rgb_context = self.rgb_norm(rgb_seq)
        nir_context = self.nir_norm(nir_seq)

        if self.alignment_loss_weight > 0.0:
            rgb_unit = F.normalize(rgb_context, dim=-1)
            nir_unit = F.normalize(nir_context, dim=-1)
            cos_sim = torch.clamp((rgb_unit * nir_unit).sum(dim=-1), -1.0, 1.0)
            self._alignment_loss = (1.0 - cos_sim).mean() * self.alignment_loss_weight
        else:
            self._alignment_loss = self.zero_tensor

        rgb_cross = self.rgb_to_nir(rgb_context, context=nir_context)
        nir_cross = self.nir_to_rgb(nir_context, context=rgb_context)

        fused_input = torch.cat([rgb_cross, nir_cross], dim=-1) # shape (b, low_h*low_w, 2*c)
        shared = self.shared_act(self.fuse_linear(fused_input))
        gates = torch.sigmoid(self.gate_linear(shared))
        rgb_gate = gates[..., 0:1]
        nir_gate = gates[..., 1:2]

        rgb_updated = rgb_seq + rgb_gate * (shared - rgb_seq)
        nir_updated = nir_seq + nir_gate * (shared - nir_seq)

        fused_seq = self.merge_linear(torch.cat([rgb_updated, nir_updated], dim=-1))
        fused_seq = fused_seq + shared

        mixed = self.mix_ffn(fused_seq, (low_h, low_w))
        fused_seq = fused_seq + self.dropout(mixed)
        fused_seq = self.mid_norm(fused_seq)

        channel_enhanced = self.channel_attn(fused_seq, (low_h, low_w))
        fused_seq = fused_seq + self.dropout(channel_enhanced)

        fused_seq = self.out_norm(fused_seq)
        fused_low_feat = rearrange(fused_seq, 'b (h w) c -> b c h w', h=low_h, w=low_w)

        if self.sr_ratio > 1:
            fused_high_feat = F.interpolate(
                fused_low_feat, size=(h, w), mode='bilinear', align_corners=False
            )
        else:
            fused_high_feat = fused_low_feat

        fused_high_feat = fused_high_feat + 0.5 * (rgb_feat + nir_feat)
        return fused_high_feat

    def get_alignment_loss(self) -> Tensor:
        if self._alignment_loss is None:
            return self.zero_tensor.to(self.rgb_norm.weight.device)
        return self._alignment_loss

# 转置自注意力
class MDTA(nn.Module):
    """
        Multi-DConv Head Transposed Self-Attention.
        多深度卷积头转置自注意力模块

        In the deepest stage, 
        we employ a Multi-DConv Head Transposed Self-Attention (MDTA) module that performs self-attention on a transposed head–channel space and injects local priors via depthwise convolutions.
        此模块源于经典的图像融合网络Restormer中的设计理念。
        trick: 早期模块负责把 RGB/NIR 组合成语义空间，MDTA 负责在这个语义空间中做精细的通道关系建模和选择
        RGB+NIR 的引入为通道空间注入了额外维度和对生理状态敏感的信号，
        而 MDTA 则是让网络在这种高维模态混合空间上做“场景自适应特征选择”

        eg.
        在某一幅图中，加拿大一枝黄花已经进入营养生长后期，NIR 反射明显且形态突出，那么 MDTA 会偏向那些“高 NIR + 竖直冠层 + 特定纹理”的通道组合；
        在另一幅图中，目标较少而背景道路、建筑物更多，那么那些“低 NIR + 规则边界 + 低纹理”的通道可能被强化，用来更好地识别背景。

    """

    def __init__(self, dim: int, num_heads: int, bias: bool) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads.'
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim * 3,
            bias=bias,
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        b, n, c = x.shape
        h, w = hw_shape
        feat = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        qkv = self.qkv_dwconv(self.qkv(feat))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c_head) h w -> b head c_head (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c_head) h w -> b head c_head (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c_head) h w -> b head c_head (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c_head (h w) -> b (head c_head) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return rearrange(out, 'b c h w -> b (h w) c')

"""
    Gated Depthwise Feed-Forward Network (GDFN)
    门控深度前馈网络
    The MDTA is followed by a Gated Depthwise Feed-Forward Network (GDFN), 
    which expands channels by a factor γ and models high-order channel interactions through depthwise separable convolutions and gated activations.
"""
class GDFN(nn.Module):
    def __init__(self, dim: int, ffn_expansion_factor: float, bias: bool) -> None:
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# MDTA块
class MDTABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: float,
        bias: bool,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GDFN(dim, expansion_factor, bias)

    def forward(self, x: Tensor, hw_shape: Tuple[int, int]) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), hw_shape))

        ff_input = self.norm2(x)
        b, n, c = ff_input.shape
        h, w = hw_shape
        ff_feat = rearrange(ff_input, 'b (h w) c -> b c h w', h=h, w=w)
        ff_feat = self.ffn(ff_feat)
        ff_tokens = rearrange(ff_feat, 'b c h w -> b (h w) c')

        x = x + self.drop_path(ff_tokens)
        return x

# Tailwind 不同体型的配置
tailwind_scale = {
    'B0': ([32, 64, 160, 256], [2, 2, 2, 2]),
    'B1': ([64, 128, 320, 512], [2, 2, 2, 2]),
    'B2': ([64, 128, 320, 512], [3, 4, 6, 3]),
    'B3': ([64, 128, 320, 512], [3, 4, 18, 3]),
}

"""
    TailwindV6：双分支门控跨模态转置注意力骨干网络
    TailwindV6: Dual-Branch Gated Cross-Modal Transposed-Attention Backbone
"""
@MODELS.register_module()
class TailwindV6(BaseModule):
    def __init__(
        self,
        model_name: str = 'B0', # 模型体型
        in_channels: int = 13, # 输入通道数
        out_indices: Sequence[int] = (0, 1, 2, 3), # 输出的阶段索引
        drop_path_rate: float = 0.1, # 随机深度丢弃率
        rgb_channels: int = 3, # RGB通道数
        nir_channels: int = 10, # NIR通道数
        use_mdta: bool = True, # 是否在第四阶段使用 MDTA/GDFN（若 False 则使用基础 HSRTBlock 替代）
        fusion_reduction: int = 16, # 融合模块的通道缩减率
        fusion_heads: int | Sequence[int] | None = None, # 融合模块的注意力头数
        fusion_dim_head: int | None = None, # 融合模块的注意力头维度
        fusion_dropout: float = 0.0, # 融合模块的dropout率
        fusion_sr_ratio: int | None = None, # 融合模块的空间缩减比率
        fusion_alignment_weight: float = 0.0, # 融合模块的对齐度损失权重
        fusion_enable: bool = True, # 是否启用融合模块
        fusion_use_gcmfa: bool = True, # 是否使用 GCMFA；若 False 则改用 SimpleFusion 但仍保留双分支
        mdta_heads: int | None = None, # mdta模块的注意力头数
        mdta_ffn_expansion: float = 2.66, # mdta模块的ffn扩展因子 在 MDTABlock 中，先把输入通道数乘以这个倍率，得到更高维的中间表示，再做深度可分离卷积、激活和压回原通道。
                                          # 在配置里把它调高可以提升 第四阶段 的建模能力，但会增加显存和计算；调低能节约算力，但可能削弱效果。默认值 2.66 是在性能和开销之间的折中。
        mdta_bias: bool = False, # mdta模块的偏置
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert model_name in tailwind_scale, (
            '不受支持的模型型号，请使用 ' + ', '.join(tailwind_scale) + f' | {model_name}'
        )
        assert rgb_channels + nir_channels == in_channels, (
            '输入通道数不匹配, rgb + nir 必须等于 in_channels.'
        )
        # 根据体型配置获取模型配置, 获取各阶段的嵌入维度和深度
        embed_dims, depths = tailwind_scale[model_name]
        self.out_indices = tuple(out_indices)
        self.channels = embed_dims
        self.rgb_channels = rgb_channels
        self.nir_channels = nir_channels
        self.use_mdta = use_mdta
        # 是否启用融合模块
        self.fusion_enabled = fusion_enable and nir_channels > 0
        self.use_gcmfa = fusion_use_gcmfa and self.fusion_enabled
        self.fusion_alignment_weight = (
            float(fusion_alignment_weight) if self.use_gcmfa else 0.0
        )
        self._fusion_alignment_loss = None

        total_depth = sum(depths)
        # 计算每个块的丢弃率
        # 用 torch.linspace 在区间 [0, drop_path_rate] 上平均取 total_depth 个点，得到一个从 0 逐步增大的序列，然后转成列表。
        # total_depth 是所有 Stage 的 Block 数之和，所以这个列表长度和所有 Transformer Block 的数量一致。
        # 这样做的目的是给每个 Block 分配不同的随机深度丢弃概率（Drop Path 概率）：靠前的 Block 使用较低的丢弃率，越往后的 Block 丢弃率越高，实现线性递增的 Drop Path 调度，有助于训练稳定性和性能
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        # 阶段序号累加器
        idx = 0
        stage_heads = [1, 2, 5, 8]
        stage_sr = [8, 4, 2, 1]
        # 取出第一个阶段的丢弃率列表
        stage1_dpr = dpr[idx:idx + depths[0]]
        # 第一阶段下采样率  H/4 × W/4
        self.rgb_patch_embed1 = PatchEmbed(rgb_channels, embed_dims[0], 7, 4, 7 // 2)
        self.rgb_block1 = nn.ModuleList([
            HSRTBlock(embed_dims[0], stage_heads[0], stage_sr[0], stage1_dpr[i])
            for i in range(depths[0])
        ])
        self.rgb_norm1 = nn.LayerNorm(embed_dims[0])

        if self.fusion_enabled:
            self.nir_patch_embed1 = PatchEmbed(nir_channels, embed_dims[0], 7, 4, 7 // 2)
            self.nir_block1 = nn.ModuleList([
                HSRTBlock(embed_dims[0], stage_heads[0], stage_sr[0], stage1_dpr[i])
                for i in range(depths[0])
            ])
            self.nir_norm1 = nn.LayerNorm(embed_dims[0])
        else:
            self.nir_patch_embed1 = None
            self.nir_block1 = nn.ModuleList()
            self.nir_norm1 = None
        idx += depths[0]
        
        # 配置融合模块
        if self.fusion_enabled:
            if fusion_heads is None:
                fusion_heads_val = max(1, embed_dims[0] // 32) # 如果未指定，则根据嵌入维度自动计算
            elif isinstance(fusion_heads, Sequence) and not isinstance(fusion_heads, (str, bytes)): # 如果是序列，取第一个元素
                fusion_heads_val = max(1, int(fusion_heads[0]))
            else:
                fusion_heads_val = max(1, int(fusion_heads)) # 配置文件中直接指定的值

            if fusion_dim_head is None:
                fusion_dim_head_val = max(1, embed_dims[0] // fusion_heads_val) # 如果未指定，则根据嵌入维度和头数自动计算
            else:
                fusion_dim_head_val = max(1, int(fusion_dim_head)) # 配置文件中直接指定的值

            if fusion_sr_ratio is None:
                fusion_sr_ratio_val = 2 # 如果未指定，则默认为2
            elif isinstance(fusion_sr_ratio, Sequence) and not isinstance(fusion_sr_ratio, (str, bytes)):
                fusion_sr_ratio_val = max(1, int(fusion_sr_ratio[0]))
            else:
                fusion_sr_ratio_val = max(1, int(fusion_sr_ratio))

            if self.use_gcmfa:
                self.fusion_block = GCMFA(
                    channels=embed_dims[0],
                    reduction=fusion_reduction,
                    attn_heads=fusion_heads_val,
                    attn_dim_head=fusion_dim_head_val,
                    dropout=fusion_dropout,
                    sr_ratio=fusion_sr_ratio_val,
                    alignment_loss_weight=self.fusion_alignment_weight,
                )
            else:
                self.fusion_block = SimpleFusion()
        else:
            self.fusion_block = SimpleFusion()

        # 第二阶段下采样率  H/8 × W/8
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        stage2_dpr = dpr[idx:idx + depths[1]]
        self.block2 = nn.ModuleList([
            HSRTBlock(embed_dims[1], stage_heads[1], stage_sr[1], stage2_dpr[i])
            for i in range(depths[1])
        ])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        idx += depths[1]

        # 第三阶段下采样率  H/16 × W/16
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        stage3_dpr = dpr[idx:idx + depths[2]]
        self.block3 = nn.ModuleList([
            HSRTBlock(embed_dims[2], stage_heads[2], stage_sr[2], stage3_dpr[i])
            for i in range(depths[2])
        ])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        idx += depths[2]

        # 第四阶段下采样率  H/32 × W/32
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)
        stage4_dpr = dpr[idx:idx + depths[3]]
        # 配置第四阶段块：默认使用 MDTA/GDFN 复合块；当 use_mdta=False 时退回为基础 HSRTBlock（PVT 风格）
        if self.use_mdta:
            heads4 = mdta_heads if mdta_heads is not None else max(1, embed_dims[3] // 32)
            heads4 = int(heads4)
            assert embed_dims[3] % heads4 == 0, '阶段4的维度必须能被mdta_heads整除，这样每个头才能有相同的维度。'
            self.block4 = nn.ModuleList([
                MDTABlock(embed_dims[3], heads4, mdta_ffn_expansion, mdta_bias, stage4_dpr[i])
                for i in range(depths[3])
            ])
        else:
            # 使用基础 HSRTBlock（与前面阶段一致的构造方式）作为第四阶段的替代
            self.block4 = nn.ModuleList([
                HSRTBlock(embed_dims[3], stage_heads[3], stage_sr[3], stage4_dpr[i])
                for i in range(depths[3])
            ])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    # 初始化权重
    # truncated normal initialization 和 He initialization
    def init_weights(self) -> None:
        if self.init_cfg is not None:
            super().init_weights()
            return
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=(2.0 / fan_out) ** 0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        # 取出rgb tensor
        rgb = x[:, :self.rgb_channels, ...]
        rgb_tokens, rgb_hw = self.rgb_patch_embed1(rgb) # shape (b, num_patches/8, embed_dim[0])
        for blk in self.rgb_block1:
            rgb_tokens = blk(rgb_tokens, rgb_hw)
        rgb_tokens = self.rgb_norm1(rgb_tokens)
        rgb_h, rgb_w = rgb_hw
        rgb_feat1 = rearrange(rgb_tokens, 'b (h w) c -> b c h w', h=rgb_h, w=rgb_w)

        if self.fusion_enabled:
            nir = x[:, self.rgb_channels:, ...]
            nir_tokens, nir_hw = self.nir_patch_embed1(nir)
            for blk in self.nir_block1:
                nir_tokens = blk(nir_tokens, nir_hw)
            nir_tokens = self.nir_norm1(nir_tokens)
            nir_h, nir_w = nir_hw
            nir_feat1 = rearrange(nir_tokens, 'b (h w) c -> b c h w', h=nir_h, w=nir_w)
            assert rgb_hw == nir_hw, 'rgb和nir的特征分辨率必须匹配才能进行融合。'
            fused_stage1 = self.fusion_block(rgb_feat1, nir_feat1)
            if self.use_gcmfa:
                self._fusion_alignment_loss = self.fusion_block.get_alignment_loss()
            else:
                self._fusion_alignment_loss = fused_stage1.new_zeros(())
        else:
            fused_stage1 = rgb_feat1
            self._fusion_alignment_loss = fused_stage1.new_zeros(())

        outs: List[Tensor] = []
        if 0 in self.out_indices:
            outs.append(fused_stage1)

        tokens2, hw2 = self.patch_embed2(fused_stage1)
        for blk in self.block2:
            tokens2 = blk(tokens2, hw2)
        tokens2 = self.norm2(tokens2)
        h2, w2 = hw2
        feat2 = rearrange(tokens2, 'b (h w) c -> b c h w', h=h2, w=w2)
        if 1 in self.out_indices:
            outs.append(feat2)

        tokens3, hw3 = self.patch_embed3(feat2)
        for blk in self.block3:
            tokens3 = blk(tokens3, hw3)
        tokens3 = self.norm3(tokens3)
        h3, w3 = hw3
        feat3 = rearrange(tokens3, 'b (h w) c -> b c h w', h=h3, w=w3)
        if 2 in self.out_indices:
            outs.append(feat3)

        tokens4, hw4 = self.patch_embed4(feat3)
        for blk in self.block4:
            tokens4 = blk(tokens4, hw4)
        tokens4 = self.norm4(tokens4)
        h4, w4 = hw4
        feat4 = rearrange(tokens4, 'b (h w) c -> b c h w', h=h4, w=w4)
        if 3 in self.out_indices:
            outs.append(feat4)

        return outs

    def get_fusion_alignment_loss(self) -> Tensor:
        if self._fusion_alignment_loss is None:
            if self.fusion_enabled and self.use_gcmfa:
                return self.fusion_block.zero_tensor.to(self.fusion_block.rgb_norm.weight.device)
            param = next(self.parameters(), None)
            if param is None:
                return torch.tensor(0.0, device=self.rgb_norm1.weight.device)
            return param.new_zeros(())
        return self._fusion_alignment_loss


if __name__ == '__main__':
    backbone = TailwindV6(model_name='B0', in_channels=4, rgb_channels=3, nir_channels=1)
    dummy = torch.randn(1, backbone.rgb_channels + backbone.nir_channels, 256, 256)
    outs = backbone(dummy)
    print([feat.shape for feat in outs])
