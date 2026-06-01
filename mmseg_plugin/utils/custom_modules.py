import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
import antialiased_cnns

class DRFD(nn.Module):
    """
    双路径特征下采样模块 (Dual-Route Feature Downsampling)
    通过卷积和最大池化两条路径对特征进行下采样，然后融合特征
    """
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        # 深度可分离卷积，将通道数扩展到2倍
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        # 卷积路径：步长为2的下采样卷积
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, dim * 2)[1]
        # 最大池化路径：步长为2的下采样池化
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, dim * 2)[1]
        # 特征融合层
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):  # x = [B, C, H, W]
        # 特征提取
        shortcut = x
        x = self.conv(x)
        # 卷积路径
        x_c = self.conv_c(x)
        x_c = self.norm_c(x_c)
        x_c = self.act_c(x_c)
        # 最大池化路径
        x_m = self.max_m(x)
        x_m = self.norm_m(x_m)
        # 特征融合
        x = torch.cat((x_c, x_m), dim=1)
        x = self.fusion(x)
        return x

class PA(nn.Module):
    """
    像素级注意力模块 (Pixel Attention)
    通过逐点卷积和门控机制实现像素级的注意力增强
    """
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        # 像素级注意力计算：1x1卷积 -> 批归一化 -> 激活 -> 1x1卷积
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            build_norm_layer(norm_layer, dim * 4)[1],
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        # Sigmoid门控函数
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        # 注意力计算
        attn = self.p_conv(x)
        attn = self.gate_fn(attn)
        # 应用注意力
        x = x * attn
        return x

class LA(nn.Module):
    """
    局部注意力模块 (Local Attention)
    通过3x3卷积捕获局部空间信息
    """
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        # 3x3卷积提取局部特征
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            build_norm_layer(norm_layer, dim)[1],
            act_layer()
        )

    def forward(self, x):
        # 局部特征提取
        x = self.conv(x)
        return x

class MRA(nn.Module):
    """
    多分辨率注意力模块 (Multi-Resolution Attention)
    通过不同方向的条形卷积和多尺度池化实现多分辨率注意力机制
    """
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        # 多尺度池化
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = antialiased_cnns.BlurPool(channel, stride=3)
        # 水平和垂直方向的条形卷积
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, x):
        # 特征变换
        x1 = self.h_transform(x)
        x2 = self.v_transform(x)
        # 注意力计算
        attn1 = self.H_att1(x1) + self.V_att1(x1)
        attn2 = self.H_att2(x2) + self.V_att2(x2)
        attn = attn1 + attn2
        attn = self.norm(attn)
        attn = self.gate_fn(attn)
        # 应用注意力
        x = x * attn
        return x

    def h_transform(self, x):
        """水平变换：将特征图进行水平方向的变换以增强条形卷积效果"""
        return x

    def inv_h_transform(self, x):
        """水平逆变换：恢复水平变换后的特征图"""
        return x

    def v_transform(self, x):
        """垂直变换：将特征图进行垂直方向的变换以增强条形卷积效果"""
        return x

    def inv_v_transform(self, x):
        """垂直逆变换：恢复垂直变换后的特征图"""
        return x

class GA12(nn.Module):
    """
    全局注意力模块12 (Global Attention 12)
    通过下采样-处理-上采样的方式实现高效的全局注意力机制
    结合空间卷积和通道注意力
    """
    def __init__(self, dim, act_layer):
        super().__init__()
        # 最大池化下采样（保存索引用于上采样）
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)
        # 投影层
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        # 空间卷积层
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 通道分离层
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 注意力压缩层
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # 下采样
        x, indices = self.downpool(x)
        # 特征提取与注意力计算
        x0 = self.conv0(x)
        x1 = self.conv_spatial(x)
        x = x0 + x1
        x = self.activation(x)
        # 通道分离与注意力压缩
        b, c, h, w = x.size()
        x1 = self.conv1(x).view(b, -1, h, w)
        x2 = self.conv2(x).view(b, -1, h, w)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_squeeze(x)
        x = self.conv(x)
        # 上采样
        x = self.uppool(x, indices)
        x = self.proj_2(x)
        return x

class D_GA(nn.Module):
    """
    下采样全局注意力模块 (Downsampled Global Attention)
    结合下采样和全局注意力机制，减少计算复杂度
    """
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.attn = GA(dim)
        # 下采样和上采样操作
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.uppool = nn.MaxUnpool2d((2, 2), 2, padding=0)

    def forward(self, x):
        # 下采样
        x, indices = self.downpool(x)
        # 全局注意力计算
        x = self.attn(x)
        # 上采样
        x = self.uppool(x, indices)
        x = self.norm(x)
        return x

class GA(nn.Module):
    """
    全局注意力模块 (Global Attention)
    基于自注意力机制的全局特征交互模块
    使用多头注意力机制捕获长距离依赖关系
    """
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # 计算注意力头数
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        # QKV线性变换层
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 输出投影层
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # QKV计算
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # 特征聚合
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x