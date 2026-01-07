# ================================================================
# Project    : FISHnet PyTorch Reproduction
#
# Author     : Shuwen Liang
#
# Description:
#   Strict PyTorch implementation of the Spatial Resolution Enhancer (SRE)
#   module described in:
#     "FISHnet: Learning to Segment the Silhouettes of Swimmers"
#     (IEEE Access, 2020).
#
#   This implementation follows the paper specification (Table 1, Fig. 5)
#   and is intended for research reproduction and analysis.
#
# Project Init Date : 2025-12-20
# Last Revision     : 2026-01-07
#
# Notes:
#   - This file is part of an academic reproduction project.
#   - Implementation choices are aligned with the original paper text
#     and figures, avoiding undocumented extensions.
#   -核心思想：对 decoder 上采样后的特征做多尺度空洞卷积（dilation rates 按论文 Table 1），把不同感受野的信息拼起来，再用 1×1 卷积融合
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 该字典用于定义 Spatial Resolution Enhancer（SRE）模块在不同解码器
# 深度（decoder depth）下所采用的空洞卷积（atrous convolution）膨胀率，
# 严格对应 FISHnet 论文中的 Table 1。
#
# 键（int）：解码器的深度 d
#   - d = 4 表示最深层的解码阶段，对应最低空间分辨率的特征图
#   - d = 1 表示最浅层的解码阶段，对应最高空间分辨率的特征图
#
# 值（List[int]）：在该深度 d 下并行使用的 3×3 空洞卷积膨胀率（dilation rates）
#
# 设计动机（来源于论文的结构设计思想）：
#   - 在解码器较深层（d 较大）时，特征图空间分辨率较低，
#     因此仅使用较少的空洞卷积分支即可覆盖足够大的感受野。
#   - 在解码器较浅层（d 较小）时，特征图空间分辨率较高，
#     引入更多、且膨胀率更大的空洞卷积分支，
#     有助于在保持分辨率的同时捕获更大范围的上下文信息。
#
# 具体设置如下（与论文 Table 1 完全一致）：
#   d = 4 → [4]
#   d = 3 → [4, 8]
#   d = 2 → [4, 8, 16]
#   d = 1 → [4, 8, 16, 32]
#
# 在前向传播中，所有空洞卷积分支并行作用于同一输入特征图，
# 分支输出在通道维度上进行拼接（concatenate），
# 随后通过一个 1×1 卷积进行融合，形成类似 ASPP 的多尺度上下文表示，
# 用于增强解码阶段的空间分辨率表达能力。
# ------------------------------------------------------------------
SRE_RATES = {
    4: [4],
    3: [4, 8],
    2: [4, 8, 16],
    1: [4, 8, 16, 32],
}

class SREStrict(nn.Module):
    """
    Strict SRE per FISHnet (Figure 5 + Table 1):
      - variable number of parallel atrous conv layers (3x3, dilation=r)
      - followed by a 1x1 conv layer
    No BN/activation added beyond what paper explicitly states.
    """
    def __init__(self, depth: int, in_ch: int, out_ch: int):
        super().__init__()
        if depth not in SRE_RATES:
            raise ValueError(f"depth must be in {list(SRE_RATES.keys())}, got {depth}")
        self.depth = depth
        self.rates = SRE_RATES[depth]

        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=True)
            for r in self.rates
        ])
        self.fuse_1x1 = nn.Conv2d(out_ch * len(self.rates), out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        x_cat = torch.cat(feats, dim=1)
        return self.fuse_1x1(x_cat)
