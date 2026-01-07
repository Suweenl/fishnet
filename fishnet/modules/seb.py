# fishnet/modules/seb.py
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
#
# ================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ModifiedSEBStrict(nn.Module):
    """
    Strict modified SEB per FISHnet Figure 3:
      - Total 4x (3x3 conv); last 2 are the "added" ones with ReLU (explicit)
      - then 1x1 conv + BatchNorm
      - bilinear upsample to match x_d spatial size
      - if multiple high-level maps exist (d<4), multiply all gates element-wise
      - multiply final gate with x_d element-wise
    """
    def __init__(self, channels_d: int):
        super().__init__()
        c = channels_d

        # Two "original" 3x3 convs (activation not specified by FISHnet text)
        self.orig1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.orig2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

        # Two "added" 3x3 convs with ReLU (explicitly stated)
        self.add3 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.add4 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

        self.conv1 = nn.Conv2d(c, c, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(c)

    def _high_to_gate(self, x_high: torch.Tensor, target_hw) -> torch.Tensor:
        h = self.orig1(x_high)
        h = self.orig2(h)
        h = F.relu(self.add3(h), inplace=True)
        h = F.relu(self.add4(h), inplace=True)
        h = self.conv1(h)
        h = self.bn(h)

        h = F.interpolate(h, size=target_hw, mode="bilinear", align_corners=False)
        return h

    def forward(self, x_d: torch.Tensor, x_deeper: List[torch.Tensor]) -> torch.Tensor:
        """
        x_d: [B, C_d, H, W]
        x_deeper: list of deeper features [x_{d+1}, ..., x_D]
                 IMPORTANT: each x_k should already have channel count == C_d
                 (handled by encoder taps or additional 1x1 adapters).
        """
        if len(x_deeper) == 0:
            return x_d

        target_hw = (x_d.shape[-2], x_d.shape[-1])

        gate = None
        for x_high in x_deeper:
            g = self._high_to_gate(x_high, target_hw)
            gate = g if gate is None else gate * g  # multiply all high-level gates
        return x_d * gate
