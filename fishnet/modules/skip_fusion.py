# fishnet/modules/skip_fusion.py
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

from .sre_strict import SREStrict
from .seb_strict import ModifiedSEBStrict

class FishnetSkipFusion(nn.Module):
    """
    Implements Eq.(3) fusion at decoder depth d:
      y_d = concat( g(Up(y_{d+1})), f(x_d, x_{d+1}..x_D) )
    """
    def __init__(self, depth: int, channels_d: int):
        super().__init__()
        self.depth = depth
        self.channels_d = channels_d

        self.sre = SREStrict(depth=depth, in_ch=channels_d, out_ch=channels_d)
        self.seb = ModifiedSEBStrict(channels_d=channels_d)

    def forward(self, y_next: torch.Tensor, x_d: torch.Tensor, x_deeper: List[torch.Tensor]) -> torch.Tensor:
        # UpSampling(y_{d+1}) : bilinear
        u = F.interpolate(y_next, scale_factor=2, mode="bilinear", align_corners=False)
        g = self.sre(u)
        f = self.seb(x_d, x_deeper)
        return torch.cat([g, f], dim=1)
