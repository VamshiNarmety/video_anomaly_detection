import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate that computes a spatial attention map from a skip feature `x`
    and a gating feature `g` (here: memory read). It returns the gated skip
    feature (x * alpha) and the downsampled attention map at `g`'s spatial size.

    This implements a lightweight Attention U-Net style gate.
    """
    def __init__(self, F_g, F_l, F_int=None):
        super(AttentionGate, self).__init__()
        if F_int is None:
            F_int = max(1, F_l // 2)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        """
        x: skip feature tensor (B, F_l, Hx, Wx)
        g: gating tensor (B, F_g, Hg, Wg)

        returns: gated_x (B, F_l, Hx, Wx), alpha_down (B,1,Hg,Wg)
        """
        # Project
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample gating projection to match x spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        f = self.relu(g1 + x1)
        alpha = self.psi(f)  # (B,1,Hx,Wx)

        # Downsample alpha to gating spatial size for gating the memory read
        if alpha.shape[2:] != g.shape[2:]:
            alpha_down = F.interpolate(alpha, size=g.shape[2:], mode='bilinear', align_corners=False)
        else:
            alpha_down = alpha

        gated_x = x * alpha
        return gated_x, alpha_down
