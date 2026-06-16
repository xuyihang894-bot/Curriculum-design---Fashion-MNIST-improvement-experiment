"""Color-randomization augmentation used to improve OOD performance.

For each black-and-white training image we draw a random bg color and a
random fg color (different enough), exactly mirroring the test-time
distribution shift. This is the "data augmentation" improvement strategy.
"""
from __future__ import annotations

import torch


def random_color_recolor(batch_bw: torch.Tensor, min_distance: float = 0.30,
                          p: float = 1.0) -> torch.Tensor:
    """Recolor a (B, 3, H, W) black/white batch with random fg/bg colors.

    batch_bw: float tensor in [0, 1]. Pixel is treated as foreground if its
              intensity (mean over channels) > 0.5, otherwise background.
    p: probability of applying recolor per sample (the rest stay B/W).
    min_distance: minimum L1 distance (in [0,3]) between bg and fg colors.
    """
    B, C, H, W = batch_bw.shape
    assert C == 3
    device = batch_bw.device
    mask = (batch_bw.mean(dim=1, keepdim=True) > 0.5).float()  # (B, 1, H, W)

    bg = torch.rand(B, 3, device=device)
    fg = torch.rand(B, 3, device=device)
    for _ in range(6):
        dist = (bg - fg).abs().sum(dim=1)
        too_close = dist < min_distance
        if not too_close.any():
            break
        fg[too_close] = torch.rand(int(too_close.sum().item()), 3, device=device)

    bg = bg.view(B, 3, 1, 1)
    fg = fg.view(B, 3, 1, 1)
    recolored = mask * fg + (1.0 - mask) * bg

    if p < 1.0:
        keep = (torch.rand(B, device=device) < p).view(B, 1, 1, 1).float()
        return keep * recolored + (1.0 - keep) * batch_bw
    return recolored
