"""
Lightweight U-Net backbone with EDM-compatible timestep conditioning.
Inspired by standard DDPM U-Nets and public references such as lucidrains' denoising-diffusion-pytorch.
This implementation exposes a forward(x, t, y) signature to be drop-in compatible with EDMPreconditioner.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timesteps (EDM c_noise).

    Produces a vector of length embedding_dim from a (N,) tensor of timesteps.
    """

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim, bias=True),
        )

    def _build_sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = self._build_sinusoidal(t)
        return self.mlp(base)


class ResidualBlock(nn.Module):
    """A simple residual block with GroupNorm and SiLU, optionally modulated by time embedding.
    The time embedding vector is projected and added as a bias after the first conv.
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        # Inject time embedding
        t_bias = self.time_proj(t_emb).view(t_emb.shape[0], -1, 1, 1)
        h = h + t_bias
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class MidAttnBlock(nn.Module):
    """Self-attention at the bottleneck over spatial tokens.
    Uses torch.nn.MultiheadAttention with batch_first.
    """

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        y = y.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, HW, C)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x + y


class Unet(nn.Module):
    """Minimal U-Net for diffusion with time conditioning.

    Signature: forward(x, t, y) -> tensor with same spatial shape and out_channels.
    The label y is accepted for API compatibility but not used.
    """

    def __init__(
        self,
        # lucidrains-style args / requested defaults
        dim: int = 64,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8, 8),
        channels: int = 1,
        out_dim: Optional[int] = 1,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,  # kept for API, unused here
        mid_attn: bool = True,
        attn_heads: int = 4,
        attn_dim_head: int = 64,  # kept for API, unused by torch MHA
        # internal
        time_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = channels
        self.out_channels = out_dim or channels
        self.time_mlp = SinusoidalTimeEmbedding(time_embedding_dim)
        self.groups = resnet_block_groups

        # Channels per level
        levels = [dim * m for m in dim_mults]
        n_levels = len(levels)

        # Encoder
        self.enc_in = nn.Conv2d(self.in_channels, levels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skips_channels = []

        cur_ch = levels[0]
        for i, ch in enumerate(levels):
            self.down_blocks.append(ResidualBlock(cur_ch, ch, time_embedding_dim, groups=self.groups))
            self.skips_channels.append(ch)
            cur_ch = ch
            if i < n_levels - 1:
                self.downsamples.append(Downsample(cur_ch))

        # Bottleneck
        mid_ch = levels[-1]
        self.mid1 = ResidualBlock(cur_ch, mid_ch, time_embedding_dim, groups=self.groups)
        self.mid2 = ResidualBlock(mid_ch, mid_ch, time_embedding_dim, groups=self.groups)
        self.mid_attn = MidAttnBlock(mid_ch, heads=attn_heads) if mid_attn else None

        # Decoder
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(n_levels - 1)):
            self.upsamples.append(Upsample(cur_ch))
            cur_ch = cur_ch  # unchanged by transposed conv
            cat_ch = cur_ch + self.skips_channels[i]
            out_ch = levels[i]
            self.up_blocks.append(ResidualBlock(cat_ch, out_ch, time_embedding_dim, groups=self.groups))
            cur_ch = out_ch

        self.out_norm = nn.GroupNorm(num_groups=self.groups, num_channels=cur_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(cur_ch, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute time embedding once
        t_emb = self.time_mlp(t)

        # Encoder with skips
        h = self.enc_in(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h)

        # Decoder with skip connections
        for i, (up, block) in enumerate(zip(self.upsamples, self.up_blocks)):
            h = up(h)
            skip = skips[-(i + 2)]  # align with corresponding encoder level
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        out = self.out_conv(self.out_act(self.out_norm(h)))
        return out


