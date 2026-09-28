"""A U-Net whose encoder runs a ConvGRU over time at every scale.

Optical flow already handles where storms move. This network learns what they do next:
form, grow or decay. The lead time is an input, so one network serves every lead from
10 to 120 minutes. Each sensor group comes with an availability mask, so the model can be
trained to keep working when a radar or the satellite feed is missing.
"""

from __future__ import annotations

import math

import torch
from torch import nn

OUTPUTS = ("lightning within 10 km", "echo of at least 40 dBZ")


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.gates = nn.Conv2d(2 * channels, 2 * channels, kernel, padding=pad)
        self.candidate = nn.Conv2d(2 * channels, channels, kernel, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros_like(x)
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.candidate(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


def _block(c_in: int, c_out: int) -> nn.Sequential:
    groups = 8 if c_out % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1),
        nn.GroupNorm(groups, c_out),
        nn.SiLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1),
        nn.GroupNorm(groups, c_out),
        nn.SiLU(),
    )


def lead_embedding(lead_min: torch.Tensor, dim: int, max_minutes: float = 120.0) -> torch.Tensor:
    """Sinusoidal code for the lead time in minutes."""
    half = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(100.0), half, device=lead_min.device))
    x = (lead_min.float() / max_minutes)[:, None] * freqs[None]
    return torch.cat([torch.sin(x), torch.cos(x)], -1)


class FiLM(nn.Module):
    """Scales and shifts feature maps according to the lead time."""

    def __init__(self, channels: int, embed: int):
        super().__init__()
        self.net = nn.Linear(embed, 2 * channels)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(e).chunk(2, -1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class VajraNet(nn.Module):
    """``groups`` maps each sensor group to its channel count, in input order,
    e.g. {"radar": 1, "satellite": 2, "lightning": 1, "model": 4}.
    """

    def __init__(self, groups: dict[str, int], widths=(32, 64, 128, 256), n_out: int = 2, embed: int = 64):
        super().__init__()
        self.groups = dict(groups)
        self.embed = embed
        self.register_buffer("group_sizes", torch.tensor(list(self.groups.values())), persistent=False)
        prev = sum(self.groups.values()) + len(self.groups)  # data plus one availability mask per group
        self.enc = nn.ModuleList()
        self.gru = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(_block(prev, w) if i == 0 else nn.Sequential(nn.MaxPool2d(2), _block(prev, w)))
            self.gru.append(ConvGRUCell(w))
            prev = w
        self.film_mid = FiLM(widths[-1], embed)
        self.up, self.dec, self.film = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(_block(2 * widths[i - 1], widths[i - 1]))
            self.film.append(FiLM(widths[i - 1], embed))
        self.head = nn.Conv2d(widths[0], n_out, 1)
        self.levels = len(widths)

    def forward(
        self, x: torch.Tensor, lead_min: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``x``: (batch, time, channels, H, W), channels in ``groups`` order; H and W divisible by
        2 ** (levels - 1). ``lead_min``: (batch,) minutes. ``available``: (batch, n_groups), 1 = present.

        Returns logits shaped (batch, n_out, H, W), in the order of ``OUTPUTS``.
        """
        b, t, _, h, w = x.shape
        n_g = len(self.groups)
        if h % 2 ** (self.levels - 1) or w % 2 ** (self.levels - 1):
            raise ValueError(f"H and W must be divisible by {2 ** (self.levels - 1)}")
        if available is None:
            available = x.new_ones(b, n_g)
        channel_on = torch.repeat_interleave(available, self.group_sizes, dim=1)
        x = x * channel_on[:, None, :, None, None]
        x = torch.cat([x, available[:, None, :, None, None].expand(b, t, n_g, h, w)], 2)
        hidden: list[torch.Tensor | None] = [None] * self.levels
        for step in range(t):
            f = x[:, step]
            for i, (enc, gru) in enumerate(zip(self.enc, self.gru, strict=True)):
                f = enc(f)
                hidden[i] = gru(f, hidden[i])
        e = lead_embedding(lead_min, self.embed)
        y = self.film_mid(hidden[-1], e)
        for j, (up, dec, film) in enumerate(zip(self.up, self.dec, self.film, strict=True)):
            y = film(dec(torch.cat([up(y), hidden[-2 - j]], 1)), e)
        return self.head(y)
