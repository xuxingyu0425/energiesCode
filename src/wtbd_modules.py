"""Task-specific lightweight modules for the WTBD experiments.

The registration helper swaps Ultralytics' YAML-visible ``C3k2`` symbol for a
backward-compatible subclass. Standard C3k2 layers remain unchanged, while
selected P3/P4 blocks can enable an elongated-strip or large-selective-kernel
residual branch through extra YAML arguments. A lightweight DySample module is
also exposed to the YAML parser. Keeping the original layer indices preserves
pretrained-weight transfer for the rest of YOLO11n.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import C2f as UltralyticsC2f
from ultralytics.nn.modules import C3k2 as UltralyticsC3k2


class ElongatedStripBlock(nn.Module):
    """Lightweight gated local/horizontal/vertical depthwise feature block."""

    def __init__(self, channels: int, kernel_size: int = 7, reduction: int = 4) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        hidden = max(channels // reduction, 8)
        pad = kernel_size // 2

        def depthwise(kernel: tuple[int, int], padding: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel,
                    padding=padding,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        self.local = depthwise((3, 3), (1, 1))
        self.horizontal = depthwise((1, kernel_size), (0, pad))
        self.vertical = depthwise((kernel_size, 1), (pad, 0))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 3, 1, bias=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Equal initial branch weights and an identity residual at initialization
        # make loading YOLO11 pretrained weights stable.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x).softmax(dim=1)
        mixed = (
            weights[:, 0:1] * self.local(x)
            + weights[:, 1:2] * self.horizontal(x)
            + weights[:, 2:3] * self.vertical(x)
        )
        return x + self.project(mixed)


class LargeSelectiveKernelBlock(nn.Module):
    """Identity-stable LSK-style spatial context block.

    The two spatial paths follow the central LSKNet idea: a local depthwise
    kernel and a larger dilated depthwise kernel are selected per spatial
    position. A zero-initialized residual projection makes pretrained YOLO11
    features unchanged before the new block starts learning.
    """

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=9, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.reduce_local = nn.Conv2d(channels, hidden, 1, bias=False)
        self.reduce_context = nn.Conv2d(channels, hidden, 1, bias=False)
        self.spatial_select = nn.Conv2d(2, 2, 7, padding=3, bias=True)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        context = self.context(local)
        local = self.reduce_local(local)
        context = self.reduce_context(context)
        joined = torch.cat((local, context), dim=1)
        selector = torch.cat(
            (joined.mean(dim=1, keepdim=True), joined.amax(dim=1, keepdim=True)),
            dim=1,
        )
        weights = self.spatial_select(selector).sigmoid()
        mixed = weights[:, 0:1] * local + weights[:, 1:2] * context
        return x + self.project(mixed)


class DySample(nn.Module):
    """Ultra-lightweight dynamic 2x upsampler implemented with grid_sample.

    This is the ``lp`` (linear then pixel-shuffle) formulation from DySample.
    It uses standard PyTorch operators and requires no custom CUDA extension.
    """

    def __init__(
        self,
        channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = False,
    ) -> None:
        super().__init__()
        if style != "lp":
            raise ValueError("WTBD DySample currently supports only style='lp'")
        if scale != 2:
            raise ValueError("WTBD feature experiments use a fixed 2x DySample scale")
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        self.scale = int(scale)
        self.groups = int(groups)
        self.offset = nn.Conv2d(channels, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.offset.bias)
        self.scope = nn.Conv2d(channels, 2 * groups * scale * scale, 1, bias=False) if dyscope else None
        if self.scope is not None:
            nn.init.zeros_(self.scope.weight)
        self.register_buffer("init_pos", self._initial_positions(), persistent=True)

    def _initial_positions(self) -> torch.Tensor:
        start = (-self.scale + 1) / (2 * self.scale)
        stop = (self.scale - 1) / (2 * self.scale)
        positions = torch.linspace(start, stop, steps=self.scale)
        mesh = torch.stack(torch.meshgrid(positions, positions, indexing="ij"), dim=0)
        return mesh.transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        base = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"), dim=0)
        base = base.transpose(1, 2).unsqueeze(0).unsqueeze(2)
        normalizer = x.new_tensor([width, height]).view(1, 2, 1, 1, 1)
        coords = 2.0 * (base + offset) / normalizer - 1.0
        coords = F.pixel_shuffle(coords.view(batch, -1, height, width), self.scale)
        coords = coords.view(
            batch,
            2,
            -1,
            self.scale * height,
            self.scale * width,
        ).permute(0, 2, 3, 4, 1)
        coords = coords.contiguous().flatten(0, 1)
        sampled = F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled.view(batch, -1, self.scale * height, self.scale * width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        if self.scope is None:
            offset = 0.25 * offset + self.init_pos
        else:
            offset = 0.5 * offset * self.scope(x).sigmoid() + self.init_pos
        return self._sample(x, offset)


class C3k2ESB(UltralyticsC3k2):
    """C3k2-compatible block with one optional WTBD feature residual branch."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        use_esb: bool = False,
        strip_kernel: int = 7,
        use_lsk: bool = False,
    ) -> None:
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if use_esb and use_lsk:
            raise ValueError("ESB and LSK are competing branches and cannot share one C3k2 block")
        if use_esb:
            self.feature_block = ElongatedStripBlock(c2, strip_kernel)
        elif use_lsk:
            self.feature_block = LargeSelectiveKernelBlock(c2)
        else:
            self.feature_block = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_block(super().forward(x))


class C2fWTBD(UltralyticsC2f):
    """YOLOv8 C2f block with an optional identity-stable ESB or LSK branch.

    Keeping the native C2f parameter names allows the ordinary YOLOv8n
    checkpoint to initialize the complete original path.  Only the additional
    residual branch starts from its identity initialization.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        use_esb: bool = False,
        strip_kernel: int = 7,
        use_lsk: bool = False,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, g, e)
        if use_esb and use_lsk:
            raise ValueError("ESB and LSK cannot share one C2f block")
        if use_esb:
            self.feature_block = ElongatedStripBlock(c2, strip_kernel)
        elif use_lsk:
            self.feature_block = LargeSelectiveKernelBlock(c2)
        else:
            self.feature_block = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_block(super().forward(x))


def register_wtbd_modules() -> None:
    """Expose WTBD feature modules to Ultralytics' YAML parser."""
    import ultralytics.nn.tasks as tasks

    tasks.C2f = C2fWTBD
    tasks.C3k2 = C3k2ESB
    tasks.DySample = DySample


def register_standard_modules() -> None:
    """Restore Ultralytics' native parser symbols for true baselines."""
    import ultralytics.nn.tasks as tasks

    tasks.C2f = UltralyticsC2f
    tasks.C3k2 = UltralyticsC3k2
