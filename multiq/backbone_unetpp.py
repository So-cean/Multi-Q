"""Three-dimensional UNet++ backbone for paired residual MRI mapping."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import ResidualSynthesisBase


class _ResidualConvBlock3D(nn.Module):
    """Residual convolution stack used at every UNet++ grid node."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        norm_groups: int,
    ) -> None:
        super().__init__()
        self.projection = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        units = []
        for index in range(blocks):
            unit_in = in_channels if index == 0 else out_channels
            units.append(
                nn.Sequential(
                    nn.Conv3d(unit_in, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, out_channels),
                    nn.PReLU(out_channels),
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(norm_groups, out_channels),
                )
            )
        self.units = nn.ModuleList(units)
        self.activations = nn.ModuleList(nn.PReLU(out_channels) for _ in range(blocks))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for index, (unit, activation) in enumerate(zip(self.units, self.activations)):
            residual = self.projection(features) if index == 0 else features
            features = activation(unit(features) + residual)
        return features


class UNetPlusPlusBackbone3D(nn.Module):
    """Full-resolution 3D UNet++ with nested dense decoder connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        levels: int,
        blocks_per_level: int,
        norm_groups: int,
        deep_supervision: bool = False,
        zero_initialize_output: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        channels = tuple(base_channels * (2**level) for level in range(levels))
        if any(width % norm_groups for width in channels):
            raise ValueError("Every UNet++ width must be divisible by norm_groups")
        self.levels = levels
        self.deep_supervision = bool(deep_supervision)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList(
            _ResidualConvBlock3D(
                in_channels if level == 0 else channels[level - 1],
                channels[level],
                blocks_per_level,
                norm_groups,
            )
            for level in range(levels)
        )
        self.upsamplers = nn.ModuleDict()
        self.nested = nn.ModuleDict()
        for stage in range(1, levels):
            for level in range(levels - stage):
                key = f"{level}_{stage}"
                self.upsamplers[key] = nn.Conv3d(
                    channels[level + 1], channels[level], kernel_size=1
                )
                dense_channels = stage * channels[level] + channels[level]
                self.nested[key] = _ResidualConvBlock3D(
                    dense_channels,
                    channels[level],
                    blocks_per_level,
                    norm_groups,
                )
        head_count = levels - 1 if self.deep_supervision else 1
        self.output_heads = nn.ModuleList(
            nn.Conv3d(channels[0], out_channels, kernel_size=1)
            for _ in range(head_count)
        )
        if zero_initialize_output:
            for head in self.output_heads:
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def _run_block(self, block: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                block,
                inputs,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return block(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        nodes: dict[tuple[int, int], torch.Tensor] = {}
        nodes[(0, 0)] = self._run_block(self.encoders[0], inputs)
        for level in range(1, self.levels):
            nodes[(level, 0)] = self._run_block(
                self.encoders[level], self.pool(nodes[(level - 1, 0)])
            )
        for stage in range(1, self.levels):
            for level in range(self.levels - stage):
                key = f"{level}_{stage}"
                target_size = nodes[(level, 0)].shape[2:]
                upsampled = F.interpolate(
                    nodes[(level + 1, stage - 1)],
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                )
                upsampled = self.upsamplers[key](upsampled)
                dense = [nodes[(level, prior)] for prior in range(stage)]
                nodes[(level, stage)] = self._run_block(
                    self.nested[key], torch.cat([*dense, upsampled], dim=1)
                )
        if self.deep_supervision:
            predictions = [
                head(nodes[(0, stage)])
                for stage, head in enumerate(self.output_heads, start=1)
            ]
            return torch.stack(predictions).mean(dim=0)
        return self.output_heads[0](nodes[(0, self.levels - 1)])


class PairedResidualUNetPlusPlus3D(ResidualSynthesisBase):
    """UNet++ backbone with the shared paired-mapping output contract."""

    def __init__(
        self,
        base_channels: int = 16,
        levels: int = 4,
        blocks_per_level: int = 2,
        norm_groups: int = 1,
        deep_supervision: bool = False,
        activation_checkpointing: bool = False,
        zero_initialize_residual: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(zero_initialize_residual=zero_initialize_residual, **kwargs)
        actual_in = self.in_channels + (3 if self.use_coordinates else 0)
        self.backbone = UNetPlusPlusBackbone3D(
            in_channels=actual_in,
            out_channels=self.out_channels,
            base_channels=base_channels,
            levels=levels,
            blocks_per_level=blocks_per_level,
            norm_groups=norm_groups,
            deep_supervision=deep_supervision,
            activation_checkpointing=activation_checkpointing,
            zero_initialize_output=zero_initialize_residual,
        )
