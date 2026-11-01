"""Coordinate input and bounded residual output used by the Multi-Q UNet++."""

from __future__ import annotations
import torch
from torch import nn


class ResidualSynthesisBase(nn.Module):
    """Apply a gated logit residual to the calibrated source intensity."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channel=0,
        use_coordinates=True,
        use_global_intensity=False,
        residual_scale=1.0,
        zero_initialize_residual=False,
        initial_residual_gate=1.0,
        output_mode="logit_residual",
        logit_epsilon=0.001,
        require_support_mask=True,
    ):
        super().__init__()
        if use_global_intensity or output_mode != "logit_residual":
            raise ValueError("Multi-Q uses a logit residual with source/grid inputs")
        if (
            not 0 <= base_channel < in_channels
            or base_channel + out_channels > in_channels
        ):
            raise ValueError("The residual base must be within the source channels")
        if not 0 < logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must be in (0, 0.5)")
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.base_channel = int(base_channel)
        self.use_coordinates = bool(use_coordinates)
        self.use_global_intensity = False
        self.residual_scale = float(residual_scale)
        self.output_mode = output_mode
        self.logit_epsilon = float(logit_epsilon)
        self.require_support_mask = bool(require_support_mask)
        if zero_initialize_residual:
            self.residual_gate = nn.Parameter(
                torch.tensor(float(initial_residual_gate))
            )
        else:
            self.register_buffer("residual_gate", torch.tensor(20.0), persistent=False)

    def forward(
        self, source, coordinates=None, residual_multiplier=1.0, support_mask=None
    ):
        if source.ndim != 5 or source.shape[1] != self.in_channels:
            raise ValueError("Source must have the configured BCDHW channel count")
        base = source[:, self.base_channel : self.base_channel + self.out_channels]
        inputs = [source]
        if self.use_coordinates:
            if coordinates is None or coordinates.shape != (
                source.shape[0],
                3,
                *source.shape[2:],
            ):
                raise ValueError("Coordinates must have shape (B,3,D,H,W)")
            inputs.append(coordinates.to(dtype=source.dtype))
        residual = self.backbone(torch.cat(inputs, dim=1))
        if support_mask is None:
            if self.require_support_mask:
                raise ValueError("support_mask is required for logit_residual output")
            support_mask = base > 0
        if support_mask.shape != base.shape:
            raise ValueError("Support mask and output must have the same shape")
        base_logits = torch.logit(
            base.clamp(self.logit_epsilon, 1.0 - self.logit_epsilon)
        )
        correction = self.residual_scale * torch.tanh(self.residual_gate) * residual
        prediction = torch.sigmoid(
            base_logits + float(residual_multiplier) * correction
        )
        return prediction * support_mask.to(dtype=prediction.dtype)
