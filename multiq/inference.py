"""Coordinate-aware 3D Gaussian sliding-window inference."""

from __future__ import annotations
import itertools
from typing import Sequence
import numpy as np
import torch
from .precision import autocast_context
from .data import coordinate_patch, spatial_flip_views


def _starts(full: int, patch: int, overlap: float) -> list[int]:
    if patch == full:
        return [0]
    stride = max(1, int(round(patch * (1.0 - overlap))))
    values = list(range(0, full - patch + 1, stride))
    if values[-1] != full - patch:
        values.append(full - patch)
    return values


def flip_tta_views(mode: str) -> list[tuple[int, ...]]:
    """Return spatial axes for interpolation-free MNI-consistent flip TTA."""
    return list(spatial_flip_views(mode))


def _gaussian_weight(patch_size: Sequence[int]) -> np.ndarray:
    axes = []
    for size in patch_size:
        coordinate = np.linspace(-1.0, 1.0, int(size), dtype=np.float32)
        axes.append(np.exp(-0.5 * (coordinate / 0.35) ** 2))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight /= weight.max()
    return np.maximum(weight, 1e-3).astype(np.float32)[None]


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    volume: np.ndarray,
    patch_size: Sequence[int],
    overlap: float,
    device: torch.device,
    batch_size: int = 1,
    use_coordinates: bool = True,
    residual_multiplier: float = 1.0,
    support_mask: np.ndarray | None = None,
    view_flips: Sequence[Sequence[int]] | None = None,
    view_fusion: str = "intensity",
) -> np.ndarray:
    """Reconstruct one `[0,1]` 3D volume from one or more aligned channels."""
    if volume.ndim == 3:
        channel_volume = volume[None]
    elif volume.ndim == 4:
        channel_volume = volume
    else:
        raise ValueError(f"Expected 3D or channel-first 4D input, got {volume.shape}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    patch_size = tuple(int(v) for v in patch_size)
    normalized_flips = (
        [()]
        if view_flips is None
        else [tuple(int(axis) for axis in axes) for axes in view_flips]
    )
    if not normalized_flips:
        raise ValueError("view_flips must contain at least one view")
    if len(set(normalized_flips)) != len(normalized_flips):
        raise ValueError("view_flips must not contain duplicate views")
    if any(axis not in {0, 1, 2} for axes in normalized_flips for axis in axes):
        raise ValueError("flip axes must be spatial axes 0, 1, or 2")
    if view_fusion not in {"intensity", "logit_residual"}:
        raise ValueError("view_fusion must be 'intensity' or 'logit_residual'")
    original_shape = channel_volume.shape[1:]
    padding = []
    for full, patch in zip(original_shape, patch_size):
        missing = max(0, patch - full)
        before = missing // 2
        padding.append((before, missing - before))
    working_volume = np.pad(
        channel_volume,
        ((0, 0), *padding),
        mode="constant",
    )
    working_support = None
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=np.float32)
        if support_mask.shape != original_shape:
            raise ValueError(
                f"support_mask shape {support_mask.shape} != volume {original_shape}"
            )
        working_support = np.pad(support_mask, padding, mode="constant")
    start_axes = [
        _starts(full, patch, overlap)
        for full, patch in zip(working_volume.shape[1:], patch_size)
    ]
    locations = list(itertools.product(*start_axes))
    weight = _gaussian_weight(patch_size)
    output_channels = int(getattr(model, "out_channels", 1))
    output = np.zeros((output_channels, *working_volume.shape[1:]), dtype=np.float32)
    normalizer = np.zeros_like(output)
    model.eval()
    base_channel = int(getattr(model, "base_channel", 0))
    if not 0 <= base_channel < channel_volume.shape[0]:
        raise ValueError("Model base_channel is outside the input volume")
    for offset in range(0, len(locations), batch_size):
        batch_locations = locations[offset : offset + batch_size]
        sources, coordinates, supports = [], [], []
        for start in batch_locations:
            slices = tuple(
                slice(origin, origin + size) for origin, size in zip(start, patch_size)
            )
            sources.append(np.ascontiguousarray(working_volume[(slice(None), *slices)]))
            if working_support is not None:
                supports.append(np.ascontiguousarray(working_support[slices][None]))
            if use_coordinates:
                original_start = tuple(
                    origin - pad[0] for origin, pad in zip(start, padding)
                )
                coordinates.append(
                    coordinate_patch(original_start, patch_size, original_shape)
                )
        source_tensor = torch.from_numpy(np.stack(sources)).to(device)
        coordinate_tensor = (
            torch.from_numpy(np.stack(coordinates)).to(device)
            if use_coordinates
            else None
        )
        support_tensor = (
            torch.from_numpy(np.stack(supports)).to(device)
            if working_support is not None
            else None
        )
        view_predictions = []
        with autocast_context(device, enabled=device.type == "cuda"):
            for flip_axes in normalized_flips:
                tensor_dims = tuple(axis + 2 for axis in flip_axes)
                source_view = (
                    torch.flip(source_tensor, tensor_dims)
                    if tensor_dims
                    else source_tensor
                )
                coordinate_view = (
                    torch.flip(coordinate_tensor, tensor_dims)
                    if tensor_dims and coordinate_tensor is not None
                    else coordinate_tensor
                )
                support_view = (
                    torch.flip(support_tensor, tensor_dims)
                    if tensor_dims and support_tensor is not None
                    else support_tensor
                )
                model_kwargs = {}
                if support_view is not None:
                    model_kwargs["support_mask"] = support_view
                if residual_multiplier == 1.0:
                    prediction_view = model(
                        source_view,
                        coordinate_view,
                        **model_kwargs,
                    )
                else:
                    prediction_view = model(
                        source_view,
                        coordinate_view,
                        residual_multiplier=residual_multiplier,
                        **model_kwargs,
                    )
                if tensor_dims:
                    prediction_view = torch.flip(prediction_view, tensor_dims)
                view_predictions.append(prediction_view.float())
        prediction_stack = torch.stack(view_predictions)
        if view_fusion == "intensity" or len(view_predictions) == 1:
            prediction_tensor = prediction_stack.mean(dim=0)
        else:
            base = source_tensor[
                :, base_channel : base_channel + output_channels
            ].float()
            epsilon = float(getattr(model, "logit_epsilon", 1e-4))
            base_logits = torch.logit(base.clamp(epsilon, 1.0 - epsilon))
            prediction_logits = torch.logit(
                prediction_stack.clamp(epsilon, 1.0 - epsilon)
            )
            residuals = prediction_logits - base_logits.unsqueeze(0)
            prediction_tensor = torch.sigmoid(base_logits + residuals.mean(dim=0))
        predictions = prediction_tensor.cpu().numpy()
        for prediction, start in zip(predictions, batch_locations):
            slices = (slice(None),) + tuple(
                slice(origin, origin + size) for origin, size in zip(start, patch_size)
            )
            output[slices] += prediction * weight
            normalizer[slices] += weight
    prediction = output / np.maximum(normalizer, 1e-8)
    crop = (slice(None),) + tuple(
        slice(before, before + full)
        for (before, _), full in zip(padding, original_shape)
    )
    prediction = prediction[crop]
    if support_mask is not None:
        prediction *= support_mask[None].astype(np.float32)
    else:
        base_foreground = (
            channel_volume[base_channel : base_channel + output_channels] > 0
        )
        prediction *= base_foreground.astype(np.float32)
    prediction = np.clip(prediction, 0.0, 1.0)
    return prediction[0] if output_channels == 1 else prediction
