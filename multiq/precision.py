"""Mixed-precision configuration shared by training and memory probes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import torch


def make_grad_scaler(device: torch.device, *, enabled: bool) -> Any:
    """Construct a GradScaler across the PyTorch 2.2 and 2.3+ APIs.

    ``torch.amp.GradScaler`` was added after the CUDA-only scaler.  The GPU
    accounts intentionally use more than one supported PyTorch minor release,
    so selecting by API availability is safer than parsing version strings.
    """

    amp_module = getattr(torch, "amp", None)
    modern_scaler = getattr(amp_module, "GradScaler", None)
    if modern_scaler is not None:
        return modern_scaler(device.type, enabled=bool(enabled))
    if device.type == "cuda" or not enabled:
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))
    raise RuntimeError(
        f"Gradient scaling is enabled for unsupported device type {device.type!r}"
    )


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    dtype: torch.dtype | None = None,
) -> Any:
    """Return an autocast context compatible with modern and legacy PyTorch."""

    generic_autocast = getattr(torch, "autocast", None)
    if generic_autocast is not None:
        return generic_autocast(
            device_type=device.type,
            enabled=bool(enabled),
            dtype=dtype,
        )
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=bool(enabled), dtype=dtype)
    if enabled:
        raise RuntimeError(
            f"Autocast is enabled for unsupported device type {device.type!r}"
        )
    return nullcontext()


def resolve_amp(
    training: Mapping[str, Any], device: torch.device
) -> tuple[bool, torch.dtype, bool]:
    """Resolve autocast and scaler settings without changing legacy defaults.

    Configs that omit ``amp_dtype`` retain CUDA FP16.  BF16 is opt-in because
    changing the implicit dtype of historical experiments would make their
    resumes non-reproducible.  Gradient scaling is useful for FP16 but is
    unnecessary for BF16's wider exponent range.
    """

    names = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    name = str(training.get("amp_dtype", "float16")).lower()
    try:
        dtype = names[name]
    except KeyError as error:
        raise ValueError(
            "training.amp_dtype must be one of: float16, fp16, bfloat16, bf16"
        ) from error

    enabled = bool(training.get("amp", False)) and device.type == "cuda"
    if enabled and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured CUDA device does not support BF16 autocast")
    scaler_enabled = enabled and dtype is torch.float16
    return enabled, dtype, scaler_enabled
