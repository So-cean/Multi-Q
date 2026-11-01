"""Shared validation/test masking; targets remain in the base intensity domain."""

from __future__ import annotations

import numpy as np


def mask_for_evaluation(prediction, target, source, policy="intersection"):
    """Use a method-independent mask without fitting any evaluation transform."""
    arrays = [np.asarray(x) for x in (prediction, target, source)]
    if any(x.ndim != 3 or x.shape != arrays[0].shape for x in arrays):
        raise ValueError("Evaluation requires matching 3D grids")
    if any(not np.isfinite(x).all() for x in arrays):
        raise ValueError("Evaluation inputs must be finite")
    if policy == "none":
        return arrays[0], arrays[1]
    if policy != "intersection":
        raise ValueError(f"Unknown evaluation mask: {policy}")
    mask = (arrays[1] > 0) & (arrays[2] > 0)
    if not mask.any():
        raise ValueError("Empty shared foreground")
    return np.where(mask, arrays[0], 0.0), np.where(mask, arrays[1], 0.0)
