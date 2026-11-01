"""Validation metrics aligned with the 30-slice submission scorer."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


class OfficialAxialPerceptualMetric:
    """Batched implementation of the challenge's axial AlexNet LPIPS metric."""

    def __init__(self, device: str, batch_size: int = 4) -> None:
        import lpips
        import torch

        if int(batch_size) < 1:
            raise ValueError("Official LPIPS batch_size must be positive")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = lpips.LPIPS(net="alex").to(self.device).eval()
        self.model.requires_grad_(False)

    def __call__(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        slice_axis: int = 2,
    ) -> float:
        import torch

        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(
                "Official axial LPIPS requires matching 3D prediction and target"
            )
        prediction_slices = np.moveaxis(
            prediction.astype(np.float64, copy=False), slice_axis, 0
        )
        target_slices = np.moveaxis(
            target.astype(np.float64, copy=False), slice_axis, 0
        )
        prediction_slices = prediction_slices * 2.0 - 1.0
        target_slices = target_slices * 2.0 - 1.0
        # Preserve Evaluation/evaluate.py's empty-slice condition exactly.
        keep = np.asarray(
            [np.abs(values).max() >= 1e-10 for values in target_slices],
            dtype=bool,
        )
        if not keep.any():
            return 0.0
        prediction_tensor = (
            torch.from_numpy(np.ascontiguousarray(prediction_slices[keep, None]))
            .float()
            .repeat(1, 3, 1, 1)
        )
        target_tensor = (
            torch.from_numpy(np.ascontiguousarray(target_slices[keep, None]))
            .float()
            .repeat(1, 3, 1, 1)
        )
        total = 0.0
        count = int(prediction_tensor.shape[0])
        with torch.no_grad():
            for start in range(0, count, self.batch_size):
                stop = min(start + self.batch_size, count)
                values = self.model(
                    prediction_tensor[start:stop].to(self.device),
                    target_tensor[start:stop].to(self.device),
                )
                total += float(values.float().sum())
        return total / count


def evaluation_view(
    volume: np.ndarray,
    slice_range: tuple[int, int] | list[int] | None,
    axis: int = 2,
) -> np.ndarray:
    """Select the scorer's half-open axial slab from an original-grid volume."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape={volume.shape}")
    if not 0 <= axis < volume.ndim:
        raise ValueError(f"Invalid slice axis {axis} for shape={volume.shape}")
    if slice_range is None:
        return volume
    if len(slice_range) != 2:
        raise ValueError("slice_range must contain [start, end]")
    start, end = (int(value) for value in slice_range)
    if not 0 <= start < end <= volume.shape[axis]:
        raise ValueError(
            f"slice_range [{start}, {end}) is invalid for axis {axis} "
            f"with length {volume.shape[axis]}"
        )
    selector = [slice(None)] * volume.ndim
    selector[axis] = slice(start, end)
    return volume[tuple(selector)]


def nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    denominator = np.linalg.norm(target)
    return (
        float(np.linalg.norm(prediction - target) / denominator)
        if denominator > 1e-10
        else 0.0
    )


def slice_psnr(prediction: np.ndarray, target: np.ndarray, axis: int = 2) -> float:
    """Average per-slice PSNR exactly as ``Evaluation/evaluate.py`` does."""
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    values = []
    for index in range(target.shape[axis]):
        selector = [slice(None)] * target.ndim
        selector[axis] = index
        pred_slice, target_slice = prediction[tuple(selector)], target[tuple(selector)]
        if target_slice.max() - target_slice.min() < 1e-10:
            continue
        error = np.mean((pred_slice - target_slice) ** 2)
        if error < 1e-10:
            continue
        values.append(20.0 * np.log10(1.0 / np.sqrt(error)))
    return float(np.mean(values)) if values else float("inf")


def foreground_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    foreground = target > 0
    difference = np.abs(prediction - target)
    return float(
        difference[foreground].mean() if foreground.any() else difference.mean()
    )


def gradient_sharpness(volume: np.ndarray) -> float:
    sampled = volume[::2, ::2, ::2]
    return float(
        np.mean([np.abs(np.diff(sampled, axis=axis)).mean() for axis in range(3)])
    )


def scale_target_quantiles_to_reference(
    target: np.ndarray,
    support_source: np.ndarray,
    reference_histogram: np.ndarray,
    quantiles: tuple[float, ...] | list[float],
    foreground_threshold: float = 0.0,
    scale_range: tuple[float, float] | list[float] = (0.5, 2.0),
    clip_output: bool = False,
    statistics_slice_range: tuple[int, int] | list[int] | None = None,
    statistics_axis: int = 2,
) -> np.ndarray:
    """Scale a paired target to one fixed reference distribution without bias.

    The fitted transform is ``target' = a * target``.  ``a`` minimizes the
    squared error between selected foreground quantiles of the target and a
    fixed reference histogram.  No additive offset is allowed,
    so the exact zero background remains the intensity origin.  Values are
    clipped to the model's declared [0, 1] output range after scaling only
    when ``clip_output`` is true. Keeping clipping explicit lets matched
    experiments train against the genuinely linear target ``a * target`` and
    report any out-of-range tail instead of silently truncating it.
    """
    probabilities = np.asarray(quantiles, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or probabilities.size < 1
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
        or np.any(np.diff(probabilities) <= 0.0)
    ):
        raise ValueError("quantiles must be a strictly increasing sequence in (0, 1)")
    histogram = np.asarray(reference_histogram, dtype=np.float64)
    if histogram.ndim != 1 or histogram.size < 16 or histogram.sum() <= 0.0:
        raise ValueError("reference_histogram must be a positive 1D histogram")
    low_scale, high_scale = (float(value) for value in scale_range)
    if not 0.0 < low_scale <= high_scale:
        raise ValueError("scale_range must be positive and increasing")

    target_statistics = evaluation_view(target, statistics_slice_range, statistics_axis)
    support_statistics = evaluation_view(
        support_source, statistics_slice_range, statistics_axis
    )
    mask = (target_statistics > foreground_threshold) & (
        support_statistics > foreground_threshold
    )
    if int(mask.sum()) < 2:
        return target.astype(np.float32, copy=True)

    normalized_histogram = histogram / histogram.sum()
    reference_cdf = np.cumsum(normalized_histogram)
    bin_centers = (np.arange(histogram.size, dtype=np.float64) + 0.5) / histogram.size
    reference_quantiles = np.interp(probabilities, reference_cdf, bin_centers)
    return scale_target_quantiles_to_landmarks(
        target,
        support_source,
        reference_quantiles,
        quantiles=probabilities,
        foreground_threshold=foreground_threshold,
        scale_range=scale_range,
        clip_output=clip_output,
        statistics_slice_range=statistics_slice_range,
        statistics_axis=statistics_axis,
    )


def scale_target_quantiles_to_landmarks(
    target: np.ndarray,
    support_source: np.ndarray,
    reference_quantiles: np.ndarray | tuple[float, ...] | list[float],
    quantiles: np.ndarray | tuple[float, ...] | list[float],
    foreground_threshold: float = 0.0,
    scale_range: tuple[float, float] | list[float] = (0.5, 2.0),
    clip_output: bool = False,
    statistics_slice_range: tuple[int, int] | list[int] | None = None,
    statistics_axis: int = 2,
) -> np.ndarray:
    """Scale a target to explicit, frozen quantile landmarks.

    Explicit landmarks make a fold's prospective target contract completely
    self-contained: no target cohort is opened at training time, and the
    configuration records exactly which output intensity domain is learned.
    """
    if target.shape != support_source.shape:
        raise ValueError("target and support_source must have matching shapes")
    probabilities = np.asarray(quantiles, dtype=np.float64)
    landmarks = np.asarray(reference_quantiles, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or probabilities.size < 1
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
        or np.any(np.diff(probabilities) <= 0.0)
    ):
        raise ValueError("quantiles must be a strictly increasing sequence in (0, 1)")
    if (
        landmarks.shape != probabilities.shape
        or not np.isfinite(landmarks).all()
        or np.any(landmarks <= float(foreground_threshold))
        or np.any(np.diff(landmarks) <= 0.0)
    ):
        raise ValueError(
            "reference_quantiles must be finite, strictly increasing foreground "
            "landmarks matching quantiles"
        )
    low_scale, high_scale = (float(value) for value in scale_range)
    if not 0.0 < low_scale <= high_scale:
        raise ValueError("scale_range must be positive and increasing")

    target_statistics = evaluation_view(target, statistics_slice_range, statistics_axis)
    support_statistics = evaluation_view(
        support_source, statistics_slice_range, statistics_axis
    )
    mask = (target_statistics > foreground_threshold) & (
        support_statistics > foreground_threshold
    )
    if int(mask.sum()) < 2:
        return target.astype(np.float32, copy=True)

    target_quantiles = np.quantile(target_statistics[mask], probabilities)
    denominator = float(np.dot(target_quantiles, target_quantiles))
    scale = np.clip(
        float(np.dot(target_quantiles, landmarks)) / max(denominator, 1e-12),
        low_scale,
        high_scale,
    )
    scaled = target.astype(np.float32) * np.float32(scale)
    if clip_output:
        scaled = np.clip(scaled, 0.0, 1.0)
    return np.where(target > foreground_threshold, scaled, 0.0).astype(np.float32)


def slice_ssim(prediction: np.ndarray, target: np.ndarray, axis: int = 2) -> float:
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    data_range = float(target.max() - target.min())
    if data_range < 1e-10:
        return 1.0
    values = []
    for index in range(target.shape[axis]):
        selector = [slice(None)] * 3
        selector[axis] = index
        pred_slice, target_slice = prediction[tuple(selector)], target[tuple(selector)]
        if target_slice.max() - target_slice.min() < 1e-10:
            continue
        values.append(
            structural_similarity(pred_slice, target_slice, data_range=data_range)
        )
    return float(np.mean(values)) if values else 1.0


def evaluate_volume(
    prediction: np.ndarray,
    target: np.ndarray,
    slice_range: tuple[int, int] | list[int] | None = None,
    slice_axis: int = 2,
) -> dict[str, float]:
    """Score one volume, optionally on original-grid ``[z0, z1)`` only.

    The challenge submission builder crops full predictions to ``[:, :,
    150:180]`` before scoring. Passing that range here makes LOSO checkpoint
    selection see exactly the same voxel slab as the packaged prediction.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction={prediction.shape}, target={target.shape}"
        )
    prediction = evaluation_view(prediction, slice_range, slice_axis)
    target = evaluation_view(target, slice_range, slice_axis)
    ssim_value = slice_ssim(prediction, target, slice_axis)
    nrmse_value = nrmse(prediction, target)
    prediction_sharpness = gradient_sharpness(prediction)
    target_sharpness = gradient_sharpness(target)
    return {
        "ssim": ssim_value,
        "nrmse": nrmse_value,
        "psnr": slice_psnr(prediction, target, slice_axis),
        "foreground_mae": foreground_mae(prediction, target),
        "prediction_sharpness": prediction_sharpness,
        "target_sharpness": target_sharpness,
        "sharpness_ratio": prediction_sharpness / max(target_sharpness, 1e-8),
        "ssim_minus_nrmse": ssim_value - nrmse_value,
    }
