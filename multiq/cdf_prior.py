"""Frozen target-population histograms and scalar multi-quantile calibration."""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class CDFPrior:
    """Equal-subject target-cohort foreground intensity distribution."""

    target_histogram: np.ndarray
    full_shape: tuple[int, int, int]
    bins: int
    foreground_threshold: float
    target_subject_count: int
    histogram_slice_range: tuple[int, int] | None
    metadata: dict[str, object]


def histogram_quantiles(
    histogram: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Return continuous bin-center quantiles from a normalized histogram."""
    values = np.asarray(histogram, dtype=np.float64)
    probabilities = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 1 or values.size < 16 or values.sum() <= 0.0:
        raise ValueError("histogram must be a positive 1D array with >=16 bins")
    if (
        probabilities.ndim != 1
        or probabilities.size < 1
        or not np.isfinite(probabilities).all()
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
        or np.any(np.diff(probabilities) <= 0.0)
    ):
        raise ValueError("quantiles must be a strictly increasing sequence in (0, 1)")
    cdf = np.cumsum(values / values.sum())
    centers = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    return np.interp(probabilities, cdf, centers)


def histogram_scale_to_reference(
    source_histogram: np.ndarray,
    reference_histogram: np.ndarray,
    quantiles: Sequence[float],
    scale_range: Sequence[float] = (0.5, 2.0),
) -> float:
    """Fit one zero-preserving scale between two fixed distributions."""
    if len(scale_range) != 2:
        raise ValueError("scale_range must contain [minimum, maximum]")
    low, high = (float(value) for value in scale_range)
    if not 0.0 < low <= high:
        raise ValueError("scale_range must be positive and increasing")
    source_quantiles = histogram_quantiles(source_histogram, quantiles)
    reference_quantiles = histogram_quantiles(reference_histogram, quantiles)
    denominator = float(np.dot(source_quantiles, source_quantiles))
    return float(
        np.clip(
            np.dot(source_quantiles, reference_quantiles) / max(denominator, 1e-12),
            low,
            high,
        )
    )


def scale_volume_to_histogram_reference(
    volume: np.ndarray,
    reference_histogram: np.ndarray,
    *,
    quantiles: Sequence[float],
    histogram_stride: int = 4,
    foreground_threshold: float = 0.0,
    scale_range: Sequence[float] = (0.5, 2.0),
    clip_output: bool = False,
) -> tuple[np.ndarray, float]:
    """Remove subject-level brightness with one zero-preserving robust scale.

    The scale is the least-squares fit over the configured histogram
    landmarks.  Keeping the intercept fixed at zero preserves the strict
    background contract and avoids clipping low-intensity foreground through
    a negative affine offset.
    """
    source_histogram = _normalized_histogram(
        volume,
        bins=int(np.asarray(reference_histogram).size),
        foreground_threshold=float(foreground_threshold),
        stride=int(histogram_stride),
    )
    scale = histogram_scale_to_reference(
        source_histogram,
        reference_histogram,
        quantiles=quantiles,
        scale_range=scale_range,
    )
    support = np.asarray(volume) > float(foreground_threshold)
    transformed = np.asarray(volume, dtype=np.float32) * np.float32(scale)
    if clip_output:
        transformed = np.clip(transformed, 0.0, 1.0)
    scaled = np.where(support, transformed, 0.0)
    return scaled.astype(np.float32, copy=False), scale


def _normalized_histogram(
    volume: np.ndarray,
    *,
    bins: int,
    foreground_threshold: float,
    stride: int,
) -> np.ndarray:
    sampled = volume[::stride, ::stride, ::stride]
    foreground = sampled[sampled > foreground_threshold]
    if foreground.size == 0:
        raise ValueError("CDF subject has no strict foreground samples")
    histogram = np.histogram(foreground, bins=bins, range=(0.0, 1.0))[0]
    histogram = histogram.astype(np.float64)
    return histogram / histogram.sum()


def build_cdf_prior(
    target_paths: Sequence[str | Path],
    *,
    bins: int = 2048,
    histogram_stride: int = 4,
    foreground_threshold: float = 0.0,
    histogram_slice_range: Sequence[int] | None = None,
    affine_tolerance: float = 1e-4,
) -> tuple[CDFPrior, np.ndarray]:
    """Build an equal-subject CDF prior from target-domain volumes."""
    paths = [Path(path) for path in target_paths]
    if not paths:
        raise ValueError("Cannot build a CDF prior from an empty cohort")
    if bins < 16 or histogram_stride < 1:
        raise ValueError("bins must be >=16 and histogram_stride must be positive")
    slice_range = (
        tuple(int(value) for value in histogram_slice_range)
        if histogram_slice_range is not None
        else None
    )
    if slice_range is not None and len(slice_range) != 2:
        raise ValueError("histogram_slice_range must contain [start, end]")

    histogram_sum = np.zeros(bins, dtype=np.float64)
    expected_shape: tuple[int, int, int] | None = None
    expected_affine: np.ndarray | None = None
    for index, path in enumerate(paths, start=1):
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape)
        if len(shape) != 3:
            raise ValueError(f"Expected a 3D target volume, got {shape}: {path}")
        if expected_shape is None:
            expected_shape = shape
            expected_affine = np.asarray(image.affine, dtype=np.float64)
        elif shape != expected_shape or not np.allclose(
            image.affine, expected_affine, rtol=0, atol=affine_tolerance
        ):
            raise ValueError(f"CDF cohort grid mismatch: {path}")
        volume = np.asarray(image.dataobj, dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError(f"Non-finite target intensity: {path}")
        volume = np.clip(volume, 0.0, 1.0)
        if slice_range is not None:
            start, end = slice_range
            if not 0 <= start < end <= shape[2]:
                raise ValueError(f"Invalid histogram slice range for {path}")
            volume = volume[:, :, start:end]
        histogram_sum += _normalized_histogram(
            volume,
            bins=bins,
            foreground_threshold=foreground_threshold,
            stride=histogram_stride,
        )
        print(f"cdf [{index}/{len(paths)}] {path.name}", flush=True)

    assert expected_shape is not None and expected_affine is not None
    prior = CDFPrior(
        target_histogram=(histogram_sum / len(paths)).astype(np.float32),
        full_shape=expected_shape,
        bins=bins,
        foreground_threshold=float(foreground_threshold),
        target_subject_count=len(paths),
        histogram_slice_range=slice_range,
        metadata={},
    )
    return prior, expected_affine


def save_cdf_prior(
    path: str | Path,
    prior: CDFPrior,
    affine: np.ndarray,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save the compact CDF-only format; no spatial atlas arrays are stored."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(prior.metadata)
    payload.update(metadata or {})
    np.savez_compressed(
        destination,
        format=np.asarray("cdf_prior_v1"),
        target_histogram=prior.target_histogram.astype(np.float32),
        full_shape=np.asarray(prior.full_shape, dtype=np.int32),
        bins=np.asarray(prior.bins, dtype=np.int32),
        foreground_threshold=np.asarray(prior.foreground_threshold, dtype=np.float32),
        target_subject_count=np.asarray(prior.target_subject_count, dtype=np.int32),
        histogram_slice_range=np.asarray(
            prior.histogram_slice_range or (), dtype=np.int32
        ),
        affine=np.asarray(affine, dtype=np.float64),
        metadata=np.asarray(json.dumps(payload, sort_keys=True)),
    )


def load_cdf_prior(path: str | Path) -> tuple[CDFPrior, np.ndarray]:
    """Load and validate the compact CDF-only prior format."""
    with np.load(path, allow_pickle=False) as values:
        if str(values["format"]) != "cdf_prior_v1":
            raise ValueError(f"Unsupported CDF prior format: {values['format']}")
        target_histogram = values["target_histogram"].astype(np.float32)
        bins = int(values["bins"])
        if target_histogram.shape != (bins,) or target_histogram.sum() <= 0:
            raise ValueError("Invalid target histogram in CDF prior")
        prior = CDFPrior(
            target_histogram=target_histogram,
            full_shape=tuple(int(value) for value in values["full_shape"]),
            bins=bins,
            foreground_threshold=float(values["foreground_threshold"]),
            target_subject_count=int(values["target_subject_count"]),
            histogram_slice_range=(
                tuple(int(value) for value in values["histogram_slice_range"])
                if values["histogram_slice_range"].size
                else None
            ),
            metadata=json.loads(str(values["metadata"])),
        )
        affine = values["affine"].astype(np.float64)
    return prior, affine
