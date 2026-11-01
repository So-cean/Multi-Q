"""Paired MRI volumes, independent scalar calibration and 3D patch sampling."""

from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence
import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset
from monai.transforms import Compose, EnsureTyped, LoadImaged, LoadImage, Orientationd
from torch.utils.data import BatchSampler, Dataset
from .cdf_prior import CDFPrior, load_cdf_prior, scale_volume_to_histogram_reference
from .metrics import (
    scale_target_quantiles_to_landmarks,
    scale_target_quantiles_to_reference,
)


def spatial_flip_views(mode: str) -> tuple[tuple[int, ...], ...]:
    """Return interpolation-free spatial flip views.

    Axes ``0/1/2`` refer to the three spatial dimensions after the channel
    dimension.  ``flip8`` is the complete reflection group for a 3D patch.
    """
    choices = {
        "none": ((),),
        "flip-x": ((), (0,)),
        "flip-y": ((), (1,)),
        "flip-z": ((), (2,)),
        "flip-xy": ((), (0, 1)),
        "flip4": ((), (0,), (1,), (0, 1)),
        "flip8": (
            (),
            (0,),
            (1,),
            (2,),
            (0, 1),
            (0, 2),
            (1, 2),
            (0, 1, 2),
        ),
    }
    try:
        return choices[str(mode)]
    except KeyError as error:
        raise ValueError(f"Unsupported spatial flip mode: {mode!r}") from error


def flip_channel_first_spatial(
    array: np.ndarray, spatial_axes: Sequence[int]
) -> np.ndarray:
    """Flip channel-first data and return a Torch-compatible contiguous copy."""
    axes = tuple(int(axis) for axis in spatial_axes)
    if len(set(axes)) != len(axes) or any(axis not in {0, 1, 2} for axis in axes):
        raise ValueError("spatial_axes must contain unique axes from 0, 1, and 2")
    if array.ndim != 4:
        raise ValueError(f"Expected a channel-first 4D array, got {array.shape}")
    return np.ascontiguousarray(np.flip(array, axis=tuple(axis + 1 for axis in axes)))


def source_foreground_mask(volume: np.ndarray, source_channel: int = 0) -> np.ndarray:
    """Return the current subject's strict nonzero foreground mask."""
    channel_volume = volume[None] if volume.ndim == 3 else volume
    if channel_volume.ndim != 4:
        raise ValueError(f"Expected 3D or channel-first 4D input, got {volume.shape}")
    source_channel = int(source_channel)
    if not 0 <= source_channel < channel_volume.shape[0]:
        raise ValueError(
            f"source_channel={source_channel} is outside {channel_volume.shape[0]} channels"
        )
    return np.ascontiguousarray(
        channel_volume[source_channel] > 0,
        dtype=np.float32,
    )


def apply_paired_foreground_intersection(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_channel: int = 0,
    foreground_threshold: float = 0.0,
    target_support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask an aligned training pair to its strict shared foreground.

    This operation is deliberately training-only: it uses the paired target to
    remove source-only and target-only boundary voxels before patch sampling.
    Deployment still derives its output support exclusively from the available
    source volume.
    """
    channel_source = source[None] if source.ndim == 3 else source
    if channel_source.ndim != 4 or target.ndim != 3:
        raise ValueError(
            "Expected channel-first 4D source and 3D target, got "
            f"{source.shape} and {target.shape}"
        )
    source_channel = int(source_channel)
    if not 0 <= source_channel < channel_source.shape[0]:
        raise ValueError(
            f"source_channel={source_channel} is outside "
            f"{channel_source.shape[0]} channels"
        )
    if channel_source.shape[1:] != target.shape:
        raise ValueError(
            f"Source/target shape mismatch: {channel_source.shape[1:]} vs "
            f"{target.shape}"
        )
    threshold = float(foreground_threshold)
    if target_support is None:
        target_support = target > threshold
    else:
        target_support = np.asarray(target_support, dtype=bool)
        if target_support.shape != target.shape:
            raise ValueError("target_support must have the same shape as target")
    support = (channel_source[source_channel] > threshold) & target_support
    support = np.ascontiguousarray(support, dtype=np.float32)
    masked_source = np.ascontiguousarray(
        channel_source * support[None], dtype=np.float32
    )
    masked_target = np.ascontiguousarray(target * support, dtype=np.float32)
    return masked_source, masked_target, support


@dataclass(frozen=True)
class VolumePair:
    subject_id: str
    source: Path
    target: Path
    shape: tuple[int, int, int]
    auxiliary_sources: tuple[Path, ...] = ()
    source_field: str = ""

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Primary source followed by optional aligned auxiliary modalities."""
        return (self.source, *self.auxiliary_sources)


def _volume_path(
    root: Path, split: str, modality: str, field: str, subject_id: str
) -> Path:
    return root / split / modality / field / f"P_{modality}_{field}_{subject_id}.nii.gz"


def discover_pairs(
    root: str | Path,
    split: str,
    modality: str,
    source_field: str,
    target_field: str,
    subject_ids: Iterable[str],
    affine_tolerance: float = 1e-4,
    source_modalities: Iterable[str] | None = None,
) -> list[VolumePair]:
    """Resolve pairs and reject any shape/affine mismatch before training."""
    root = Path(root)
    modalities = list(source_modalities or [modality])
    if not modalities or modalities[0] != modality:
        raise ValueError("source_modalities must begin with the target modality")
    if len(set(modalities)) != len(modalities):
        raise ValueError("source_modalities contains duplicates")
    pairs: list[VolumePair] = []
    for raw_subject_id in subject_ids:
        subject_id = str(raw_subject_id).zfill(4)
        source = _volume_path(root, split, modality, source_field, subject_id)
        auxiliary_sources = tuple(
            _volume_path(root, split, source_modality, source_field, subject_id)
            for source_modality in modalities[1:]
        )
        target = _volume_path(root, split, modality, target_field, subject_id)
        missing = [
            str(path)
            for path in (source, *auxiliary_sources, target)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing paired volume(s): " + ", ".join(missing))
        source_image = nib.load(str(source))
        target_image = nib.load(str(target))
        if source_image.shape != target_image.shape:
            raise ValueError(
                f"Shape mismatch for {subject_id}: {source_image.shape} vs {target_image.shape}"
            )
        if len(source_image.shape) != 3:
            raise ValueError(
                f"Expected a 3D volume for {subject_id}, got {source_image.shape}"
            )
        if not np.allclose(
            source_image.affine, target_image.affine, atol=affine_tolerance
        ):
            raise ValueError(f"Affine mismatch for paired subject {subject_id}")
        for auxiliary in auxiliary_sources:
            auxiliary_image = nib.load(str(auxiliary))
            if auxiliary_image.shape != source_image.shape:
                raise ValueError(
                    f"Auxiliary shape mismatch for {subject_id}: "
                    f"{auxiliary_image.shape} vs {source_image.shape}"
                )
            if not np.allclose(
                auxiliary_image.affine,
                source_image.affine,
                atol=affine_tolerance,
            ):
                raise ValueError(f"Auxiliary affine mismatch for subject {subject_id}")
        pairs.append(
            VolumePair(
                subject_id,
                source,
                target,
                tuple(source_image.shape),
                auxiliary_sources,
                str(source_field),
            )
        )
    if len({pair.subject_id for pair in pairs}) != len(pairs):
        raise ValueError("subject_ids contains duplicates")
    return pairs


def load_volume(path: str | Path) -> np.ndarray:
    loader = LoadImage(image_only=True, ensure_channel_first=False, dtype=np.float32)
    volume = np.asarray(loader(str(path)), dtype=np.float32)
    if not np.isfinite(volume).all():
        raise ValueError(f"Preprocessed volume contains non-finite values: {path}")
    low = float(volume.min())
    if low < -1e-6:
        raise ValueError(
            f"Preprocessed volume must be non-negative; got minimum {low:.6g}: {path}"
        )
    # Do not silently clip values above one. Residual N4 correction is
    # multiplicative and can legitimately move a very small high-intensity
    # tail above the release-time min-max range. Source normalization and the
    # model head define their own bounded contracts; target normalization may
    # intentionally supervise the un-clipped scaled values.
    return volume


def reorient_spatial_array(
    volume: np.ndarray,
    source_axcodes: Sequence[str] | str,
    target_axcodes: Sequence[str] | str,
) -> np.ndarray:
    """Reorient a 3D or channel-first 4D array without interpolation."""
    source = (
        tuple(str(source_axcodes).upper())
        if isinstance(source_axcodes, str)
        else tuple(source_axcodes)
    )
    target = (
        tuple(str(target_axcodes).upper())
        if isinstance(target_axcodes, str)
        else tuple(target_axcodes)
    )
    if len(source) != 3 or len(target) != 3:
        raise ValueError("source_axcodes and target_axcodes must each contain 3 axes")
    transform = nib.orientations.ornt_transform(
        nib.orientations.axcodes2ornt(source),
        nib.orientations.axcodes2ornt(target),
    )
    if volume.ndim == 3:
        oriented = nib.orientations.apply_orientation(volume, transform)
    elif volume.ndim == 4:
        oriented = np.stack(
            [
                nib.orientations.apply_orientation(channel, transform)
                for channel in volume
            ],
            axis=0,
        )
    else:
        raise ValueError(f"Expected a 3D or channel-first 4D array, got {volume.shape}")
    return np.ascontiguousarray(oriented, dtype=volume.dtype)


def reorient_volume_from_affine(
    volume: np.ndarray,
    affine: np.ndarray,
    target_axcodes: Sequence[str] | str | None,
) -> tuple[np.ndarray, tuple[str, str, str]]:
    """Reorient an array from its NIfTI affine and return original axis codes."""
    source = tuple(str(code) for code in nib.aff2axcodes(affine))
    if target_axcodes is None:
        return np.ascontiguousarray(volume), source
    target = (
        tuple(str(target_axcodes).upper())
        if isinstance(target_axcodes, str)
        else tuple(target_axcodes)
    )
    return reorient_spatial_array(volume, source, target), source


def _load_affine(path: str | Path) -> np.ndarray:
    """Load the native NIfTI affine."""
    return np.asarray(nib.load(str(path)).affine, dtype=np.float64)


@lru_cache(maxsize=8)
def _cached_cdf_prior(path: str) -> tuple[CDFPrior, np.ndarray]:
    return load_cdf_prior(path)


def load_pair_source(
    pair: VolumePair,
    source_intensity_config: dict[str, Any] | None = None,
    orientation_axcodes: Sequence[str] | str | None = None,
) -> np.ndarray:
    """Load one primary volume or a channel-first aligned multimodal stack."""
    volumes = [load_volume(path) for path in pair.source_paths]
    if source_intensity_config:
        if len(volumes) != 1:
            raise ValueError(
                "Source calibration accepts one primary volume; incompatible "
                "with source_modalities"
            )
        affine = nib.load(str(pair.source)).affine
        source = apply_source_intensity_input(
            volumes[0], source_intensity_config, affine=affine
        )
    elif len(volumes) == 1:
        source = volumes[0]
    else:
        source = np.stack(volumes, axis=0).astype(np.float32, copy=False)
    if orientation_axcodes is not None:
        affine = np.asarray(nib.load(str(pair.source)).affine)
        source, _ = reorient_volume_from_affine(source, affine, orientation_axcodes)
    return source


def coordinate_patch(
    start: Sequence[int], patch_size: Sequence[int], full_shape: Sequence[int]
) -> np.ndarray:
    """Return normalized coordinates on the prepared spatial grid in [-1, 1]."""
    axes = []
    for origin, size, full in zip(start, patch_size, full_shape):
        denominator = max(int(full) - 1, 1)
        axis = (
            2.0 * (np.arange(origin, origin + size, dtype=np.float32) / denominator)
            - 1.0
        )
        axes.append(axis)
    grids = np.meshgrid(*axes, indexing="ij")
    return np.stack(grids, axis=0).astype(np.float32, copy=False)


class PairedPatchDataset(Dataset):
    """Random synchronized patches cached from a small set of aligned pairs."""

    def __init__(
        self,
        pairs: Sequence[VolumePair],
        patch_size: Sequence[int],
        patches_per_subject: int,
        edge_probability: float = 0.5,
        foreground_probability: float = 0.85,
        foreground_threshold: float = 0.0,
        evaluation_probability: float = 0.0,
        evaluation_slice_range: Sequence[int] | None = None,
        source_intensity_config: dict[str, Any] | None = None,
        target_normalization_config: dict[str, Any] | None = None,
        sampling_source_channel: int = 0,
        spatial_flip_augmentation: str = "none",
        training_pair_mask: str = "none",
        training_pair_mask_source_channel: int = 0,
        seed: int = 2026,
        orientation_axcodes: str | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("PairedPatchDataset requires at least one volume pair")
        self.pairs = list(pairs)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.patches_per_subject = int(patches_per_subject)
        self.edge_probability = float(edge_probability)
        self.foreground_probability = float(foreground_probability)
        self.foreground_threshold = float(foreground_threshold)
        self.evaluation_probability = float(evaluation_probability)
        self.source_intensity_config = source_intensity_config
        self.target_normalization_config = target_normalization_config
        self.sampling_source_channel = int(sampling_source_channel)
        self.spatial_flip_augmentation = str(spatial_flip_augmentation)
        self.spatial_flip_views = spatial_flip_views(self.spatial_flip_augmentation)
        self.training_pair_mask = str(training_pair_mask)
        self.training_pair_mask_source_channel = int(training_pair_mask_source_channel)
        if self.training_pair_mask not in {"none", "intersection"}:
            raise ValueError("training_pair_mask must be 'none' or 'intersection'")
        self.evaluation_slice_range = (
            tuple(int(value) for value in evaluation_slice_range)
            if evaluation_slice_range is not None
            else None
        )
        self.seed = int(seed)
        self.orientation_axcodes = (
            str(orientation_axcodes).upper()
            if orientation_axcodes is not None
            else None
        )
        self.epoch = 0
        if min(self.edge_probability, self.foreground_probability) < 0.0:
            raise ValueError("Sampling probabilities must be non-negative")
        if self.edge_probability + self.foreground_probability > 1.0:
            raise ValueError("edge_probability + foreground_probability must be <= 1")
        if not 0.0 <= self.evaluation_probability <= 1.0:
            raise ValueError("evaluation_probability must be in [0, 1]")
        if self.evaluation_slice_range is not None:
            if len(self.evaluation_slice_range) != 2:
                raise ValueError("evaluation_slice_range must contain [start, end]")
            if not 0 <= self.evaluation_slice_range[0] < self.evaluation_slice_range[1]:
                raise ValueError(
                    "evaluation_slice_range must be increasing and non-negative"
                )
        if self.patches_per_subject < 1:
            raise ValueError("patches_per_subject must be positive")

        auxiliary_count = len(self.pairs[0].auxiliary_sources)
        if any(len(pair.auxiliary_sources) != auxiliary_count for pair in self.pairs):
            raise ValueError("All pairs must have the same number of source modalities")
        source_keys = ("source",) + tuple(
            f"source_aux{index}" for index in range(auxiliary_count)
        )
        cache_data = []
        for pair in self.pairs:
            item = {"source": str(pair.source), "target": str(pair.target)}
            item.update(
                {
                    f"source_aux{index}": str(path)
                    for index, path in enumerate(pair.auxiliary_sources)
                }
            )
            cache_data.append(item)
        cache_keys = (*source_keys, "target")
        cache_transforms = [
            LoadImaged(
                keys=cache_keys,
                image_only=True,
                ensure_channel_first=True,
                dtype=np.float32,
            ),
        ]
        if self.orientation_axcodes is not None:
            cache_transforms.append(
                Orientationd(
                    keys=cache_keys,
                    axcodes=self.orientation_axcodes,
                    labels=(("L", "R"), ("P", "A"), ("I", "S")),
                )
            )
        cache_transforms.append(EnsureTyped(keys=cache_keys, dtype=torch.float32))
        cache_transform = Compose(cache_transforms)
        self._cache = CacheDataset(
            data=cache_data,
            transform=cache_transform,
            cache_rate=1.0,
            num_workers=0,
            progress=True,
        )
        self._volumes: list[tuple[np.ndarray, np.ndarray]] = []
        self._foreground: list[np.ndarray] = []
        self._edges: list[np.ndarray] = []
        self._evaluation_foreground: list[np.ndarray] = []
        self._evaluation_edges: list[np.ndarray] = []
        self._target_normalization_stats: list[dict[str, float | str]] = []
        for pair_index, pair in enumerate(self.pairs):
            cached = self._cache[pair_index]
            source = np.stack(
                [np.asarray(cached[key][0], dtype=np.float32) for key in source_keys],
                axis=0,
            )
            target = np.asarray(cached["target"][0], dtype=np.float32)
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError(
                    "Preprocessed source/target must contain only finite values "
                    f"for subject {pair.subject_id}"
                )
            raw_low = min(float(source.min()), float(target.min()))
            if raw_low < -1e-6:
                raise ValueError(
                    "Preprocessed source/target must be non-negative; got minimum "
                    f"{raw_low:.6g} for subject {pair.subject_id}"
                )
            raw_target_support = target > self.foreground_threshold
            if self.source_intensity_config:
                if auxiliary_count:
                    raise ValueError(
                        "Source calibration accepts one primary volume; incompatible "
                        "source_modalities"
                    )
                source = apply_source_intensity_input(
                    source[0],
                    self.source_intensity_config,
                    affine=_load_affine(pair.source),
                )
            if self.target_normalization_config:
                target = normalize_target_volume(
                    target,
                    source,
                    self.target_normalization_config,
                    affine=_load_affine(pair.target),
                )
                normalized_values = target[raw_target_support]
                normalization_stats: dict[str, float | str] = {
                    "subject_id": pair.subject_id,
                    "minimum": float(normalized_values.min()),
                    "maximum": float(normalized_values.max()),
                    "fraction_below_zero": float(np.mean(normalized_values < 0.0)),
                    "fraction_above_one": float(np.mean(normalized_values > 1.0)),
                }
                self._target_normalization_stats.append(normalization_stats)
                print(
                    "target_normalization "
                    f"subject={pair.subject_id} "
                    f"range=[{normalization_stats['minimum']:.7f},"
                    f"{normalization_stats['maximum']:.7f}] "
                    f"below0={normalization_stats['fraction_below_zero']:.9g} "
                    f"above1={normalization_stats['fraction_above_one']:.9g}"
                )
            if self.training_pair_mask == "intersection":
                source, target, _ = apply_paired_foreground_intersection(
                    source,
                    target,
                    source_channel=self.training_pair_mask_source_channel,
                    foreground_threshold=self.foreground_threshold,
                    target_support=raw_target_support,
                )
            if not 0 <= self.sampling_source_channel < source.shape[0]:
                raise ValueError(
                    f"sampling_source_channel={self.sampling_source_channel} is "
                    f"outside the {source.shape[0]} input channels"
                )
            primary_source = source[self.sampling_source_channel]
            if any(
                size > full for size, full in zip(self.patch_size, primary_source.shape)
            ):
                raise ValueError(
                    f"patch_size {self.patch_size} exceeds volume "
                    f"{pair.subject_id} shape {primary_source.shape}"
                )
            foreground = np.argwhere(primary_source > self.foreground_threshold)
            if foreground.size == 0:
                raise ValueError(f"No foreground voxels found in {pair.source}")
            if self.edge_probability > 0 or self.evaluation_probability > 0:
                edge_score = np.zeros_like(target, dtype=np.float32)
                for axis in range(3):
                    difference = np.abs(np.diff(target, axis=axis))
                    before = [slice(None)] * 3
                    after = [slice(None)] * 3
                    before[axis] = slice(None, -1)
                    after[axis] = slice(1, None)
                    edge_score[tuple(before)] += difference
                    edge_score[tuple(after)] += difference
                sampled_scores = edge_score[::4, ::4, ::4]
                sampled_foreground = (
                    primary_source[::4, ::4, ::4] > self.foreground_threshold
                )
                if not sampled_foreground.any():
                    sampled_scores = edge_score[
                        primary_source > self.foreground_threshold
                    ]
                else:
                    sampled_scores = sampled_scores[sampled_foreground]
                edge_threshold = np.quantile(sampled_scores, 0.75)
                edges = np.argwhere(
                    (edge_score >= edge_threshold)
                    & (primary_source > self.foreground_threshold)
                )
                del edge_score
            else:
                edges = np.empty((0, 3), dtype=np.int64)
            self._volumes.append((source, target))
            self._foreground.append(foreground)
            self._edges.append(edges)
            if self.evaluation_slice_range is None:
                self._evaluation_foreground.append(foreground)
                self._evaluation_edges.append(edges)
            else:
                z0, z1 = self.evaluation_slice_range
                if z1 > primary_source.shape[2]:
                    raise ValueError(
                        f"evaluation_slice_range {self.evaluation_slice_range} exceeds "
                        f"volume {pair.subject_id} shape {primary_source.shape}"
                    )
                eval_foreground = foreground[
                    (foreground[:, 2] >= z0) & (foreground[:, 2] < z1)
                ]
                eval_edges = edges[(edges[:, 2] >= z0) & (edges[:, 2] < z1)]
                if not len(eval_foreground) or not len(eval_edges):
                    raise ValueError(
                        f"No evaluation-slab samples for {pair.subject_id}"
                    )
                self._evaluation_foreground.append(eval_foreground)
                self._evaluation_edges.append(eval_edges)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs) * self.patches_per_subject

    def _start_from_center(
        self, center: np.ndarray, shape: Sequence[int]
    ) -> tuple[int, ...]:
        starts = []
        for coordinate, size, full in zip(center, self.patch_size, shape):
            starts.append(int(np.clip(coordinate - size // 2, 0, full - size)))
        return tuple(starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair_index = index // self.patches_per_subject
        source, target = self._volumes[pair_index]
        rng = np.random.default_rng(self.seed + self.epoch * len(self) + index)
        evaluation_draw = rng.random()
        if evaluation_draw < self.evaluation_probability:
            total = self.edge_probability + self.foreground_probability
            edge_share = self.edge_probability / total if total > 0 else 0.5
            candidates = (
                self._evaluation_edges[pair_index]
                if rng.random() < edge_share
                else self._evaluation_foreground[pair_index]
            )
            center = candidates[rng.integers(0, len(candidates))]
            start = self._start_from_center(center, target.shape)
        elif (draw := rng.random()) < self.edge_probability:
            edges = self._edges[pair_index]
            center = edges[rng.integers(0, len(edges))]
            start = self._start_from_center(center, target.shape)
        elif draw < self.edge_probability + self.foreground_probability:
            foreground = self._foreground[pair_index]
            center = foreground[rng.integers(0, len(foreground))]
            start = self._start_from_center(center, target.shape)
        else:
            start = tuple(
                int(rng.integers(0, full - size + 1))
                for size, full in zip(self.patch_size, target.shape)
            )
        slices = tuple(
            slice(origin, origin + size) for origin, size in zip(start, self.patch_size)
        )
        source_patch = np.ascontiguousarray(source[(slice(None), *slices)])
        target_patch = np.ascontiguousarray(target[slices][None])
        coordinates = coordinate_patch(start, self.patch_size, target.shape)
        score_mask = np.zeros((1, *self.patch_size), dtype=np.float32)
        if self.evaluation_slice_range is not None:
            z0, z1 = self.evaluation_slice_range
            global_z = np.arange(start[2], start[2] + self.patch_size[2])
            score_mask[:, :, :, (global_z >= z0) & (global_z < z1)] = 1.0
        flip_axes = self.spatial_flip_views[
            int(rng.integers(0, len(self.spatial_flip_views)))
        ]
        if flip_axes:
            # Coordinates are spatial fields, not scalar coordinates to negate.
            # Moving them with the voxel preserves the absolute MNI identity of
            # every source/target location in the augmented view.
            source_patch = flip_channel_first_spatial(source_patch, flip_axes)
            target_patch = flip_channel_first_spatial(target_patch, flip_axes)
            coordinates = flip_channel_first_spatial(coordinates, flip_axes)
            score_mask = flip_channel_first_spatial(score_mask, flip_axes)
        sample = {
            "source": torch.from_numpy(source_patch),
            "target": torch.from_numpy(target_patch),
            "coordinates": torch.from_numpy(coordinates),
            "score_mask": torch.from_numpy(score_mask),
            "subject_id": self.pairs[pair_index].subject_id,
        }
        return sample


class SubjectBalancedBatchSampler(BatchSampler):
    """Yield equal numbers of patches from every training pair per batch."""

    def __init__(
        self,
        subject_count: int,
        patches_per_subject: int,
        batch_size: int,
        seed: int = 2026,
    ) -> None:
        self.subject_count = int(subject_count)
        self.patches_per_subject = int(patches_per_subject)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        if min(self.subject_count, self.patches_per_subject, self.batch_size) < 1:
            raise ValueError("Sampler dimensions must be positive")
        if self.batch_size % self.subject_count:
            raise ValueError(
                "subject-balanced batch_size must be divisible by training pair count"
            )
        self.per_subject_batch = self.batch_size // self.subject_count
        if self.patches_per_subject % self.per_subject_batch:
            raise ValueError(
                "patches_per_subject must be divisible by per-subject batch size"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.patches_per_subject // self.per_subject_batch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        permutations = [
            rng.permutation(self.patches_per_subject) + index * self.patches_per_subject
            for index in range(self.subject_count)
        ]
        for batch_index in range(len(self)):
            start = batch_index * self.per_subject_batch
            stop = start + self.per_subject_batch
            batch = np.concatenate([values[start:stop] for values in permutations])
            rng.shuffle(batch)
            yield batch.tolist()


def resolve_source_intensity_config(data_config):
    return data_config.get("source_intensity")


def apply_source_intensity_input(volume, config, *, affine=None):
    if not config:
        return volume
    method = str(config.get("method"))
    if method == "reference_quantile_scale":
        reference_path = Path(str(config["reference_path"])).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing source reference prior: {reference_path}")
        reference, reference_affine = _cached_cdf_prior(str(reference_path))
        if tuple(volume.shape) != reference.full_shape:
            raise ValueError(
                f"Volume shape {volume.shape} != reference {reference.full_shape}"
            )
        if affine is not None and not np.allclose(
            affine, reference_affine, rtol=0, atol=1e-4
        ):
            raise ValueError(
                "Input grid affine does not match the source reference prior"
            )
        scaled, _ = scale_volume_to_histogram_reference(
            volume,
            reference.target_histogram,
            quantiles=config.get(
                "quantiles",
                [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            ),
            histogram_stride=int(config.get("histogram_stride", 4)),
            foreground_threshold=float(config.get("foreground_threshold", 0.0)),
            scale_range=config.get("scale_range", [0.25, 4.0]),
            clip_output=bool(config.get("clip_output", False)),
        )
        return scaled[None]
    raise ValueError("Source calibration must use reference_quantile_scale")


def normalize_target_volume(target, source, config, *, affine=None):
    """Calibrate the supervised target against the frozen reference."""
    if not config:
        return target.astype(np.float32, copy=False)
    method = str(config.get("method"))
    if method == "reference_quantile_scale":
        support_mask = str(config.get("support_mask", "source"))
        if support_mask == "target":
            support_source = target
        elif support_mask == "source":
            source_channel = int(config.get("source_channel", 0))
            if source.ndim == 3:
                if source_channel != 0:
                    raise ValueError(
                        "target_normalization.source_channel must be 0 for a 3D source"
                    )
                support_source = source
            elif source.ndim == 4:
                if not 0 <= source_channel < source.shape[0]:
                    raise ValueError(
                        f"target_normalization.source_channel={source_channel} is "
                        f"outside the {source.shape[0]} input channels"
                    )
                support_source = source[source_channel]
            else:
                raise ValueError(f"Expected 3D/4D source, got shape={source.shape}")
        else:
            raise ValueError(
                "target_normalization.support_mask must be 'source' or 'target'"
            )
        probabilities = config.get(
            "quantiles", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        common_kwargs = {
            "quantiles": probabilities,
            "foreground_threshold": float(config.get("foreground_threshold", 0.0)),
            "scale_range": config.get("scale_range", [0.5, 2.0]),
            "clip_output": bool(config.get("clip_output", False)),
            "statistics_slice_range": config.get("slice_range"),
            "statistics_axis": int(config.get("slice_axis", 2)),
        }
        explicit_landmarks = config.get("reference_quantiles")
        reference_path_value = config.get("reference_path")
        if explicit_landmarks is not None:
            if reference_path_value is not None:
                raise ValueError(
                    "Set only one of target_normalization.reference_quantiles "
                    "and reference_path"
                )
            return scale_target_quantiles_to_landmarks(
                target,
                support_source,
                explicit_landmarks,
                **common_kwargs,
            )
        if reference_path_value is None:
            raise ValueError(
                "reference_quantile_scale requires reference_quantiles or "
                "reference_path"
            )
        reference_path = Path(str(reference_path_value)).expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Missing target reference prior: {reference_path}")
        reference, reference_affine = _cached_cdf_prior(str(reference_path))
        if affine is not None and not np.allclose(
            affine, reference_affine, rtol=0, atol=1e-4
        ):
            raise ValueError("Target grid affine does not match the reference prior")
        return scale_target_quantiles_to_reference(
            target,
            support_source,
            reference.target_histogram,
            **common_kwargs,
        )
    raise ValueError("Target calibration must use reference_quantile_scale")
