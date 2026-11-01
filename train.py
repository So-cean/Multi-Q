#!/usr/bin/env python3
"""Train Multi-Q or its no-calibration control with explicit validation roles."""

from __future__ import annotations
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from multiq.config import load_config, seed_everything
from multiq.data import (
    VolumePair,
    PairedPatchDataset,
    SubjectBalancedBatchSampler,
    discover_pairs,
    load_volume,
    load_pair_source,
    normalize_target_volume,
    reorient_spatial_array,
    reorient_volume_from_affine,
    source_foreground_mask,
    resolve_source_intensity_config,
)
from multiq.evaluation import mask_for_evaluation
from multiq.factory import build_mapping_model
from multiq.inference import sliding_window_predict
from multiq.losses import MappingLoss
from multiq.metrics import (
    OfficialAxialPerceptualMetric,
    evaluate_volume,
    evaluation_view,
)
from multiq.precision import resolve_amp, make_grad_scaler, autocast_context

ROOT = Path(__file__).resolve().parent


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to load checkpoints serialized by NumPy 2.x."""
    try:
        import numpy._core  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.multiarray

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def _split_pairs(
    pairs: list[VolumePair], fold: str, data_config: dict[str, Any] | None = None
) -> tuple[list[VolumePair], list[VolumePair]]:
    if fold == "full":
        return pairs, []
    if fold == "split":
        specification = data_config or {}
        training = {str(x).zfill(4) for x in specification.get("train_subject_ids", [])}
        validation = {
            str(x).zfill(4) for x in specification.get("validation_subject_ids", [])
        }
        discovered = {pair.subject_id for pair in pairs}
        if (
            not training
            or not validation
            or training & validation
            or discovered != training | validation
        ):
            raise ValueError(
                "Explicit split requires complete disjoint training and validation IDs"
            )
        return (
            [p for p in pairs if p.subject_id in training],
            [p for p in pairs if p.subject_id in validation],
        )
    index = int(fold)
    if index < 0:
        raise ValueError("Fold index must be non-negative")
    subjects = list(dict.fromkeys(pair.subject_id for pair in pairs))
    if index >= len(subjects):
        raise ValueError(f"Fold {index} is invalid for {len(subjects)} subjects")
    held_out = subjects[index]
    return (
        [pair for pair in pairs if pair.subject_id != held_out],
        [pair for pair in pairs if pair.subject_id == held_out],
    )


def _run_dir(config: dict[str, Any], fold: str) -> Path:
    output_root = Path(config["experiment"]["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    return (
        output_root
        / config["experiment"]["name"]
        / ("full" if fold == "full" else f"fold{fold}")
    )


def _save_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _capture_rng_state(
    loader_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Capture all stochastic state needed for a faithful training resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if loader_generator is not None:
        state["loader_generator"] = loader_generator.get_state()
    return state


def _restore_rng_state(
    state: dict[str, Any] | None,
    loader_generator: torch.Generator | None = None,
) -> bool:
    """Restore checkpoint RNG state; return false for legacy checkpoints."""
    if not state:
        return False
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"Checkpoint RNG state is missing {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if loader_generator is not None and "loader_generator" in state:
        loader_generator.set_state(state["loader_generator"])
    return True


def _add_constrained_lpips_prescreen(
    metrics: dict[str, float],
    reference: dict[str, float],
    selection_config: dict[str, Any],
) -> None:
    """Minimize official LPIPS subject to reconstruction safety constraints.

    SSIM, nRMSE, and LPIPS have different numeric scales and leaderboard value
    only changes when a competitor boundary is crossed.  Averaging their raw
    or relative changes therefore does not approximate rank-sum.  During LOSO
    training, where hidden leaderboard ranks and SynthSeg scores are absent,
    use this only as a cheap pre-screen: preserve SSIM/nRMSE within configured
    budgets and rank eligible checkpoints by LPIPS gain. Final promotion still
    requires post-training Dice/Volume evaluation with SynthSeg.
    """
    names = tuple(selection_config.get("metrics", ["ssim", "nrmse", "lpips"]))
    if names != ("ssim", "nrmse", "lpips"):
        raise ValueError(
            "constrained_lpips_prescreen metrics must be [ssim, nrmse, lpips]"
        )
    missing = [name for name in names if name not in metrics or name not in reference]
    if missing:
        raise KeyError(f"constrained_lpips_prescreen is missing metrics: {missing}")
    reference_lpips = float(reference["lpips"])
    if reference_lpips <= 1e-12:
        raise ValueError("constrained_lpips_prescreen reference LPIPS must be positive")
    deltas = {name: float(metrics[name]) - float(reference[name]) for name in names}
    perceptual_gain = -deltas["lpips"] / reference_lpips
    constraints = selection_config.get("constraints", {})
    if "full_ssim_min_delta" in constraints:
        missing_full = [
            container_name
            for container_name, container in (
                ("metrics", metrics),
                ("reference", reference),
            )
            if "full_ssim" not in container
        ]
        if missing_full:
            raise KeyError(
                "constrained_lpips_prescreen full_ssim constraint requires full_ssim in "
                + " and ".join(missing_full)
            )
    eligible = (
        float(metrics["ssim"])
        >= float(reference["ssim"])
        + float(constraints.get("ssim_min_delta", -math.inf))
        and float(metrics["nrmse"])
        <= float(reference["nrmse"])
        + float(constraints.get("nrmse_max_delta", math.inf))
        and float(metrics["lpips"])
        <= float(reference["lpips"])
        + float(constraints.get("lpips_max_delta", math.inf))
        and float(metrics.get("full_ssim", 0.0))
        >= float(reference.get("full_ssim", 0.0))
        + float(constraints.get("full_ssim_min_delta", -math.inf))
    )
    score = perceptual_gain
    # All eligible candidates outrank every ineligible one; the continuous
    # component still orders candidates within either group for diagnostics.
    if not eligible:
        score = min(score, 0.999999) - 1.0
    metrics.update(
        {
            "constrained_lpips_prescreen": float(score),
            "selection_eligible": float(eligible),
            "perceptual_gain": float(perceptual_gain),
            **{f"selection_delta_{name}": value for name, value in deltas.items()},
        }
    )


def _guarded_checkpoint_selection(
    primary_metrics: dict[str, Any],
    guard_metrics: dict[str, Any],
    primary_monitor: str,
    guard_monitor: str,
    guard_tolerance: float,
) -> dict[str, Any]:
    """Choose the official-metric optimum unless it violates the volume guard."""
    if guard_tolerance < 0:
        raise ValueError("validation.guard_tolerance must be non-negative")
    for name, metrics, key in (
        ("primary", primary_metrics, primary_monitor),
        ("primary", primary_metrics, guard_monitor),
        ("guard", guard_metrics, guard_monitor),
    ):
        if key not in metrics:
            raise KeyError(f"{name} checkpoint metrics do not contain {key!r}")
    primary_guard = float(primary_metrics[guard_monitor])
    best_guard = float(guard_metrics[guard_monitor])
    primary_is_safe = primary_guard >= best_guard - guard_tolerance
    selected_kind = "primary" if primary_is_safe else "guard"
    selected_metrics = primary_metrics if primary_is_safe else guard_metrics
    return {
        "selection_policy": "primary_metric_with_full_volume_guard",
        "primary_monitor": primary_monitor,
        "guard_monitor": guard_monitor,
        "guard_tolerance": float(guard_tolerance),
        "primary_checkpoint": "best_primary.pth",
        "guard_checkpoint": "best_guard.pth",
        "selected_kind": selected_kind,
        "selected_checkpoint": f"best_{selected_kind}.pth",
        "selected_epoch": int(selected_metrics["epoch"]),
        "primary_best_epoch": int(primary_metrics["epoch"]),
        "primary_best_score": float(primary_metrics[primary_monitor]),
        "primary_guard_score": primary_guard,
        "guard_best_epoch": int(guard_metrics["epoch"]),
        "guard_best_score": best_guard,
        "guard_margin": primary_guard - best_guard,
    }


def _update_guarded_checkpoint_alias(
    run_dir: Path,
    primary_metrics: dict[str, Any],
    guard_metrics: dict[str, Any],
    primary_monitor: str,
    guard_monitor: str,
    guard_tolerance: float,
) -> dict[str, Any]:
    """Atomically update legacy best.* files to the guarded selection."""
    selection = _guarded_checkpoint_selection(
        primary_metrics, guard_metrics, primary_monitor, guard_monitor, guard_tolerance
    )
    selected_checkpoint = str(selection["selected_checkpoint"])
    temporary_alias = run_dir / "best.pth.tmp"
    temporary_alias.unlink(missing_ok=True)
    temporary_alias.hardlink_to(run_dir / selected_checkpoint)
    temporary_alias.replace(run_dir / "best.pth")
    selected_metrics = (
        primary_metrics if selection["selected_kind"] == "primary" else guard_metrics
    )
    _save_json_atomic(
        run_dir / "best_metrics.json",
        {**selected_metrics, "guarded_selection": selection},
    )
    _save_json_atomic(run_dir / "checkpoint_selection.json", selection)
    return selection


def _accumulation_window_size(
    batch_index: int, batch_count: int, accumulation_steps: int
) -> int:
    """Return the actual microbatch count in this optimizer-update window."""
    if accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be positive")
    if not 1 <= batch_index <= batch_count:
        raise ValueError("batch_index must lie inside the epoch")
    window_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps + 1
    return min(accumulation_steps, batch_count - window_start + 1)


def _optimizer_update_due(
    batch_index: int, batch_count: int, accumulation_steps: int
) -> bool:
    """Return whether this microbatch closes an accumulation window."""
    _accumulation_window_size(batch_index, batch_count, accumulation_steps)
    return batch_index % accumulation_steps == 0 or batch_index == batch_count


@torch.no_grad()
def _update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, decay: float
) -> None:
    model_state = model.state_dict()
    floating_groups: dict[
        tuple[torch.device, torch.dtype], tuple[list[torch.Tensor], list[torch.Tensor]]
    ] = {}
    for name, ema_value in ema_model.state_dict().items():
        current = model_state[name].detach()
        if torch.is_floating_point(ema_value):
            key = (ema_value.device, ema_value.dtype)
            ema_values, current_values = floating_groups.setdefault(key, ([], []))
            ema_values.append(ema_value)
            current_values.append(current)
        else:
            ema_value.copy_(current)
    for ema_values, current_values in floating_groups.values():
        torch._foreach_lerp_(ema_values, current_values, 1.0 - decay)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default="0", help="LOSO index, split, or full")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--check-data", action="store_true")
    return parser.parse_args()


def _validate(
    model,
    pair,
    config,
    device,
    run_dir,
    epoch,
    support_mask=None,
    criterion=None,
    official_perceptual_metric=None,
):
    """Score the prepared target; aligned-target metrics are diagnostics."""
    data, validation = config["data"], config["validation"]
    orientation = data.get("orientation_axcodes")
    source = load_pair_source(pair, resolve_source_intensity_config(data), orientation)
    support = source_foreground_mask(
        source, int(data.get("support_mask_source_channel", 0))
    )
    if support_mask is not None:
        support = support_mask
    target_path = (
        pair.target
        if validation.get("raw_target_dir") is None
        else Path(validation["raw_target_dir"]) / pair.target.name
    )
    target_image = nib.load(target_path)
    pair_image = nib.load(pair.target)
    if target_image.shape != pair_image.shape or not np.allclose(
        target_image.affine, pair_image.affine, rtol=0, atol=1e-4
    ):
        raise ValueError("Validation target must use the paired target grid")
    raw_target = load_volume(target_path)
    prediction = sliding_window_predict(
        model=model,
        volume=source,
        device=device,
        patch_size=validation.get("patch_size", data["patch_size"]),
        overlap=float(validation["overlap"]),
        batch_size=int(validation["sliding_window_batch_size"]),
        use_coordinates=bool(config["model"].get("use_coordinates", True)),
        support_mask=support,
    )
    original_codes = tuple(nib.aff2axcodes(target_image.affine))
    native_prediction = (
        prediction
        if orientation is None
        else reorient_spatial_array(prediction, orientation, original_codes)
    )
    metric_prediction, metric_target = mask_for_evaluation(
        native_prediction,
        raw_target,
        load_volume(pair.source),
        validation.get("evaluation_mask", "none"),
    )
    slab, axis = validation.get("slice_range"), int(validation.get("slice_axis", 2))
    metrics = evaluate_volume(
        metric_prediction, metric_target, slice_range=slab, slice_axis=axis
    )
    if validation.get("full_volume_metrics", False):
        metrics.update(
            {
                f"full_{name}": value
                for name, value in evaluate_volume(
                    metric_prediction, metric_target
                ).items()
            }
        )
    if official_perceptual_metric is not None:
        metrics["lpips"] = official_perceptual_metric(
            evaluation_view(metric_prediction, slab, axis),
            evaluation_view(metric_target, slab, axis),
            slice_axis=axis,
        )
    if data.get("target_normalization"):
        target, _ = reorient_volume_from_affine(
            raw_target, target_image.affine, orientation
        )
        aligned = normalize_target_volume(
            target, source, data["target_normalization"], affine=target_image.affine
        )
        metrics.update(
            {
                f"aligned_{name}": value
                for name, value in evaluate_volume(
                    prediction, aligned, slice_range=slab, slice_axis=axis
                ).items()
            }
        )
    return metrics


def _validate_pairs(model, pairs, config, device, run_dir, epoch, **kwargs):
    if not pairs:
        raise ValueError("Validation requires held-out subjects")
    values = [
        _validate(model, pair, config, device, run_dir, epoch, **kwargs)
        for pair in pairs
    ]
    return {key: float(np.mean([value[key] for value in values])) for key in values[0]}


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    fold,
    epoch,
    global_step,
    selection,
    ema_model,
    generator,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "method": "paired3dunet",
        "model": model.state_dict(),
        "model_config": copy.deepcopy(config["model"]),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "fold": fold,
        "rng_state": _capture_rng_state(generator),
        **selection,
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    _install_numpy_pickle_compat()
    config = load_config(args.config)
    data, training, validation = (
        config["data"],
        config["training"],
        config["validation"],
    )
    if training.get("loss_balance", {}).get("enabled") or training.get(
        "adversarial", {}
    ).get("enabled"):
        raise ValueError("Multi-Q uses the configured fixed supervised objective")
    monitor = str(validation["monitor"])
    if monitor not in {"ssim", "constrained_lpips_prescreen"}:
        raise ValueError(f"Unsupported MRIxFields checkpoint monitor: {monitor}")
    if monitor == "constrained_lpips_prescreen" and (
        validation.get("selection", {}).get("method") != monitor
        or not validation.get("compute_official_lpips")
        or not validation.get("before_training")
    ):
        raise ValueError(
            "Constrained LPIPS selection requires the original selection config, LPIPS and epoch-zero validation"
        )
    seed_everything(int(config["experiment"]["seed"]))
    pairs = discover_pairs(
        **{
            key: data[key]
            for key in (
                "root",
                "split",
                "modality",
                "source_field",
                "target_field",
                "subject_ids",
            )
        }
    )
    train_pairs, validation_pairs = _split_pairs(pairs, args.fold, data)
    if args.check_data:
        print(
            f"Validated {len(train_pairs)} training and {len(validation_pairs)} validation pairs"
        )
        return
    epochs = int(training["epochs"] if args.epochs is None else args.epochs)
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if epochs < 0 or (epochs == 0 and args.fold != "full") or accumulation < 1:
        raise ValueError(
            "Positive training duration and accumulation are required; full fitting also accepts epoch zero"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dataset = PairedPatchDataset(
        pairs=train_pairs,
        patch_size=data["patch_size"],
        patches_per_subject=int(data["patches_per_subject"]),
        edge_probability=float(data.get("edge_probability", 0)),
        foreground_probability=float(data["foreground_probability"]),
        foreground_threshold=float(data.get("foreground_threshold", 0)),
        source_intensity_config=resolve_source_intensity_config(data),
        target_normalization_config=data.get("target_normalization"),
        sampling_source_channel=int(data.get("sampling_source_channel", 0)),
        training_pair_mask=data.get("training_pair_mask", "none"),
        training_pair_mask_source_channel=int(
            data.get("training_pair_mask_source_channel", 0)
        ),
        orientation_axcodes=data.get("orientation_axcodes"),
        spatial_flip_augmentation=data.get("spatial_flip_augmentation", "none"),
        seed=int(config["experiment"]["seed"]),
    )
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    sampler = None
    loader_options = {
        "batch_size": int(training["batch_size"]),
        "shuffle": True,
        "generator": generator,
    }
    # MRIxFields reads this optional flag from training, not data.
    if training.get("subject_balanced_batches", False):
        sampler = SubjectBalancedBatchSampler(
            subject_count=len(train_pairs),
            patches_per_subject=int(data["patches_per_subject"]),
            batch_size=int(training["batch_size"]),
            seed=int(config["experiment"]["seed"]),
        )
        loader_options = {"batch_sampler": sampler}
    loader = DataLoader(
        dataset,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        **loader_options,
    )
    model = build_mapping_model(config["model"]).to(device)
    criterion = MappingLoss(**config["loss"]).to(device)
    gate = model.residual_gate
    gate_parameters = (
        [gate] if isinstance(gate, torch.nn.Parameter) and gate.requires_grad else []
    )
    gate_ids = {id(value) for value in gate_parameters}
    groups = [
        {
            "params": [
                value
                for value in model.parameters()
                if value.requires_grad and id(value) not in gate_ids
            ]
        }
    ]
    if gate_parameters:
        groups.append(
            {
                "params": gate_parameters,
                "weight_decay": 0.0,
                "lr": float(training["learning_rate"])
                * float(training.get("residual_gate_lr_multiplier", 1.0)),
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        fused=bool(training.get("fused_optimizer", False)) and device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    amp_enabled, amp_dtype, scaler_enabled = resolve_amp(training, device)
    scaler = make_grad_scaler(device, enabled=scaler_enabled)
    ema_decay = float(training.get("ema_decay", 0))
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    ema_model = copy.deepcopy(model).requires_grad_(False).eval() if ema_decay else None
    run_dir = _run_dir(config, args.fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    selection = {
        "best_score": -math.inf,
        "best_epoch": 0,
        "best_guard_score": -math.inf,
        "best_guard_epoch": 0,
        "stale_validations": 0,
    }
    start_epoch, global_step, history = 1, 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            str(checkpoint["fold"]) != args.fold
            or checkpoint["model_config"] != config["model"]
        ):
            raise ValueError("Resume checkpoint must match the fold and model")
        for key in (
            "subject_ids",
            "source_field",
            "target_field",
            "source_intensity",
            "target_normalization",
        ):
            if checkpoint["config"]["data"].get(key) != data.get(key):
                raise ValueError(f"Resume data contract changed: {key}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if ema_model is not None:
            ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
        for key in selection:
            selection[key] = checkpoint.get(key, selection[key])
        start_epoch, global_step = (
            int(checkpoint["epoch"]) + 1,
            int(checkpoint.get("global_step", 0)),
        )
        _restore_rng_state(checkpoint.get("rng_state"), generator)
        if (run_dir / "history.json").is_file():
            history = [
                row
                for row in json.loads((run_dir / "history.json").read_text())
                if int(row["epoch"]) < start_epoch
            ]
    perceptual = (
        OfficialAxialPerceptualMetric(
            str(device), int(validation.get("official_perceptual_batch_size", 4))
        )
        if validation_pairs and validation.get("compute_official_lpips")
        else None
    )
    guard = validation.get("guard_monitor")
    if guard not in (None, "full_ssim") or (
        guard and not validation.get("full_volume_metrics")
    ):
        raise ValueError("The optional full-SSIM guard requires full-volume metrics")

    def save(name, epoch):
        _save_checkpoint(
            run_dir / name,
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            args.fold,
            epoch,
            global_step,
            selection,
            ema_model,
            generator,
        )

    selection_reference = None
    if args.resume and monitor == "constrained_lpips_prescreen":
        selection_reference = checkpoint.get("selection_reference")
        if selection_reference is None:
            for row in history:
                if row["epoch"] == 0:
                    selection_reference = row["validation"]
                    break
        if selection_reference is None:
            raise ValueError(
                "Resume requires the original epoch-zero selection reference"
            )

    def validate(epoch):
        nonlocal selection_reference
        metrics = _validate_pairs(
            ema_model if ema_model is not None else model,
            validation_pairs,
            config,
            device,
            run_dir,
            epoch,
            official_perceptual_metric=perceptual,
        )
        if monitor == "constrained_lpips_prescreen":
            if epoch == 0:
                selection_reference = {
                    key: float(metrics[key])
                    for key in ("ssim", "nrmse", "lpips", "full_ssim")
                }
            if selection_reference is None:
                raise RuntimeError(
                    "Constrained selection requires epoch-zero validation"
                )
            _add_constrained_lpips_prescreen(
                metrics, selection_reference, validation["selection"]
            )
            selection["selection_reference"] = selection_reference
        improved = metrics[monitor] > selection["best_score"] + float(
            validation["min_delta"]
        )
        if improved:
            selection.update(
                best_score=float(metrics[monitor]),
                best_epoch=epoch,
                stale_validations=0,
            )
        else:
            selection["stale_validations"] += 1
        guard_improved = guard is not None and float(metrics[guard]) > selection[
            "best_guard_score"
        ] + float(validation["min_delta"])
        if guard_improved:
            selection.update(
                best_guard_score=float(metrics[guard]), best_guard_epoch=epoch
            )
        record = {"epoch": epoch, **metrics}
        if improved:
            save("best_primary.pth" if guard else "best.pth", epoch)
            _save_json_atomic(
                run_dir
                / ("best_primary_metrics.json" if guard else "best_metrics.json"),
                record,
            )
        if guard_improved:
            save("best_guard.pth", epoch)
            _save_json_atomic(run_dir / "best_guard_metrics.json", record)
        if guard and (improved or guard_improved):
            _update_guarded_checkpoint_alias(
                run_dir,
                json.loads((run_dir / "best_primary_metrics.json").read_text()),
                json.loads((run_dir / "best_guard_metrics.json").read_text()),
                monitor,
                guard,
                float(validation.get("guard_tolerance", 0)),
            )
        elif improved:
            _save_json_atomic(
                run_dir / "checkpoint_selection.json",
                {
                    "selected_epoch": epoch,
                    "selected_checkpoint": "best.pth",
                    "selection_policy": f"validation_{monitor}",
                },
            )
        return metrics

    print(
        f"fold={args.fold} train={[pair.subject_id for pair in train_pairs]} val={[pair.subject_id for pair in validation_pairs]} epochs={epochs} device={device}"
    )
    if (
        not args.resume
        and validation_pairs
        and validation.get("before_training", False)
    ):
        history.append({"epoch": 0, "validation": validate(0)})
        _save_json_atomic(run_dir / "history.json", history)
    terminal_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {}
        for index, batch in enumerate(loader, start=1):
            source, target, coordinates = [
                batch[key].to(device, non_blocking=True)
                for key in ("source", "target", "coordinates")
            ]
            channel = int(data.get("support_mask_source_channel", model.base_channel))
            support = source[:, channel : channel + model.out_channels] > 0
            loss_support = support if criterion.mask_to_output_support else None
            score_mask = (
                batch["score_mask"].to(device)
                if criterion.score_region_weight
                else None
            )
            with autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                prediction = model(source, coordinates, support_mask=support)
                losses = criterion(
                    prediction, target, score_mask=score_mask, valid_mask=loss_support
                )
            scaler.scale(
                losses["total"]
                / _accumulation_window_size(index, len(loader), accumulation)
            ).backward()
            if _optimizer_update_due(index, len(loader), accumulation):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["grad_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if ema_model is not None:
                    _update_ema(ema_model, model, ema_decay)
            for key, value in losses.items():
                sums[key] = sums.get(key, 0) + value.detach()
        scheduler.step()
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: float(value / len(loader)) for key, value in sums.items()},
        }
        should_validate = bool(validation_pairs) and (
            epoch == 1 or epoch % int(validation["every"]) == 0
        )
        if should_validate:
            row["validation"] = validate(epoch)
        history.append(row)
        _save_json_atomic(run_dir / "history.json", history)
        if (
            should_validate
            or epoch == epochs
            or epoch % int(training.get("save_last_every", 1)) == 0
        ):
            save("last.pth", epoch)
        if epoch % int(training.get("save_every", epochs or 1)) == 0:
            save(f"epoch{epoch:04d}.pth", epoch)
        print(f"epoch={epoch:04d} loss={row['train']['total']:.6f}", flush=True)
        terminal_epoch = epoch
        if (
            validation_pairs
            and epoch >= int(validation.get("early_stopping_min_epoch", 1))
            and selection["stale_validations"] >= int(validation["patience"])
        ):
            print(
                f"Early stopping at epoch {epoch}; best epoch {selection['best_epoch']}"
            )
            break
    if args.fold == "full":
        save("final.pth", terminal_epoch)
    else:
        save("last.pth", terminal_epoch)
    _save_json_atomic(
        run_dir / "run_summary.json",
        {
            "status": "complete",
            "fold": args.fold,
            "terminal_epoch": terminal_epoch,
            "global_step": global_step,
            "resumed": args.resume is not None,
            "invocation_start_epoch": start_epoch,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "wall_seconds": time.time() - started,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
    )


if __name__ == "__main__":
    main()
