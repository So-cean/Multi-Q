#!/usr/bin/env python3
"""Synthesize MRI volumes from a self-contained Multi-Q checkpoint."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
import torch
from multiq.data import (
    load_volume,
    apply_source_intensity_input,
    resolve_source_intensity_config,
    reorient_volume_from_affine,
    reorient_spatial_array,
    source_foreground_mask,
)
from multiq.factory import build_mapping_model
from multiq.inference import sliding_window_predict, flip_tta_views


def _install_numpy_pickle_compat() -> None:
    """Allow NumPy 1.x to load checkpoints serialized by NumPy 2.x."""
    try:
        import numpy._core  # noqa: F401
    except ModuleNotFoundError:
        import numpy.core.multiarray

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)


def _checkpoint_model_state(checkpoint: dict, state: str) -> dict:
    """Select an inference state without changing the historical EMA default."""
    if state == "ema":
        return checkpoint.get("ema_model", checkpoint["model"])
    if state == "model":
        if "model" not in checkpoint:
            raise ValueError("Checkpoint lacks the online 'model' state")
        return checkpoint["model"]
    raise ValueError(f"Unsupported checkpoint state: {state!r}")


def load_inference_source(input_path, data_config):
    image = nib.load(input_path)
    source = apply_source_intensity_input(
        load_volume(input_path),
        resolve_source_intensity_config(data_config),
        affine=image.affine,
    )
    source, _ = reorient_volume_from_affine(
        source, image.affine, data_config.get("orientation_axcodes")
    )
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-state", choices=("ema", "model"), default="ema")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overlap", type=float)
    parser.add_argument("--support-mask", choices=("source",), default="source")
    parser.add_argument(
        "--tta",
        choices=("none", "flip-x", "flip-y", "flip-z", "flip-xy", "flip4", "flip8"),
        default="none",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    _install_numpy_pickle_compat()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "paired3dunet":
        raise ValueError("Expected a Multi-Q 3D synthesis checkpoint")
    config = checkpoint["config"]
    model = build_mapping_model(checkpoint["model_config"])
    model.load_state_dict(
        _checkpoint_model_state(checkpoint, args.checkpoint_state), strict=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    data, validation = config["data"], config["validation"]
    batch = (
        args.batch_size
        if args.batch_size is not None
        else config.get("inference", {}).get(
            "sliding_window_batch_size", validation["sliding_window_batch_size"]
        )
    )
    if batch < 1:
        raise ValueError("batch size must be positive")
    files = sorted(args.input_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        output_path = args.output_dir / path.name
        if output_path.exists() and not args.overwrite:
            continue
        image = nib.load(path)
        source = load_inference_source(path, data)
        prediction = sliding_window_predict(
            model=model,
            volume=source,
            patch_size=validation.get("patch_size", data["patch_size"]),
            overlap=float(
                validation["overlap"] if args.overlap is None else args.overlap
            ),
            device=device,
            batch_size=int(batch),
            use_coordinates=bool(config["model"].get("use_coordinates", True)),
            support_mask=source_foreground_mask(
                source, int(data.get("support_mask_source_channel", 0))
            ),
            view_flips=flip_tta_views(args.tta),
            view_fusion="logit_residual",
        )
        orientation = data.get("orientation_axcodes")
        if orientation is not None:
            prediction = reorient_spatial_array(
                prediction, orientation, tuple(nib.aff2axcodes(image.affine))
            )
        result = nib.Nifti1Image(
            prediction.astype(np.float32), image.affine, image.header
        )
        result.set_data_dtype(np.float32)
        nib.save(result, output_path)
        print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
