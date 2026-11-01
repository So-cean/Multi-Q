#!/usr/bin/env python3
"""Plan or execute the same explicit Multi-Q experiment on either dataset.

Planning validates the subject roles and writes portable runtime configs.
Execution is opt-in and belongs on a compute node. Held-out validation subjects
select checkpoints and are explicitly labelled as validation in the final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiq.config import load_config


def subject(value):
    value = str(value)
    return value.zfill(4) if value.isdigit() else value


def resolve_path(value, base):
    value = str(value)
    expanded = os.path.expandvars(value)
    if "${" in expanded:
        raise ValueError(f"Unset path environment variable: {value}")
    path = Path(expanded).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def read_manifest(path, require_ready=True):
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (
        "dataset",
        "source_field",
        "target_field",
        "modality",
        "pairs",
        "references",
        "train",
        "test",
        "preprocessing",
    ):
        if key not in manifest:
            raise ValueError(f"Missing manifest field: {key}")
    pairs = manifest["pairs"]
    if not isinstance(pairs, dict):
        raise ValueError("pairs must be a subject-to-path mapping")
    manifest["pairs"] = {subject(key): value for key, value in pairs.items()}
    if len(manifest["pairs"]) != len(pairs):
        raise ValueError("Duplicate normalized subject IDs")
    for role in ("train", "validation", "test"):
        values = [subject(x) for x in manifest.get(role, [])]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {role} subjects")
        manifest[role] = values
    protocol = manifest.get("protocol", "heldout_validation")
    if protocol not in {"heldout_validation", "holdout", "fixed_epoch"}:
        raise ValueError(
            f"Unknown protocol: {protocol}; nested refitting is not the MRIxFields recipe"
        )
    if protocol == "heldout_validation" and not manifest["validation"]:
        # Historical five-train/four-test assignments use those four subjects
        # for checkpoint selection, so record their actual validation role.
        manifest["validation"], manifest["test"] = manifest["test"], []
    train, val, test = (set(manifest[role]) for role in ("train", "validation", "test"))
    if not train or train & val or train & test or val & test:
        raise ValueError("Training, validation and test subjects must be disjoint")
    if set(manifest["pairs"]) != train | val | test:
        raise ValueError("Every supplied pair needs exactly one declared role")
    if protocol in {"holdout", "heldout_validation"} and not val:
        raise ValueError("Early stopping requires declared validation subjects")
    if protocol == "heldout_validation" and test:
        raise ValueError("Use holdout for a separate independent test cohort")
    if protocol in {"holdout", "fixed_epoch"} and not test:
        raise ValueError("An independent-test protocol requires test subjects")
    if protocol == "fixed_epoch" and val:
        raise ValueError("fixed_epoch does not select a validation checkpoint")
    anchor = manifest.get("spatial_anchor")
    if anchor is not None:
        anchor = subject(anchor)
        if anchor not in train:
            raise ValueError("A subject-derived spatial anchor must be in training")
        manifest["spatial_anchor"] = anchor
    preprocessing = manifest["preprocessing"]
    if require_ready and preprocessing.get("status") != "ready":
        raise ValueError("Preprocessing must be completed before planning training")
    if preprocessing.get("population_used_for_base_normalization") is not False:
        raise ValueError("Population statistics cannot determine base normalization")
    if not isinstance(preprocessing.get("steps"), list) or not preprocessing["steps"]:
        raise ValueError("Record the actual preprocessing steps and their order")
    reference_ids = [subject(row["subject"]) for row in manifest["references"]]
    if not reference_ids or len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Independent reference IDs must be nonempty and unique")
    if set(reference_ids) & (train | val | test):
        raise ValueError("Population reference subjects overlap paired subjects")
    all_paths = []
    for pair in manifest["pairs"].values():
        for role in ("source", "target"):
            pair[role] = str(resolve_path(pair[role], path.parent))
            all_paths.append(pair[role])
    for row in manifest["references"]:
        row["subject"] = subject(row["subject"])
        row["path"] = str(resolve_path(row["path"], path.parent))
        all_paths.append(row["path"])
    if len(all_paths) != len(set(all_paths)):
        raise ValueError(
            "An image path is reused across subject/domain/reference roles"
        )
    missing = [value for value in all_paths if not Path(value).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manifest images: {missing[:5]}")
    manifest["protocol"] = protocol
    return manifest


def check_volumes(manifest):
    """Audit geometry and numeric contracts before allocating training time."""
    import nibabel as nib
    import numpy as np

    grid = None
    paths = [
        Path(p[role])
        for p in manifest["pairs"].values()
        for role in ("source", "target")
    ]
    paths += [Path(row["path"]) for row in manifest["references"]]
    for path in paths:
        image = nib.load(path)
        array = np.asarray(image.dataobj, dtype=np.float32)
        if array.ndim != 3 or not np.isfinite(array).all() or not (array > 0).any():
            raise ValueError(f"Invalid finite 3D foreground: {path}")
        if grid is None:
            grid = image.shape, image.affine
        if image.shape != grid[0] or not np.allclose(
            image.affine, grid[1], rtol=0, atol=1e-4
        ):
            raise ValueError(
                f"Prepared images must share one declared spatial grid: {path}"
            )
        if float(array.min()) < 0 or float(array.max()) > 1 + 1e-5:
            raise ValueError(f"Prepared base intensities must be in [0,1]: {path}")


def expose(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace prepared input: {target}")
    target.symlink_to(source)


def make_layout(manifest, output):
    data = output / "inputs"
    modality, source, target = (
        manifest[k] for k in ("modality", "source_field", "target_field")
    )
    for key, pair in manifest["pairs"].items():
        # Test pairs only appear under TestInputs/TestTargets, not Images.
        for role, field in (("source", source), ("target", target)):
            split = (
                "Images"
                if key not in manifest["test"]
                else ("TestInputs" if role == "source" else "TestTargets")
            )
            expose(
                Path(pair[role]),
                data / split / modality / field / f"P_{modality}_{field}_{key}.nii.gz",
            )
            if key in manifest["validation"]:
                validation_split = (
                    "ValidationInputs" if role == "source" else "ValidationTargets"
                )
                expose(
                    Path(pair[role]),
                    data
                    / validation_split
                    / modality
                    / field
                    / f"P_{modality}_{field}_{key}.nii.gz",
                )
    for row in manifest["references"]:
        expose(
            Path(row["path"]),
            data
            / "Reference"
            / modality
            / target
            / f"R_{modality}_{target}_{row['subject']}.nii.gz",
        )
    metadata = {
        "status": "ready",
        "dataset": manifest["dataset"],
        "source_site": source,
        "target_site": target,
        "modality": modality,
        "train_subjects": manifest["train"],
        "validation_subjects": manifest["validation"],
        "test_subjects": manifest["test"],
        "counts": {"test_subjects": len(manifest["test"])},
        "protocol": manifest["protocol"],
        "evaluation_role": "validation"
        if manifest["protocol"] == "heldout_validation"
        else "test",
        "evaluation_subject_count": len(
            manifest["validation"]
            if manifest["protocol"] == "heldout_validation"
            else manifest["test"]
        ),
        "preprocessing": manifest["preprocessing"],
    }
    (data / "preprocessing_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return data


def runtime_config(manifest, output, method):
    config = load_config(ROOT / "configs/common.yaml")
    config.pop("_config_path", None)
    data = output / "inputs"
    modality, src, tgt = (
        manifest[k] for k in ("modality", "source_field", "target_field")
    )
    config["experiment"].update(name=method, output_dir=str(output / "runs"))
    config["data"].update(
        root=str(data),
        split="Images",
        modality=modality,
        source_field=src,
        target_field=tgt,
        subject_ids=manifest["train"] + manifest.get("validation", []),
    )
    config["validation"].update(raw_target_dir=str(data / "Images" / modality / tgt))
    prior = str(output / "reference.npz")
    if method == "multiq":
        calibration = dict(
            method="reference_quantile_scale",
            reference_path=prior,
            foreground_threshold=0.0,
            clip_output=False,
            scale_range=[0.25, 4.0],
        )
        config["data"]["source_intensity"] = {
            **calibration,
            "histogram_stride": 4,
            "quantiles": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        }
        config["data"]["target_normalization"] = {
            **calibration,
            "source_channel": 0,
            "support_mask": "source",
            "slice_range": None,
            "slice_axis": 2,
            "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    elif method != "no_calibration":
        raise ValueError(f"Unknown method: {method}")
    if manifest["protocol"] in {"holdout", "heldout_validation"}:
        config["data"].update(
            train_subject_ids=manifest["train"],
            validation_subject_ids=manifest["validation"],
        )
    return config


def plan(manifest_path, output, methods, check=False):
    manifest = read_manifest(manifest_path)
    if check:
        check_volumes(manifest)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    record_path = output / "plan.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if (
            previous["manifest_sha256"] != manifest_hash
            or previous["methods"] != methods
        ):
            raise ValueError(
                "Existing output belongs to a different manifest/method plan"
            )
    make_layout(manifest, output)
    configs = {}
    for method in methods:
        config = runtime_config(manifest, output, method)
        path = output / "configs" / f"{method}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[method] = str(path)
    (output / "manifest.resolved.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    record = {
        "manifest_sha256": manifest_hash,
        "dataset": manifest["dataset"],
        "protocol": manifest["protocol"],
        "methods": methods,
        "configs": configs,
        "train": manifest["train"],
        "validation": manifest["validation"],
        "test": manifest["test"],
        "selection": "MRIxFields SSIM and full-volume SSIM guard; prepared targets; EMA",
        "test_used_for_selection": False,
        "reported_cohort_used_for_selection": manifest["protocol"]
        == "heldout_validation",
        "evaluation_role": "validation"
        if manifest["protocol"] == "heldout_validation"
        else "test",
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return manifest, record


def call(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def train_one(config, path, fold, epochs):
    directory = (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
        / ("full" if fold == "full" else f"fold{fold}")
    )
    summary = directory / "run_summary.json"
    fingerprint = hashlib.sha256(
        json.dumps(
            {"config": config, "fold": fold, "epochs": epochs}, sort_keys=True
        ).encode()
    ).hexdigest()
    invocation = directory / "workflow_invocation.json"
    if (
        invocation.exists()
        and json.loads(invocation.read_text())["sha256"] != fingerprint
    ):
        raise ValueError(
            f"Refusing to reuse a run with changed training settings: {directory}"
        )
    if summary.exists() and not invocation.exists():
        raise ValueError(f"Existing run has no workflow provenance: {directory}")
    if summary.is_file():
        state = json.loads(summary.read_text())
        if state.get("status") == "complete":
            return directory
    directory.mkdir(parents=True, exist_ok=True)
    invocation.write_text(json.dumps({"sha256": fingerprint}, indent=2) + "\n")
    call(ROOT / "train.py", "--config", path, "--fold", fold, "--epochs", epochs)
    return directory


def execute(manifest, record, output, stage):
    selected = lambda name: stage in ("all", name)
    if selected("reference"):
        from multiq.cdf_prior import build_cdf_prior, save_cdf_prior

        prior_path = output / "reference.npz"
        if not prior_path.exists():
            paths = [Path(row["path"]) for row in manifest["references"]]
            prior, affine = build_cdf_prior(
                paths, bins=2048, histogram_stride=4, foreground_threshold=0.0
            )
            save_cdf_prior(
                prior_path,
                prior,
                affine,
                {
                    "manifest_sha256": record["manifest_sha256"],
                    "target_count": len(paths),
                    "input_sha256": {
                        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths
                    },
                },
            )
    checkpoints = {}
    for method, path in record["configs"].items():
        config = load_config(path)
        duration = int(config["training"]["epochs"])
        if manifest["protocol"] in {"holdout", "heldout_validation"}:
            if selected("fit"):
                train_one(config, path, "split", duration)
            checkpoint = (
                Path(config["experiment"]["output_dir"]) / method / "foldsplit/best.pth"
            )
        else:
            if selected("fit"):
                train_one(config, path, "full", duration)
            checkpoint = (
                Path(config["experiment"]["output_dir"]) / method / "full/final.pth"
            )
        checkpoints[method] = checkpoint
        if selected("infer"):
            input_split = (
                "ValidationInputs"
                if record["evaluation_role"] == "validation"
                else "TestInputs"
            )
            call(
                ROOT / "infer.py",
                "--checkpoint",
                checkpoint,
                "--input-dir",
                output
                / "inputs"
                / input_split
                / manifest["modality"]
                / manifest["source_field"],
                "--output-dir",
                output / "predictions" / method,
                "--support-mask",
                "source",
                "--tta",
                "none",
            )
    if selected("evaluate"):
        args = [
            ROOT / "tools/evaluate.py",
            "--data-root",
            output / "inputs",
            "--output",
            output / "evaluation",
            "--lpips-device",
            "cuda",
        ]
        for method in record["methods"]:
            args += ["--prediction", f"{method}={output / 'predictions' / method}"]
        call(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("no_calibration", "multiq"),
        default=["no_calibration", "multiq"],
    )
    parser.add_argument("--check-volumes", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute on a compute node; default is planning only",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "reference", "fit", "infer", "evaluate"),
        default="all",
    )
    args = parser.parse_args()
    manifest, record = plan(
        args.manifest.resolve(), args.output, args.methods, args.check_volumes
    )
    print(json.dumps(record, indent=2))
    if args.execute:
        # Execution requires a fresh numeric/grid audit; planning alone reads no
        # volume arrays. Run this branch on an allocated compute node.
        check_volumes(manifest)
        execute(manifest, record, args.output.resolve(), args.stage)


if __name__ == "__main__":
    main()
