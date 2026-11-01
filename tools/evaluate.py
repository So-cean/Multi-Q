#!/usr/bin/env python3
"""Evaluate either dataset against untouched prepared target intensities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiq.cdf_prior import (  # noqa: E402
    load_cdf_prior,
    scale_volume_to_histogram_reference,
)
from multiq.evaluation import mask_for_evaluation
from multiq.metrics import (  # noqa: E402
    OfficialAxialPerceptualMetric,
    evaluate_volume,
)


QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def prediction(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name.strip() or not directory.strip():
        raise argparse.ArgumentTypeError("prediction must be NAME=DIRECTORY")
    return name.strip(), Path(directory).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        type=prediction,
        metavar="NAME=DIRECTORY",
    )
    parser.add_argument("--reference-prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lpips-device", choices=("cpu", "cuda"))
    parser.add_argument("--lpips-batch-size", type=int, default=8)
    return parser.parse_args()


def load(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Expected one finite 3D volume: {path}")
    return values, image


def token(path: Path) -> str:
    match = re.search(r"_(\d+)\.nii\.gz$", path.name)
    if not match:
        raise ValueError(f"Cannot parse subject token from {path.name}")
    return match.group(1)


def matching_target(target_dir: Path, subject: str) -> Path:
    matches = sorted(target_dir.glob(f"*_{subject}.nii.gz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one target for subject {subject}, found {len(matches)}"
        )
    return matches[0]


def mean_std(rows: Iterable[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def paired_comparison(
    rows: Iterable[dict[str, object]],
    method: str,
    comparator: str,
    metrics: Iterable[str],
) -> dict[str, object]:
    """Summarize paired subject-wise changes without treating scans as independent."""
    indexed: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        name = str(row["method"])
        if name not in {method, comparator}:
            continue
        indexed.setdefault(str(row["subject"]), {})[name] = row
    incomplete = [
        subject
        for subject, methods in indexed.items()
        if set(methods) != {method, comparator}
    ]
    if incomplete:
        raise ValueError(
            f"Incomplete {method} versus {comparator} pairs: {sorted(incomplete)}"
        )
    subjects = sorted(indexed)
    if not subjects:
        raise ValueError(f"No paired rows for {method} versus {comparator}")
    lower_is_better = {"nrmse", "foreground_mae", "lpips"}
    higher_is_better = {"ssim", "psnr"}
    delta: dict[str, float] = {}
    wins: dict[str, int] = {}
    ties: dict[str, int] = {}
    for metric in metrics:
        differences = np.asarray(
            [
                float(indexed[subject][method][metric])
                - float(indexed[subject][comparator][metric])
                for subject in subjects
            ],
            dtype=np.float64,
        )
        delta[metric] = float(differences.mean())
        ties[metric] = int(np.count_nonzero(np.isclose(differences, 0.0)))
        if metric in lower_is_better:
            wins[metric] = int(np.count_nonzero(differences < 0))
        elif metric in higher_is_better:
            wins[metric] = int(np.count_nonzero(differences > 0))
    return {
        "method": method,
        "comparator": comparator,
        "subject_count": len(subjects),
        "subjects": subjects,
        "mean_subjectwise_delta": delta,
        "wins": wins,
        "ties": ties,
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "preprocessing_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_site = str(manifest["source_site"])
    target_site = str(manifest["target_site"])
    modality = str(manifest.get("modality", "T1W"))
    evaluation_role = manifest.get("evaluation_role", "test")
    if evaluation_role not in {"test", "validation"}:
        raise ValueError(f"Unknown evaluation cohort role: {evaluation_role}")
    prefix = "Validation" if evaluation_role == "validation" else "Test"
    source_dir = data_root / f"{prefix}Inputs" / modality / source_site
    target_dir = data_root / f"{prefix}Targets" / modality / target_site
    source_paths = sorted(source_dir.glob("*.nii.gz"))
    if len(source_paths) != int(
        manifest.get("evaluation_subject_count", manifest["counts"]["test_subjects"])
    ):
        raise ValueError("Test input count does not match preprocessing manifest")

    methods: dict[str, Path | None] = {"source": None}
    for name, directory in args.prediction:
        if name in methods:
            raise ValueError(f"Duplicate method name {name!r}")
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        methods[name] = directory

    prior = None
    if args.reference_prior is not None:
        prior, prior_affine = load_cdf_prior(str(args.reference_prior.resolve()))
        methods["multiq_input"] = None
    else:
        prior_affine = None
    lpips = (
        OfficialAxialPerceptualMetric(args.lpips_device, args.lpips_batch_size)
        if args.lpips_device is not None
        else None
    )

    rows: list[dict[str, object]] = []
    for source_path in source_paths:
        subject = token(source_path)
        target_path = matching_target(target_dir, subject)
        source, source_image = load(source_path)
        target, target_image = load(target_path)
        if source.shape != target.shape or not np.allclose(
            source_image.affine, target_image.affine, rtol=0, atol=1e-4
        ):
            raise ValueError(f"Source/target grid mismatch for subject {subject}")
        fixed_mask = (source > 0) & (target > 0)
        if not fixed_mask.any():
            raise ValueError(f"Empty shared foreground for subject {subject}")

        for name, directory in methods.items():
            if name == "source":
                values = source
            elif name == "multiq_input":
                assert prior is not None and prior_affine is not None
                if prior.full_shape != source.shape or not np.allclose(
                    prior_affine, source_image.affine, rtol=0, atol=1e-4
                ):
                    raise ValueError("Multi-Q prior grid differs from SRPBS test grid")
                values, gain = scale_volume_to_histogram_reference(
                    source,
                    prior.target_histogram,
                    quantiles=QUANTILES,
                    histogram_stride=4,
                    foreground_threshold=0.0,
                    scale_range=(0.25, 4.0),
                    clip_output=False,
                )
            else:
                assert directory is not None
                prediction_path = directory / source_path.name
                values, prediction_image = load(prediction_path)
                if values.shape != source.shape or not np.allclose(
                    prediction_image.affine, source_image.affine, rtol=0, atol=1e-4
                ):
                    raise ValueError(f"Prediction grid mismatch: {prediction_path}")
            masked_prediction, masked_target = mask_for_evaluation(
                values, target, source, "intersection"
            )
            metrics = evaluate_volume(masked_prediction, masked_target)
            row: dict[str, object] = {
                "subject": subject,
                "method": name,
                "foreground_voxels": int(fixed_mask.sum()),
                **metrics,
            }
            if name == "multiq_input":
                row["gain"] = float(gain)
            if lpips is not None:
                row["lpips"] = lpips(masked_prediction, masked_target)
            rows.append(row)

    metric_names = [
        "nrmse",
        "ssim",
        "psnr",
        "foreground_mae",
        "sharpness_ratio",
    ]
    if lpips is not None:
        metric_names.append("lpips")
    summary: dict[str, object] = {
        "status": "complete",
        "source_site": source_site,
        "target_site": target_site,
        "evaluation_role": evaluation_role,
        "evaluation_subject_count": len(source_paths),
        "reported_cohort_used_for_selection": evaluation_role == "validation",
        "test_subject_count": len(source_paths) if evaluation_role == "test" else 0,
        "evaluation_mask": "fixed source-target foreground intersection",
        "methods": {},
        "comparisons": {},
    }
    source_rows = [row for row in rows if row["method"] == "source"]
    for name in methods:
        method_rows = [row for row in rows if row["method"] == name]
        method_summary = {
            metric: mean_std(method_rows, metric) for metric in metric_names
        }
        if name != "source":
            method_summary["mean_subjectwise_delta_vs_source"] = {
                metric: float(
                    np.mean(
                        [
                            float(method_row[metric]) - float(source_row[metric])
                            for method_row, source_row in zip(method_rows, source_rows)
                        ]
                    )
                )
                for metric in metric_names
            }
        summary["methods"][name] = method_summary
    if "no_calibration" in methods and "multiq" in methods:
        summary["comparisons"]["multiq_vs_no_calibration"] = paired_comparison(
            rows, "multiq", "no_calibration", metric_names
        )
    if "multiq_input" in methods:
        summary["comparisons"]["multiq_input_vs_source"] = paired_comparison(
            rows, "multiq_input", "source", metric_names
        )

    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "per_subject_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
