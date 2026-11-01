#!/usr/bin/env python3
"""Build a location-independent target-cohort CDF intensity prior."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiq.cdf_prior import build_cdf_prior, save_cdf_prior  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument(
        "--target-template",
        type=Path,
        help="Build the target histogram from one precomputed MNI template.",
    )
    parser.add_argument("--split", default="Training_retrospective")
    parser.add_argument(
        "--modality", choices=["T1W", "T2W", "T2FLAIR"], required=True
    )
    parser.add_argument("--target-field", default="7T")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=2048)
    parser.add_argument("--histogram-stride", type=int, default=4)
    parser.add_argument(
        "--histogram-slice-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
    )
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_template is not None:
        target_paths = [args.target_template.expanduser().resolve()]
        if not target_paths[0].is_file():
            raise FileNotFoundError(f"Missing CDF target template: {target_paths[0]}")
        target_dir = target_paths[0].parent
        method = "target_template_cdf"
    else:
        target_dir = args.data_root / args.split / args.modality / args.target_field
        target_paths = sorted(target_dir.glob("*.nii.gz"))
        if not target_paths:
            raise FileNotFoundError(f"Empty CDF target cohort: {target_dir}")
        method = "target_cohort_cdf"
    if args.output.exists():
        raise FileExistsError(args.output)
    file_hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in target_paths}
    prior, affine = build_cdf_prior(
        target_paths,
        bins=args.bins,
        histogram_stride=args.histogram_stride,
        histogram_slice_range=args.histogram_slice_range,
        foreground_threshold=args.foreground_threshold,
    )
    metadata = {
        "format_version": 1,
        "input_sha256": file_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": method,
        "split": args.split,
        "modality": args.modality,
        "target_field": args.target_field,
        "target_count": len(target_paths),
        "target_dir": str(target_dir),
        "target_template": (
            str(target_paths[0]) if args.target_template is not None else None
        ),
        "histogram_slice_range": args.histogram_slice_range,
    }
    save_cdf_prior(args.output, prior, affine, metadata)
    print(f"saved={args.output} target={len(target_paths)}")


if __name__ == "__main__":
    main()
