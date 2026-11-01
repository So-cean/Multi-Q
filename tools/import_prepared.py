#!/usr/bin/env python3
"""Create an explicit manifest from already prepared MRIxFields/SRPBS volumes.

This indexes existing images without changing their intensities or spatial
headers. The preprocessing history is supplied explicitly by the caller.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.workflow import read_manifest, subject


def match(directory, key):
    matches = list(directory.glob(f"*_{key}.nii.gz"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one image for {key} in {directory}; got {len(matches)}"
        )
    return str(matches[0].resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("mrixfields", "srpbs"), required=True)
    parser.add_argument("--source-field", required=True)
    parser.add_argument("--target-field", required=True)
    parser.add_argument("--modality", choices=("T1W", "T2W", "T2FLAIR"), default="T1W")
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--validation", nargs="*", default=[])
    parser.add_argument("--test", nargs="*", default=[])
    parser.add_argument(
        "--protocol",
        choices=("heldout_validation", "holdout", "fixed_epoch"),
        default="heldout_validation",
    )
    parser.add_argument("--spatial-anchor")
    parser.add_argument("--preprocessing-step", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root.resolve()
    test = {subject(x) for x in args.test + args.validation}
    pairs = {}
    for key in [subject(x) for x in args.train + args.validation + args.test]:
        source_dir = (
            root / "TestInputs" / args.modality / args.source_field
            if key in test
            else root / "Images" / args.modality / args.source_field
        )
        target_dir = (
            root / "TestTargets" / args.modality / args.target_field
            if key in test
            else root / "Images" / args.modality / args.target_field
        )
        # A prepared all-pair Images tree can be safely filtered by the planner.
        if not source_dir.is_dir():
            source_dir = root / "Images" / args.modality / args.source_field
        if not target_dir.is_dir():
            target_dir = root / "Images" / args.modality / args.target_field
        pairs[key] = {
            "source": match(source_dir, key),
            "target": match(target_dir, key),
        }
    references = [
        {
            "subject": subject(path.name.removesuffix(".nii.gz").rsplit("_", 1)[-1]),
            "path": str(path.resolve()),
        }
        for path in sorted(args.reference_dir.glob("*.nii.gz"))
    ]
    manifest = {
        "dataset": args.dataset,
        "source_field": args.source_field,
        "target_field": args.target_field,
        "modality": args.modality,
        "protocol": args.protocol,
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
        "pairs": pairs,
        "references": references,
        "preprocessing": {
            "status": "ready",
            "steps": args.preprocessing_step,
            "population_used_for_base_normalization": False,
        },
    }
    if args.spatial_anchor:
        manifest["spatial_anchor"] = args.spatial_anchor
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(manifest, sort_keys=False))
    validated = read_manifest(args.output.resolve())
    args.output.write_text(yaml.safe_dump(validated, sort_keys=False))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
