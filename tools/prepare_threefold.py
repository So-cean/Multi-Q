#!/usr/bin/env python3
"""Create three rotating 3-train/3-validation/3-test SRPBS assignments."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.workflow import plan, read_manifest


def threefold_assignments(manifest):
    """Rotate disjoint groups so each subject trains, validates and tests once."""
    subjects = sorted(manifest["pairs"])
    if len(subjects) != 9:
        raise ValueError("The SRPBS 3/3/3 protocol requires nine paired subjects")
    if (
        manifest.get("spatial_anchor")
        or manifest["preprocessing"].get("template_independent") is not True
    ):
        raise ValueError(
            "Three-fold rotation requires a spatial template independent of the nine subjects"
        )
    groups = [subjects[start : start + 3] for start in (0, 3, 6)]
    folds = []
    for index in range(3):
        fold = copy.deepcopy(manifest)
        fold.update(
            protocol="holdout",
            train=groups[index],
            validation=groups[(index + 1) % 3],
            test=groups[(index + 2) % 3],
            split_id=f"fold_{index + 1}",
        )
        folds.append(fold)
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for fold in threefold_assignments(manifest):
        path = args.output / f"{fold['split_id']}.yaml"
        payload = yaml.safe_dump(fold, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise FileExistsError(f"Existing assignment differs: {path}")
        path.write_text(payload)
        _, record = plan(
            path.resolve(), args.output / fold["split_id"], ["no_calibration", "multiq"]
        )
        records.append(
            {"split": fold["split_id"], "manifest": str(path.resolve()), **record}
        )
    (args.output / "threefold.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
