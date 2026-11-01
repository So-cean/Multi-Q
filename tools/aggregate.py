#!/usr/bin/env python3
"""Aggregate SRPBS split-level metrics without treating repeated subjects as independent."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def fold_argument(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("fold must be NAME=DIRECTORY")
    return name, Path(directory).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", action="append", required=True, type=fold_argument)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="multiq")
    parser.add_argument("--comparator", default="no_calibration")
    return parser.parse_args()


def numeric_metrics(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"fold", "subject", "method", "foreground_voxels", "gain"}
    candidates = sorted(set().union(*(row.keys() for row in rows)).difference(excluded))
    metrics = []
    for key in candidates:
        try:
            if all(row.get(key, "") != "" and np.isfinite(float(row[key])) for row in rows):
                metrics.append(key)
        except ValueError:
            continue
    return metrics


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate(
    folds: list[tuple[str, Path]], method: str, comparator: str
) -> tuple[list[dict[str, str]], dict[str, object]]:
    rows: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for fold, directory in folds:
        if fold in seen_names:
            raise ValueError(f"Duplicate fold name {fold!r}")
        seen_names.add(fold)
        path = directory / "per_subject_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append({"fold": fold, **row})
    metrics = numeric_metrics(rows)
    methods = sorted({row["method"] for row in rows})
    fold_means: dict[str, dict[str, dict[str, float]]] = {}
    for fold in sorted(seen_names):
        fold_means[fold] = {}
        for name in methods:
            selected = [row for row in rows if row["fold"] == fold and row["method"] == name]
            if selected:
                fold_means[fold][name] = {
                    metric: float(np.mean([float(row[metric]) for row in selected]))
                    for metric in metrics
                }

    indexed = {(row["fold"], row["subject"], row["method"]): row for row in rows}
    pair_keys = sorted(
        (fold, subject)
        for fold, subject, name in indexed
        if name == method and (fold, subject, comparator) in indexed
    )
    if not pair_keys:
        raise ValueError(f"No paired {method} versus {comparator} rows")
    fold_deltas: dict[str, dict[str, float]] = {}
    for fold in sorted(seen_names):
        keys = [key for key in pair_keys if key[0] == fold]
        fold_deltas[fold] = {
            metric: float(
                np.mean(
                    [
                        float(indexed[(key[0], key[1], method)][metric])
                        - float(indexed[(key[0], key[1], comparator)][metric])
                        for key in keys
                    ]
                )
            )
            for metric in metrics
        }

    lower_is_better = {"nrmse", "foreground_mae", "lpips"}
    higher_is_better = {"ssim", "psnr", "sharpness_ratio"}
    wins: dict[str, int] = {}
    for metric in metrics:
        deltas = [
            float(indexed[(fold, subject, method)][metric])
            - float(indexed[(fold, subject, comparator)][metric])
            for fold, subject in pair_keys
        ]
        if metric in lower_is_better:
            wins[metric] = int(np.count_nonzero(np.asarray(deltas) < 0))
        elif metric in higher_is_better:
            wins[metric] = int(np.count_nonzero(np.asarray(deltas) > 0))

    summary: dict[str, object] = {
        "status": "complete",
        "aggregation_unit": "equal-weight split means",
        "dependence_note": (
            "Subjects recur across data assignments; fold-subject observations are not "
            "treated as independent participants."
        ),
        "folds": sorted(seen_names),
        "fold_means": fold_means,
        "methods": {
            name: {
                metric: mean_std(
                    [fold_means[fold][name][metric] for fold in sorted(seen_names)]
                )
                for metric in metrics
            }
            for name in methods
            if all(name in fold_means[fold] for fold in seen_names)
        },
        "comparison": {
            "method": method,
            "comparator": comparator,
            "paired_fold_subject_count": len(pair_keys),
            "unique_subject_count": len({subject for _, subject in pair_keys}),
            "fold_mean_deltas": fold_deltas,
            "equal_weight_fold_delta": {
                metric: mean_std([fold_deltas[fold][metric] for fold in sorted(seen_names)])
                for metric in metrics
            },
            "fold_subject_wins": wins,
        },
    }
    return rows, summary


def main() -> int:
    args = parse_args()
    rows, summary = aggregate(args.fold, args.method, args.comparator)
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["fold"] + sorted(set().union(*(row.keys() for row in rows)).difference({"fold"}))
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
