#!/usr/bin/env python3
"""Prepare raw MRIxFields or SRPBS images with one identical per-image chain.

Raw input manifests use the workflow schema with preprocessing.status=raw.
This opt-in raw pipeline does not reproduce historical challenge preprocessing.
Import already prepared derivatives with their actual preprocessing history.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiq.preprocessing import (
    Scan,
    prepare_native,
    register_scan,
    reuse_native_cache,
    scan_key,
    subject_minmax_to_output,
)
from tools.workflow import read_manifest, subject


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    template = parser.add_mutually_exclusive_group(required=True)
    template.add_argument(
        "--template-subject", help="A training subject whose target defines the grid"
    )
    template.add_argument(
        "--template-image", type=Path, help="An independent already prepared template"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--native-cache-manifest",
        type=Path,
        help="Provenance manifest for existing HD-BET/N4 native images",
    )
    parser.add_argument(
        "--native-cache-root",
        type=Path,
        help="Directory containing native/<scan key>/n4_brain.nii.gz and mask.nii.gz",
    )
    for flag, executable in (
        ("hdbet", "hd-bet"),
        ("n4", "N4BiasFieldCorrection"),
        ("ants-registration", "antsRegistrationSyNQuick.sh"),
        ("ants-apply", "antsApplyTransforms"),
    ):
        parser.add_argument(f"--{flag}", default=executable)
    opts = parser.parse_args()
    manifest_path = opts.manifest.resolve()
    manifest = read_manifest(manifest_path, require_ready=False)
    if manifest["preprocessing"].get("status") != "raw":
        raise ValueError("Only explicitly raw manifests may enter the raw pipeline")
    if opts.threads < 1 or opts.workers < 1:
        raise ValueError("threads and workers must be positive")
    if bool(opts.native_cache_manifest) != bool(opts.native_cache_root):
        raise ValueError("Supply both native-cache paths")
    for attr in ("hdbet", "n4", "ants_registration", "ants_apply"):
        executable = shutil.which(getattr(opts, attr))
        if executable is None:
            raise FileNotFoundError(
                f"Install the external executable or supply its path: {attr}"
            )
        setattr(opts, attr, Path(executable))
    output = opts.output.resolve()
    work = output / "work"
    args = argparse.Namespace(
        skull_stripper="hdbet",
        hdbet=opts.hdbet,
        hdbet_device=opts.device,
        n4=opts.n4,
        ants_registration=opts.ants_registration,
        ants_apply=opts.ants_apply,
        threads_per_job=opts.threads,
        overwrite=False,
    )
    scans = []
    for key, pair in manifest["pairs"].items():
        for role, domain in (
            ("source", manifest["source_field"]),
            ("target", manifest["target_field"]),
        ):
            path = Path(pair[role])
            scans.append(
                Scan(scan_key(role, key, domain, path), role, key, domain, path)
            )
    for row in manifest["references"]:
        path = Path(row["path"])
        scans.append(
            Scan(
                scan_key("reference", row["subject"], manifest["target_field"], path),
                "reference",
                row["subject"],
                manifest["target_field"],
                path,
            )
        )
    if opts.native_cache_manifest:
        reuse_native_cache(
            scans, work, opts.native_cache_manifest, opts.native_cache_root
        )
    if opts.template_subject is not None:
        anchor = subject(opts.template_subject)
        if anchor not in manifest["train"]:
            raise ValueError("Template subject must be a training subject")
        template_scan = next(
            scan for scan in scans if scan.role == "target" and scan.subject == anchor
        )
        manifest["spatial_anchor"] = anchor
        native_template = prepare_native(template_scan, work, args)
    else:
        path = opts.template_image.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        forbidden = [
            Path(pair["target"])
            for key, pair in manifest["pairs"].items()
            if key in manifest["test"] or key in manifest["validation"]
        ]
        if path in forbidden:
            raise ValueError("An evaluation target cannot be the independent template")
        template_scan = Scan(
            "external-template", "template", "external", "template", path
        )
        native_template = path
        manifest.pop("spatial_anchor", None)
    output_pairs = {key: {} for key in manifest["pairs"]}
    output_references = []

    def prepare_one(scan):
        native = prepare_native(scan, work, args)
        registered = register_scan(
            scan, native, template_scan, native_template, work, args
        )
        destination = output / "volumes" / f"{scan.key}.nii.gz"
        statistics = subject_minmax_to_output(registered, destination)
        return scan, destination, statistics

    with ThreadPoolExecutor(max_workers=opts.workers) as executor:
        results = executor.map(prepare_one, scans)
        for index, (scan, destination, statistics) in enumerate(results, start=1):
            if scan.role == "reference":
                output_references.append(
                    {"subject": scan.subject, "path": str(destination)}
                )
            else:
                output_pairs[scan.subject][scan.role] = str(destination)
            print(
                json.dumps(
                    {"scan": index, "count": len(scans), "key": scan.key, **statistics}
                ),
                flush=True,
            )
    manifest.update(pairs=output_pairs, references=output_references)
    manifest["preprocessing"] = {
        "status": "ready",
        "population_used_for_base_normalization": False,
        "steps": [
            "HD-BET with default TTA",
            "masked N4",
            "affine registration to declared template",
            "subject-level min-max [0,1]",
        ],
        "template": str(native_template),
        "template_independent": opts.template_image is not None,
        "template_sha256": hashlib.sha256(native_template.read_bytes()).hexdigest(),
        "input_manifest": str(manifest_path),
    }
    if opts.native_cache_manifest:
        manifest["preprocessing"]["native_cache_manifest"] = str(
            opts.native_cache_manifest.resolve()
        )
        manifest["preprocessing"]["native_cache_manifest_sha256"] = hashlib.sha256(
            opts.native_cache_manifest.read_bytes()
        ).hexdigest()
    destination = output / "manifest.yaml"
    destination.write_text(yaml.safe_dump(manifest, sort_keys=False))
    read_manifest(destination)
    print(f"prepared_manifest={destination}")


if __name__ == "__main__":
    main()
