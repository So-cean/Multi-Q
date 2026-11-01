"""HD-BET, N4, affine alignment and per-image min-max preprocessing."""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Sequence
import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class Scan:
    key: str
    role: str
    subject: str
    site: str
    source: Path


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not label:
        raise ValueError(f"Cannot form a safe label from {value!r}")
    return label


def scan_key(role: str, subject: str, site: str, path: Path) -> str:
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
    return safe_label(f"{role}-{subject}-{site}-{digest}")


def reuse_native_cache(
    scans: Sequence[Scan], work: Path, cache_manifest: Path, cache_root: Path
) -> None:
    """Reuse recorded per-image HD-BET/N4 outputs before fresh registration."""
    record = json.loads(cache_manifest.read_text())
    steps = record["preprocessing"]
    if (
        "HD-BET" not in steps.get("skull_strip", "")
        or steps.get("bias_correction") != "masked N4"
    ):
        raise ValueError("The native cache must record HD-BET and masked N4")
    cached = {str(Path(row["source"]).resolve()): row for row in record["scans"]}
    for scan in scans:
        row = cached.get(str(scan.source.resolve()))
        if row is None or scan.source.stat().st_size != row["source_bytes"]:
            raise ValueError(f"Native-cache source mismatch: {scan.source}")
        directory = work / "native" / scan.key
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("n4_brain.nii.gz", "mask.nii.gz"):
            source = cache_root.resolve() / "native" / row["key"] / name
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = directory / name
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise FileExistsError(destination)
            else:
                destination.symlink_to(source)


def atomic_save_nifti(
    path: Path, values: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        output_header = header.copy()
        output_header.set_data_dtype(np.float32)
        nib.save(
            nib.Nifti1Image(
                np.asarray(values, dtype=np.float32), affine, output_header
            ),
            str(temporary),
        )
        temporary.chmod(0o664)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def copy_spatial_forms(reference_path: Path, image_path: Path) -> None:
    """Make a derived image use the reference's exact qform and sform.

    Derived masks preserve the voxel grid, but for oblique NIfTI inputs it can
    reconstruct the qform from the sform.  The resulting sub-millimetre header
    difference is harmless spatially yet exceeds ITK's strict mask tolerance.
    Copying both forms from the source avoids relaxing geometry checks globally.
    """
    reference = nib.load(str(reference_path))
    image = nib.load(str(image_path))
    if reference.shape[:3] != image.shape[:3]:
        raise ValueError(
            f"Cannot copy spatial forms across different grids: "
            f"{reference.shape[:3]} versus {image.shape[:3]}"
        )
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    values = np.asarray(image.dataobj)
    header = image.header.copy()
    output = nib.Nifti1Image(values, reference.affine, header)
    output.set_qform(qform, int(qcode))
    output.set_sform(sform, int(scode))
    with tempfile.NamedTemporaryFile(
        suffix=".nii.gz", dir=image_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        nib.save(output, str(temporary))
        temporary.chmod(0o664)
        temporary.replace(image_path)
    finally:
        temporary.unlink(missing_ok=True)


def run(command: Sequence[str], log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(
            list(command),
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
        )


def prepare_native(scan: Scan, work_root: Path, args: argparse.Namespace) -> Path:
    scan_dir = work_root / "native" / scan.key
    result = scan_dir / "n4_brain.nii.gz"
    mask = scan_dir / "mask.nii.gz"
    if result.is_file() and mask.is_file() and (not args.overwrite):
        return result
    scan_dir.mkdir(parents=True, exist_ok=True)
    stripped = scan_dir / "stripped.nii.gz"
    corrected = scan_dir / "n4.nii.gz"
    log = scan_dir / "preprocess.log"
    for path in (stripped, mask, corrected, result):
        if args.overwrite:
            path.unlink(missing_ok=True)
    n4_input = scan.source
    hdbet_input = scan.source
    if not str(hdbet_input).endswith(".nii.gz"):
        hdbet_input = scan_dir / "hdbet_input.nii.gz"
        if args.overwrite or not hdbet_input.is_file():
            nib.save(nib.load(str(scan.source)), str(hdbet_input))
    n4_input = hdbet_input
    run(
        [
            str(args.hdbet),
            "-i",
            str(hdbet_input),
            "-o",
            str(stripped),
            "-device",
            args.hdbet_device,
            "--save_bet_mask",
        ],
        log,
    )
    generated_mask = stripped.with_name(stripped.name[:-7] + "_bet.nii.gz")
    if not generated_mask.is_file():
        raise RuntimeError(f"HD-BET did not create its mask for {scan.key}")
    generated_mask.replace(mask)
    copy_spatial_forms(scan.source, mask)
    aligned_mask = scan_dir / "mask_aligned.nii.gz"
    run(
        [
            str(args.ants_apply),
            "-d",
            "3",
            "-i",
            str(mask),
            "-r",
            str(n4_input),
            "-o",
            str(aligned_mask),
            "-n",
            "NearestNeighbor",
            "-t",
            "identity",
        ],
        log,
    )
    aligned_mask.replace(mask)
    run(
        [
            str(args.n4),
            "-d",
            "3",
            "-i",
            str(n4_input),
            "-x",
            str(mask),
            "-s",
            "2",
            "-b",
            "[200]",
            "-c",
            "[50x50x30x20,1e-7]",
            "-o",
            str(corrected),
        ],
        log,
    )
    image = nib.load(str(corrected))
    values = np.asarray(image.dataobj, dtype=np.float32)
    mask_values = np.asarray(nib.load(str(mask)).dataobj) > 0
    values = np.where(mask_values & np.isfinite(values), np.maximum(values, 0), 0)
    atomic_save_nifti(result, values, image.affine, image.header)
    return result


def register_scan(
    scan: Scan,
    native: Path,
    template_scan: Scan,
    template: Path,
    work_root: Path,
    args: argparse.Namespace,
) -> Path:
    scan_dir = work_root / "registered" / scan.key
    result = scan_dir / "brain.nii.gz"
    result_mask = scan_dir / "mask.nii.gz"
    if result.is_file() and result_mask.is_file() and not args.overwrite:
        return result
    scan_dir.mkdir(parents=True, exist_ok=True)
    if scan.key == template_scan.key:
        shutil.copy2(native, result)
        shutil.copy2(work_root / "native" / scan.key / "mask.nii.gz", result_mask)
        return result
    prefix = scan_dir / "affine_"
    warped = scan_dir / "affine_Warped.nii.gz"
    transform = scan_dir / "affine_0GenericAffine.mat"
    log = scan_dir / "registration.log"
    env = dict(os.environ)
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads_per_job)
    run(
        [
            str(args.ants_registration),
            "-d",
            "3",
            "-f",
            str(template),
            "-m",
            str(native),
            "-t",
            "a",
            "-o",
            str(prefix),
            "-n",
            str(args.threads_per_job),
        ],
        log,
        env,
    )
    if not warped.is_file() or not transform.is_file():
        raise RuntimeError(
            f"ANTs did not create expected affine outputs for {scan.key}"
        )
    run(
        [
            str(args.ants_apply),
            "-d",
            "3",
            "-i",
            str(work_root / "native" / scan.key / "mask.nii.gz"),
            "-r",
            str(template),
            "-o",
            str(result_mask),
            "-n",
            "NearestNeighbor",
            "-t",
            str(transform),
        ],
        log,
        env,
    )
    image = nib.load(str(warped))
    values = np.asarray(image.dataobj, dtype=np.float32)
    mask_values = np.asarray(nib.load(str(result_mask)).dataobj) > 0
    values = np.where(mask_values & np.isfinite(values), np.maximum(values, 0), 0)
    atomic_save_nifti(result, values, image.affine, image.header)
    return result


def subject_minmax_to_output(source: Path, destination: Path) -> dict[str, float]:
    """Normalize one masked volume independently without population leakage."""
    image = nib.load(str(source))
    values = np.asarray(image.dataobj, dtype=np.float32)
    finite = np.isfinite(values)
    foreground = finite & (values > 0)
    if not foreground.any():
        raise ValueError(f"No positive foreground in {source}")
    minimum = float(values[finite].min())
    maximum = float(values[foreground].max())
    if not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"Invalid subject range [{minimum}, {maximum}] in {source}")
    normalized = np.zeros(values.shape, dtype=np.float32)
    normalized[finite] = np.clip(
        (values[finite] - minimum) / (maximum - minimum), 0.0, 1.0
    )
    atomic_save_nifti(destination, normalized, image.affine, image.header)
    return {"foreground_min": minimum, "foreground_max": maximum}
