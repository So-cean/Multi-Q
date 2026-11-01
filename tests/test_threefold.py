"""Exercise balanced assignments and reuse of native preprocessing outputs."""

import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from multiq.preprocessing import Scan, reuse_native_cache
from tools.prepare_threefold import threefold_assignments
from tools.workflow import plan
from train import _split_pairs
from multiq.data import discover_pairs


def test_threefold_keeps_three_disjoint_subjects_per_role(tmp_path):
    pairs = {}
    for index in range(1, 10):
        key = f"{index:04d}"
        pairs[key] = {}
        for role in ("source", "target"):
            path = tmp_path / f"{key}_{role}.nii.gz"
            array = np.zeros((16, 16, 16), dtype=np.float32)
            array[2:14, 2:14, 2:14] = 0.5
            nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            pairs[key][role] = str(path)
    reference = tmp_path / "ref_0100.nii.gz"
    nib.save(nib.load(pairs["0001"]["target"]), reference)
    base = dict(
        dataset="srpbs",
        source_field="ATT",
        target_field="KUT",
        modality="T1W",
        pairs=pairs,
        references=[dict(subject="0100", path=str(reference))],
        preprocessing=dict(
            status="ready",
            population_used_for_base_normalization=False,
            steps=["HD-BET", "N4", "affine", "subject_minmax"],
            template_independent=True,
        ),
    )
    folds = threefold_assignments(base)
    for role in ("train", "validation", "test"):
        assert sorted(subject for fold in folds for subject in fold[role]) == sorted(
            pairs
        )
    for fold in folds:
        assert all(len(fold[role]) == 3 for role in ("train", "validation", "test"))
        assert not set(fold["train"]) & set(fold["validation"])
        assert not set(fold["test"]) & (set(fold["train"]) | set(fold["validation"]))
        path = tmp_path / f"{fold['split_id']}.yaml"
        path.write_text(yaml.safe_dump(fold))
        _, record = plan(
            path, tmp_path / fold["split_id"], ["no_calibration", "multiq"]
        )
        for config_path in record["configs"].values():
            config = yaml.safe_load(Path(config_path).read_text())
            data = config["data"]
            discovered = discover_pairs(
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
            training, validation = _split_pairs(discovered, "split", data)
            assert [pair.subject_id for pair in training] == fold["train"]
            assert [pair.subject_id for pair in validation] == fold["validation"]
            assert not set(data["subject_ids"]) & set(fold["test"])
            assert data["patches_per_subject"] == 256
    bad = copy.deepcopy(base)
    bad["spatial_anchor"] = "0001"
    with pytest.raises(ValueError, match="template"):
        threefold_assignments(bad)


def test_native_cache_checks_source_and_reuses_only_native_outputs(tmp_path):
    source = tmp_path / "source.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4)), np.eye(4)), source)
    native = tmp_path / "cache/native/original"
    native.mkdir(parents=True)
    for name in ("n4_brain.nii.gz", "mask.nii.gz"):
        nib.save(nib.load(source), native / name)
    manifest = tmp_path / "cache.json"
    manifest.write_text(
        json.dumps(
            dict(
                preprocessing=dict(skull_strip="HD-BET", bias_correction="masked N4"),
                scans=[
                    dict(
                        source=str(source),
                        source_bytes=source.stat().st_size,
                        key="original",
                    )
                ],
            )
        )
    )
    scan = Scan("current", "source", "0001", "ATT", source)
    work = tmp_path / "new"
    reuse_native_cache([scan], work, manifest, tmp_path / "cache")
    assert (
        work / "native/current/n4_brain.nii.gz"
    ).resolve() == native / "n4_brain.nii.gz"
    assert not (work / "registered").exists()
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source mismatch"):
        reuse_native_cache([scan], work, manifest, tmp_path / "cache")
