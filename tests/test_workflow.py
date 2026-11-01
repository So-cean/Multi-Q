"""Protocol tests protect subject separation and paired comparison contracts."""

import copy
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest
import yaml

from tools.workflow import check_volumes, plan, read_manifest, runtime_config


@pytest.fixture
def manifest(tmp_path):
    pairs = {}
    for key in ("0001", "0002", "0003", "0004"):
        pair = {}
        for role in ("source", "target"):
            path = tmp_path / f"{key}_{role}.nii.gz"
            values = np.zeros((16, 16, 16), dtype=np.float32)
            values[2:14, 2:14, 2:14] = np.linspace(0.1, 1.0, 12**3).reshape(12, 12, 12)
            nib.save(nib.Nifti1Image(values, np.eye(4)), path)
            pair[role] = str(path)
        pairs[key] = pair
    reference = tmp_path / "reference_0100.nii.gz"
    nib.save(nib.load(pairs["0001"]["target"]), reference)
    return {
        "dataset": "srpbs",
        "source_field": "ATT",
        "target_field": "KUT",
        "modality": "T1W",
        "protocol": "holdout",
        "train": ["0001", "0002"],
        "validation": ["0003"],
        "test": ["0004"],
        "spatial_anchor": "0001",
        "pairs": pairs,
        "references": [{"subject": "0100", "path": str(reference)}],
        "preprocessing": {
            "status": "ready",
            "population_used_for_base_normalization": False,
            "steps": ["HD-BET", "N4", "affine", "subject_minmax"],
        },
    }


def write_manifest(tmp_path, payload):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.mark.parametrize(
    "defect",
    [
        "test_overlap",
        "reference_overlap",
        "path_alias",
        "bad_anchor",
        "population_scale",
        "empty_val",
    ],
)
def test_protocol_rejects_leakage(tmp_path, manifest, defect):
    if defect == "test_overlap":
        manifest["test"] += ["0001"]
    elif defect == "reference_overlap":
        manifest["references"][0]["subject"] = "0004"
    elif defect == "path_alias":
        alias = tmp_path / "alias.nii.gz"
        alias.symlink_to(manifest["pairs"]["0004"]["target"])
        manifest["references"][0]["path"] = str(alias)
    elif defect == "bad_anchor":
        manifest["spatial_anchor"] = "0004"
    elif defect == "population_scale":
        manifest["preprocessing"]["population_used_for_base_normalization"] = True
    elif defect == "empty_val":
        manifest["validation"] = []
    with pytest.raises(ValueError):
        read_manifest(write_manifest(tmp_path, manifest))


def test_plan_keeps_test_images_outside_training_tree(tmp_path, manifest):
    path = write_manifest(tmp_path, manifest)
    output = tmp_path / "planned"
    _, record = plan(path, output, ["no_calibration", "multiq"], check=True)
    assert record["validation"] == ["0003"]
    assert not record["test_used_for_selection"]
    assert len(list((output / "inputs/Images/T1W/ATT").glob("*.nii.gz"))) == 3
    assert not list((output / "inputs/Images").rglob("*_0004.nii.gz"))
    assert (output / "inputs/TestTargets/T1W/KUT/P_T1W_KUT_0004.nii.gz").exists()
    for path in record["configs"].values():
        config = yaml.safe_load(Path(path).read_text())
        assert "0004" not in config["data"]["subject_ids"]
        assert config["validation"]["evaluation_mask"] == "intersection"


def test_both_datasets_share_learning_and_selection_config(tmp_path, manifest):
    other = copy.deepcopy(manifest)
    other.update(dataset="mrixfields", source_field="3T", target_field="7T")
    for method in ("no_calibration", "multiq"):
        a = runtime_config(manifest, tmp_path / "a", method)
        b = runtime_config(other, tmp_path / "b", method)
        assert a["model"] == b["model"]
        assert a["loss"] == b["loss"]
        assert a["training"] == b["training"]
        av, bv = a["validation"].copy(), b["validation"].copy()
        av.pop("raw_target_dir")
        bv.pop("raw_target_dir")
        assert av == bv
        assert a["data"]["patches_per_subject"] == b["data"]["patches_per_subject"]
        if method == "no_calibration":
            assert a["data"]["target_normalization"] is None
            assert "source_intensity" not in a["data"]


def test_replanning_cannot_change_test_assignment(tmp_path, manifest):
    path = write_manifest(tmp_path, manifest)
    output = tmp_path / "planned"
    plan(path, output, ["multiq"])
    manifest["train"] = ["0001", "0004"]
    manifest["validation"] = ["0002"]
    manifest["test"] = ["0003"]
    write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="different manifest"):
        plan(path, output, ["multiq"])


def test_prepared_check_rejects_population_scaled_out_of_range(tmp_path, manifest):
    path = Path(manifest["pairs"]["0002"]["source"])
    image = nib.load(path)
    values = np.asarray(image.dataobj).copy() * 2
    nib.save(nib.Nifti1Image(values, image.affine), path)
    with pytest.raises(ValueError, match="intensities"):
        check_volumes(manifest)


def test_validation_scores_uncalibrated_target_with_multiq_supervision(
    tmp_path, manifest, monkeypatch
):
    import torch
    from types import SimpleNamespace
    import train
    from multiq.cdf_prior import build_cdf_prior, save_cdf_prior
    from multiq.data import discover_pairs

    output = tmp_path / "planned"
    _, record = plan(write_manifest(tmp_path, manifest), output, ["multiq"])
    reference_path = Path(manifest["references"][0]["path"])
    reference = nib.load(reference_path)
    nib.save(
        nib.Nifti1Image(np.asarray(reference.dataobj) * 0.5, reference.affine),
        reference_path,
    )
    prior, affine = build_cdf_prior([reference_path], bins=2048, histogram_stride=4)
    save_cdf_prior(output / "reference.npz", prior, affine)
    config = yaml.safe_load(Path(record["configs"]["multiq"]).read_text())
    pair = discover_pairs(
        **{
            key: config["data"][key]
            for key in (
                "root",
                "split",
                "modality",
                "source_field",
                "target_field",
                "subject_ids",
            )
        }
    )[0]
    target = np.asarray(nib.load(pair.target).dataobj, dtype=np.float32)
    monkeypatch.setattr(train, "sliding_window_predict", lambda **kwargs: target * 0.5)
    model = SimpleNamespace(
        base_channel=0,
        output_mode="logit_residual",
        require_support_mask=True,
        residual_gate=torch.tensor(0.0),
    )
    metrics = train._validate(model, pair, config, torch.device("cpu"), tmp_path, 1)
    assert metrics["nrmse"] == pytest.approx(0.5)
    assert metrics["full_nrmse"] == pytest.approx(0.5)
    assert metrics["aligned_nrmse"] < metrics["nrmse"] / 2


def test_heldout_validation_preserves_training_pool_and_labels_selection(
    tmp_path, manifest
):
    manifest["protocol"] = "heldout_validation"
    manifest["train"] = ["0001", "0002", "0003"]
    manifest["validation"] = []
    resolved, record = plan(
        write_manifest(tmp_path, manifest), tmp_path / "heldout", ["multiq"]
    )
    assert resolved["train"] == ["0001", "0002", "0003"]
    assert resolved["validation"] == ["0004"] and not resolved["test"]
    assert record["evaluation_role"] == "validation"
    assert record["reported_cohort_used_for_selection"]
    config = yaml.safe_load(Path(record["configs"]["multiq"]).read_text())
    from multiq.data import discover_pairs
    from train import _split_pairs

    pairs = discover_pairs(
        **{
            key: config["data"][key]
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
    training, validation = _split_pairs(pairs, "split", config["data"])
    assert [p.subject_id for p in training] == resolved["train"]
    assert [p.subject_id for p in validation] == resolved["validation"]
    assert config["data"]["patches_per_subject"] == 256
    assert config["data"]["target_normalization"]["support_mask"] == "source"


def test_removed_nested_protocol_cannot_silently_run(tmp_path, manifest):
    manifest["protocol"] = "nested"
    with pytest.raises(ValueError, match="nested"):
        read_manifest(write_manifest(tmp_path, manifest))
