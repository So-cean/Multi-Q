"""Verify resumed optimization and inference from a self-contained checkpoint."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

import infer
import train
from multiq.config import load_config


def test_resume_reproduces_next_epoch_and_checkpoint_predicts(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/common.yaml")
    config.pop("_config_path")
    config["experiment"].update(name="roundtrip", output_dir=str(tmp_path / "runs"))
    config["data"].update(
        root=str(tmp_path / "data"),
        source_field="source",
        target_field="target",
        subject_ids=["0001", "0002", "0003"],
        patch_size=[16, 16, 16],
        patches_per_subject=4,
    )
    config["model"].update(base_channels=2, levels=2, blocks_per_level=1)
    config["loss"].update(perceptual_weight=0)
    config["training"].update(
        epochs=2,
        batch_size=4,
        gradient_accumulation_steps=2,
        fused_optimizer=False,
        amp=False,
        save_every=1,
        save_last_every=1,
    )
    config["validation"].update(patch_size=[16, 16, 16], sliding_window_batch_size=1)
    affine = np.diag([1.2, 1.1, 1.3, 1.0])
    for key in config["data"]["subject_ids"]:
        for field, factor in (("source", 0.7), ("target", 0.8)):
            directory = tmp_path / "data/Images/T1W" / field
            directory.mkdir(parents=True, exist_ok=True)
            values = np.zeros((16, 16, 16), dtype=np.float32)
            values[2:14, 2:14, 2:14] = np.linspace(0.1, factor, 12**3).reshape(
                12, 12, 12
            )
            nib.save(
                nib.Nifti1Image(values, affine),
                directory / f"P_T1W_{field}_{key}.nii.gz",
            )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--config", str(config_path), "--fold", "full"]
    )
    train.main()
    run = tmp_path / "runs/roundtrip/full"
    original = torch.load(run / "final.pth", map_location="cpu", weights_only=False)
    # Three microbatches per epoch include one partial accumulation window.
    assert original["global_step"] == 4

    config["experiment"]["name"] = "resumed"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--fold",
            "full",
            "--resume",
            str(run / "epoch0001.pth"),
        ],
    )
    train.main()
    resumed = torch.load(
        tmp_path / "runs/resumed/full/final.pth", map_location="cpu", weights_only=False
    )
    assert resumed["epoch"] == original["epoch"] == 2
    assert resumed["global_step"] == original["global_step"]
    for state in ("model", "ema_model"):
        for key in original[state]:
            torch.testing.assert_close(
                original[state][key], resumed[state][key], rtol=0, atol=0
            )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--checkpoint",
            str(run / "final.pth"),
            "--input-dir",
            str(tmp_path / "data/Images/T1W/source"),
            "--output-dir",
            str(tmp_path / "predictions"),
        ],
    )
    infer.main()
    outputs = sorted((tmp_path / "predictions").glob("*.nii.gz"))
    assert len(outputs) == 3
    for path in outputs:
        output = nib.load(path)
        values = output.get_fdata()
        np.testing.assert_allclose(output.affine, affine, atol=1e-6)
        assert values.shape == (16, 16, 16)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        assert not values[:2].any()
