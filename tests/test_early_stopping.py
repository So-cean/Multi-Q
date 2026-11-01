"""Exercise the real training loop and verify validation-driven termination."""

import json
import sys

import nibabel as nib
import numpy as np
import torch

import train
from multiq.config import load_config
from pathlib import Path
import yaml


def test_training_actually_stops_and_keeps_best_validation_checkpoint(
    tmp_path, monkeypatch
):
    torch.set_num_threads(2)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/common.yaml")
    config.pop("_config_path", None)
    config["experiment"].update(
        name="early_stop_smoke", output_dir=str(tmp_path / "runs")
    )
    config["data"].update(
        root=str(tmp_path / "data"),
        source_field="ATT",
        target_field="KUT",
        subject_ids=["0001", "0002"],
        patch_size=[16, 16, 16],
        patches_per_subject=2,
    )
    config["model"].update(base_channels=2, levels=2, blocks_per_level=1)
    config["loss"].update(
        mse_weight=0,
        ssim_weight=0,
        gradient_weight=0,
        high_frequency_weight=0,
        perceptual_weight=0,
    )
    config["training"].update(
        batch_size=1,
        gradient_accumulation_steps=1,
        fused_optimizer=False,
        amp=False,
        save_last_every=1,
    )
    config["validation"].update(
        every=1,
        patience=1,
        early_stopping_min_epoch=1,
        compute_official_lpips=False,
        raw_target_dir=None,
    )
    for key in config["data"]["subject_ids"]:
        for field, value in (("ATT", 0.5), ("KUT", 0.6)):
            directory = tmp_path / "data/Images/T1W" / field
            directory.mkdir(parents=True, exist_ok=True)
            array = np.zeros((16, 16, 16), dtype=np.float32)
            array[2:14, 2:14, 2:14] = value
            nib.save(
                nib.Nifti1Image(array, np.eye(4)),
                directory / f"P_T1W_{field}_{key}.nii.gz",
            )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    seen = []

    def validate(model, pairs, config, device, run_dir, epoch, **kwargs):
        seen.append((epoch, [pair.subject_id for pair in pairs]))
        return {
            "ssim": 0.9 if epoch == 0 else 0.8,
            "full_ssim": 0.9 if epoch == 0 else 0.8,
            "nrmse": 0.1,
            "psnr": 20.0,
            "sharpness_ratio": 1.0,
        }

    monkeypatch.setattr(train, "_validate_pairs", validate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--config", str(path), "--fold", "1", "--epochs", "5"]
    )
    train.main()
    output = tmp_path / "runs/early_stop_smoke/fold1"
    summary = json.loads((output / "run_summary.json").read_text())
    selection = json.loads((output / "checkpoint_selection.json").read_text())
    assert summary["terminal_epoch"] == 1
    assert selection["selected_epoch"] == 0
    assert seen == [(0, ["0002"]), (1, ["0002"])]
    assert (output / "best.pth").is_file()
