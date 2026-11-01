"""Protect the restored MRIxFields recipe and its checkpoint decisions."""

import json
from pathlib import Path

import pytest

from multiq.config import load_config
from train import _add_constrained_lpips_prescreen


ROOT = Path(__file__).resolve().parents[1]


def test_recipe_matches_frozen_mrixfields_source():
    original = json.loads(
        (ROOT / "docs/mrixfields_early_stopping_recipe.json").read_text()
    )["source_recipe"]
    current = load_config(ROOT / "configs/common.yaml")
    assert current["model"] == original["model"]
    assert current["training"] == original["training"]
    assert current["loss"] == {
        key: value
        for key, value in original["loss"].items()
        if key != "target_normalization"
    }
    for key in (
        "patch_size",
        "patches_per_subject",
        "edge_probability",
        "foreground_probability",
        "training_pair_mask",
    ):
        assert current["data"][key] == original["data"][key]
    for key in (
        "every",
        "patience",
        "min_delta",
        "before_training",
        "monitor",
        "guard_monitor",
        "guard_tolerance",
    ):
        assert current["validation"][key] == original["validation"][key]
    assert current["data"]["spatial_flip_augmentation"] == "none"
    assert current["inference"]["tta"] == "none"
    assert current["validation"]["slice_range"] is None
    assert current["validation"]["evaluation_mask"] == "intersection"


def test_constrained_selector_keeps_original_eligibility_and_score():
    reference = {"ssim": 0.9, "nrmse": 0.2, "lpips": 0.1, "full_ssim": 0.9}
    settings = {
        "constraints": {
            "ssim_min_delta": -0.003,
            "nrmse_max_delta": 0.006,
            "lpips_max_delta": 0.0,
            "full_ssim_min_delta": -0.003,
        }
    }
    eligible = {"ssim": 0.899, "nrmse": 0.201, "lpips": 0.08, "full_ssim": 0.899}
    rejected = {**eligible, "full_ssim": 0.89}
    _add_constrained_lpips_prescreen(eligible, reference, settings)
    _add_constrained_lpips_prescreen(rejected, reference, settings)
    assert eligible["selection_eligible"] == 1.0
    assert eligible["constrained_lpips_prescreen"] == pytest.approx(0.2)
    assert rejected["selection_eligible"] == 0.0
    assert rejected["constrained_lpips_prescreen"] == pytest.approx(-0.8)
