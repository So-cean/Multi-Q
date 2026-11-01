"""Numerical checks for scalar calibration and untouched evaluation targets."""

import numpy as np
import pytest

from multiq.cdf_prior import (
    histogram_quantiles,
    histogram_scale_to_reference,
    scale_volume_to_histogram_reference,
)
from multiq.evaluation import mask_for_evaluation
from multiq.metrics import evaluate_volume
from train import _guarded_checkpoint_selection, _split_pairs


def test_multiq_is_one_zero_intercept_gain():
    reference = np.exp(-(((np.linspace(0, 1, 2048) - 0.5) / 0.1) ** 2))
    source = np.zeros((16, 16, 16), dtype=np.float32)
    source[1:, 1:, 1:] = np.linspace(0.02, 0.9, 15**3).reshape(15, 15, 15)
    calibrated, gain = scale_volume_to_histogram_reference(
        source,
        reference,
        quantiles=[0.1, 0.2, 0.5, 0.8, 0.9],
        histogram_stride=1,
        scale_range=(0.25, 4),
        clip_output=False,
    )
    np.testing.assert_allclose(calibrated, source * gain, rtol=1e-6)
    assert np.all(calibrated[source == 0] == 0)
    h1, h2 = np.arange(1, 65), np.arange(64, 0, -1)
    p = [0.1, 0.5, 0.9]
    q, r = histogram_quantiles(h1, p), histogram_quantiles(h2, p)
    expected = np.clip(q @ r / (q @ q), 0.25, 4)
    assert histogram_scale_to_reference(h1, h2, p, (0.25, 4)) == pytest.approx(expected)


def test_scoring_uses_raw_target_and_same_mask_for_methods():
    target = np.zeros((16, 16, 16), dtype=np.float32)
    target[2:14, 2:14, 2:14] = 0.6
    source = target.copy()
    prediction = 0.5 * target
    scored, scored_target = mask_for_evaluation(prediction, target, source)
    np.testing.assert_array_equal(scored_target, target)
    assert evaluate_volume(scored, scored_target)["nrmse"] == pytest.approx(0.5)
    assert np.all(target[2:14, 2:14, 2:14] == np.float32(0.6))


def test_higher_fold_indices_and_explicit_validation_roles():
    from types import SimpleNamespace

    pairs = [SimpleNamespace(subject_id=f"{i:04d}") for i in range(1, 7)]
    training, validation = _split_pairs(pairs, "4")
    assert len(training) == 5 and validation[0].subject_id == "0005"
    with pytest.raises(ValueError):
        _split_pairs(pairs, "-1")
    training, validation = _split_pairs(
        pairs,
        "split",
        {
            "train_subject_ids": ["0001", "0002", "0003", "0004"],
            "validation_subject_ids": ["0005", "0006"],
        },
    )
    assert len(training) == 4 and len(validation) == 2
    with pytest.raises(ValueError):
        _split_pairs(
            pairs,
            "split",
            {"train_subject_ids": ["0001"], "validation_subject_ids": ["0001"]},
        )


def test_legacy_guard_selection_is_preserved():
    result = _guarded_checkpoint_selection(
        {"epoch": 45, "ssim": 0.923, "full_ssim": 0.939},
        {"epoch": 55, "ssim": 0.921, "full_ssim": 0.9425},
        "ssim",
        "full_ssim",
        0.002,
    )
    assert result["selected_epoch"] == 55
    assert result["selected_checkpoint"] == "best_guard.pth"
