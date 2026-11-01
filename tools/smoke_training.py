#!/usr/bin/env python3
"""Run both methods through one real GPU epoch, validation, inference and scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from multiq.config import load_config
from multiq.data import discover_pairs
from train import _split_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-training-subjects", type=int, default=3)
    parser.add_argument("--expected-batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke requires an allocated CUDA GPU")
    os.umask(0o002)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "SMOKE_PASSED.json").exists():
        raise FileExistsError("Use a fresh output directory for another smoke")
    plan = json.loads(args.plan.read_text())
    started = time.time()
    results = []

    def phase(method, stage):
        state = dict(
            method=method,
            stage=stage,
            code_commit=args.code_commit,
            elapsed_seconds=time.time() - started,
        )
        (args.output / "state.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state), flush=True)

    def run(*command):
        subprocess.run([sys.executable, "-u", *map(str, command)], cwd=ROOT, check=True)

    try:
        for method in ("no_calibration", "multiq"):
            source_path = Path(plan["configs"][method])
            original = load_config(source_path)
            data, training = original["data"], original["training"]
            pairs = discover_pairs(
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
            train_pairs, validation_pairs = _split_pairs(pairs, "split", data)
            assert len(train_pairs) == args.expected_training_subjects
            assert len(validation_pairs) == 3
            assert int(training["batch_size"]) == args.expected_batch_size
            assert int(data["patches_per_subject"]) == 256
            assert data["patch_size"] == [96, 96, 96]
            config = copy.deepcopy(original)
            config.pop("_config_path", None)
            config["experiment"]["output_dir"] = str(args.output / "runs")
            # Keep the model, objective, data and optimizer unchanged. Only the
            # output location and command-line duration differ from the real fit.
            config_path = args.output / f"{method}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            phase(method, "training_and_validation")
            run(
                ROOT / "train.py",
                "--config",
                config_path,
                "--fold",
                "split",
                "--epochs",
                "1",
            )
            directory = (
                args.output / "runs" / config["experiment"]["name"] / "foldsplit"
            )
            summary = json.loads((directory / "run_summary.json").read_text())
            history = json.loads((directory / "history.json").read_text())
            batches = math.ceil(
                len(train_pairs) * data["patches_per_subject"] / training["batch_size"]
            )
            updates = math.ceil(batches / training["gradient_accumulation_steps"])
            assert summary["terminal_epoch"] == 1 and summary["global_step"] == updates
            assert summary["parameter_count"] == 50110082
            assert [row["epoch"] for row in history] == [0, 1]
            assert all(
                math.isfinite(float(value)) for value in history[-1]["train"].values()
            )
            for row in history:
                for key in ("ssim", "nrmse", "lpips", "full_ssim"):
                    assert math.isfinite(float(row["validation"][key]))
            selection = json.loads(
                (directory / "checkpoint_selection.json").read_text()
            )
            assert (
                selection["selected_epoch"] in (0, 1)
                and (directory / "best.pth").exists()
            )
            checkpoint_path = directory / "last.pth"
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            assert (
                checkpoint["global_step"] == updates
                and checkpoint["optimizer"]["state"]
            )
            assert checkpoint["ema_model"]
            del checkpoint
            predictions = args.output / "predictions" / method
            phase(method, "checkpoint_reload_and_inference")
            run(
                ROOT / "infer.py",
                "--checkpoint",
                checkpoint_path,
                "--input-dir",
                Path(data["root"])
                / "TestInputs"
                / data["modality"]
                / data["source_field"],
                "--output-dir",
                predictions,
                "--tta",
                "none",
            )
            files = sorted(predictions.glob("*.nii.gz"))
            assert len(files) == 3
            for path in files:
                image = nib.load(path)
                values = np.asarray(image.dataobj)
                source = nib.load(
                    Path(data["root"])
                    / "TestInputs"
                    / data["modality"]
                    / data["source_field"]
                    / path.name
                )
                assert values.shape == source.shape
                np.testing.assert_allclose(
                    image.affine, source.affine, rtol=0, atol=1e-4
                )
                assert (
                    np.isfinite(values).all()
                    and values.min() >= 0
                    and values.max() <= 1
                )
            evaluation = args.output / "evaluation" / method
            phase(method, "full_volume_evaluation")
            run(
                ROOT / "tools/evaluate.py",
                "--data-root",
                data["root"],
                "--prediction",
                f"{method}={predictions}",
                "--output",
                evaluation,
                "--lpips-device",
                "cuda",
            )
            metric_summary = json.loads((evaluation / "summary.json").read_text())
            assert metric_summary["status"] == "complete"
            for key in ("nrmse", "ssim", "lpips"):
                assert math.isfinite(
                    float(metric_summary["methods"][method][key]["mean"])
                )
            results.append(
                dict(
                    method=method,
                    training_subjects=[p.subject_id for p in train_pairs],
                    validation_subjects=[p.subject_id for p in validation_pairs],
                    batch_size=training["batch_size"],
                    patches_per_subject=data["patches_per_subject"],
                    optimizer_updates=updates,
                    peak_gpu_memory_bytes=summary["peak_gpu_memory_bytes"],
                    source_config_sha256=hashlib.sha256(
                        source_path.read_bytes()
                    ).hexdigest(),
                    smoke_config_sha256=hashlib.sha256(
                        config_path.read_bytes()
                    ).hexdigest(),
                    checkpoint=str(checkpoint_path),
                    predictions=len(files),
                    evaluation=str(evaluation),
                )
            )
        payload = dict(
            status="passed",
            code_commit=args.code_commit,
            torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name(0),
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            elapsed_seconds=time.time() - started,
            methods=results,
        )
        (args.output / "SMOKE_PASSED.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        phase("both", "passed")
    except BaseException as error:
        (args.output / "SMOKE_FAILED.json").write_text(
            json.dumps(dict(error=repr(error), methods_completed=results), indent=2)
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
