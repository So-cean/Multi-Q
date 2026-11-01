# Data and experiment workflow

## Prepare MRI volumes

`tools/prepare_images.py` applies the same per-image chain to either dataset:
HD-BET with default TTA → masked N4 → affine registration → subject-level min–max.
An independent template defines the shared grid, or a training subject supplies
it and stays in training across all splits.

```bash
python tools/prepare_images.py --manifest local/raw.yaml \
  --output data/prepared --template-subject 0001 --device cuda
```

For existing prepared images, use `tools/import_prepared.py --help` to create a
manifest and record their actual preprocessing history. The raw-image tools
resolve HD-BET and ANTs from PATH or explicit executable arguments.

## Declare subject roles

A manifest supplies each pair's source/target paths, reference subjects, train,
validation and test IDs, the protocol, and preprocessing history. See
`examples/mrixfields.yaml` and `examples/srpbs.yaml`. Replace the sample reference
entries with the complete eligible population. `MULTIQ_DATA` expands to a local
data root.

Planning writes `plan.json`, runtime YAML configs and an image layout whose
`Images` directory contains training/validation pairs. Test pairs occupy
`TestInputs` and `TestTargets`. `--check-volumes` audits image ranges and grids;
execution performs this audit on the compute node.

## Train and select a checkpoint

The baseline is the original MRIxFields N4-clean early-stopping recipe recorded
in `mrixfields_early_stopping_recipe.json`: 3D UNet++, AdamW, EMA, 256 patches
per training subject per epoch, physical batch four and two-step accumulation.
Its loss weights are L1=1, MSE=0.5, SSIM=2, gradient=0.1 and high-frequency=0.05.
The original base recipe has no perceptual training loss. Source/target Multi-Q
uses the original quantiles; target statistics use their foreground intersection.
Training flips and inference TTA are disabled in the release configuration.

Validation maximizes SSIM against the original prepared target, with the original
full-volume SSIM guard of 0.002. It runs every five epochs, with twelve validation
checks of patience and a maximum of 120 epochs. Epoch zero is eligible.

- `holdout`: train on `train`, select a checkpoint on `validation`, then score
  the independent `test` cohort.
- `heldout_validation`: train on `train` and use `validation` for selection and
  reporting. The result is labelled validation, with no independent test claim.
- `fixed_epoch`: fit on `train` for the declared duration and score `test`.

No workflow computes a median duration across inner folds or refits a model at
that duration. The SRPBS 3/3/3 assignments are fixed in
[SRPBS_PROTOCOL.md](SRPBS_PROTOCOL.md).

## Execute stages

Run a real-data GPU smoke before submitting the formal fits:

```bash
python tools/smoke_training.py --plan runs/experiment/plan.json \
  --output runs/smoke --code-commit COMMIT_SHA
```

It keeps the full model, actual subject count, batch four and per-subject patch
budget, and runs both methods for one epoch through validation, checkpoint
reload, inference and evaluation. `SMOKE_PASSED.json` is written only after both
methods complete every stage. Smoke outputs are separate from paper results.

```bash
python tools/workflow.py --manifest local/experiment.yaml --output runs/experiment \
  --execute --stage reference
```

Repeat with `fit`, `infer` and `evaluate`, or use `--stage all`.
`--methods multiq` and `--methods no_calibration` support separate runs. Scheduler
scheduler scripts should activate an environment containing the declared
dependencies.

Inference uses source support and EMA weights; TTA defaults to `none`. Evaluation
reports full-volume nRMSE, axial SSIM and AlexNet LPIPS with the same source–target
foreground intersection for both methods. `tools/aggregate.py` preserves split
and subject identities when combining repeated assignments.

Grid coordinate channels cover [-1,1]. Their anatomical interpretation follows
the prepared spatial grid: MNI for MNI-registered data, or the declared subject
template for the current SRPBS images.
