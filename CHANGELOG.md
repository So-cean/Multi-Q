# Changelog

## Unreleased

- Fix the strict-sampler flag lookup to match the original MRIxFields trainer:
  read `training.subject_balanced_batches`, not the legacy `data` declaration.
- Add a real-data GPU smoke covering both calibration settings and the full
  training-to-evaluation chain before formal submissions.

- Restore the original MRIxFields early-stopping base recipe, per-subject patch
  count, calibration foreground and checkpoint-selection functions.
- Remove the added nested duration/refit workflow.
- Disable training flips and inference TTA in the release protocol.
- Label validation-based reporting separately from independent test evaluation.

## v0.1.0

- Standalone Multi-Q calibration and 3D UNet++ implementation.
- Shared MRIxFields/SRPBS preparation, selection, training and evaluation workflow.
- Validation early stopping, full-volume SSIM guard and training-only nested selection.
- Fixed supervised reconstruction/perceptual objective and EMA inference.
- Removed historical pretraining, alternative backbones, atlas transport, GAN,
  dynamic loss balancing and segmentation-loss branches.
