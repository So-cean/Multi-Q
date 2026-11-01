# SRPBS: rotating 3/3/3 experiment

The ATT-to-KUT T1W comparison uses nine travelling subjects and 159 independent
healthy KUT OPEN reference subjects. Each fold trains two models from scratch:
no calibration and Multi-Q with source-and-target calibration.

| Fold | Training | Validation | Test |
|---|---|---|---|
| 1 | 01, 02, 03 | 04, 05, 06 | 07, 08, 09 |
| 2 | 04, 05, 06 | 07, 08, 09 | 01, 02, 03 |
| 3 | 07, 08, 09 | 01, 02, 03 | 04, 05, 06 |

Each subject has one training, one validation and one test assignment. Each
method produces exactly one independent test prediction per subject. Final
aggregation uses test predictions, retaining fold and subject identities.

## Preparation

Reuse the recorded native HD-BET/masked-N4 images. Register each image directly
from its native N4 grid to independent OPEN subject 0715, the first eligible
reference ID, then apply per-image min-max normalization to [0,1]. This replaces
the earlier paired-subject-01 registration template. The same prepared images
and frozen reference are shared by all folds and methods. All 159 OPEN subjects
remain independent of the paired training, validation and test subjects.

Coordinate channels describe the independent template grid. They are not labelled
as MNI coordinates. No pooled population scale enters base normalization.

## Training and checkpoint selection

Use the original MRIxFields N4-clean early-stopping base recipe pinned in
`mrixfields_early_stopping_recipe.json`: 256 patches per training subject per
epoch, 96^3 patches, batch four, two-step accumulation, AdamW, EMA, seed 2026,
and a maximum of 120 epochs. Three training subjects give 96 optimizer updates
per epoch. Training flips and synthesis inference TTA are disabled.

The original trainer reads optional strict per-batch balancing from
`training.subject_balanced_batches`. The base recipe leaves that option off:
the loader shuffles all 768 patch indices, visiting each subject's 256 indices
once per epoch. The legacy flag under `data` does not activate that sampler.

Validate on the three declared validation subjects at epoch zero, epoch one,
and every five epochs thereafter. Average subject-level SSIM, apply the original
full-volume SSIM guard, and stop after twelve non-improving validation checks.
Use the selected EMA checkpoint directly for the three test subjects. No model
refitting, fold ensemble or test-based checkpoint selection is performed.

Score original prepared targets over full volumes, with the same fixed
source-target foreground intersection for both methods. Report nRMSE, SSIM and
AlexNet LPIPS. Preserve the earlier five-training/four-test pilot as a separate
historical experiment.

## Plan

```bash
python tools/prepare_threefold.py --manifest data/prepared/manifest.yaml \
  --output runs/srpbs_threefold
```

This writes the three manifests and six runtime configurations without submitting
jobs. Material preprocessing, training and evaluation run on designated nodes.
