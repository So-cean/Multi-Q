# Multi-Q

**Multi-Quantile intensity calibration for MRI synthesis with few paired subjects.**

Multi-Q builds a frozen intensity reference from unpaired target-domain images.
It fits one positive scalar gain to each source image and paired training target,
then learns the remaining mapping with a supervised 3D UNet++.

MRIxFields and SRPBS share the same calibration, network, losses, training,
validation and evaluation code.

## Setup

Use Python 3.10+ with PyTorch, torchvision and the dependencies in
[`pyproject.toml`](pyproject.toml), then install:

```bash
python -m pip install --no-deps -e .
python -m pytest -q
```

Raw MRI preparation uses HD-BET and ANTs. Cache the AlexNet weights before
running perceptual training on an offline node.

## Run

Create a dataset manifest from [`examples/`](examples/) with image paths,
independent reference subjects, and training/test assignments.

```bash
# Validate the manifest and generate the experiment configuration.
python tools/workflow.py --manifest local/experiment.yaml --output runs/experiment

# Run preprocessing-complete data through reference construction,
# validation-based model selection, training, inference and evaluation.
python tools/workflow.py --manifest local/experiment.yaml --output runs/experiment \
  --execute --stage all
```

The shared recipe follows the original MRIxFields early-stopping configuration:
256 patches per training subject per epoch, validation every five epochs, and
SSIM selection with a full-volume SSIM guard. Training flips and inference TTA
are disabled. Declare validation subjects explicitly in the manifest.

`holdout` reserves a separate test cohort. `heldout_validation` reports the same
held-out cohort used for checkpoint selection; `fixed_epoch` uses a prescribed
training duration.

SRPBS uses the [rotating 3/3/3 protocol](docs/SRPBS_PROTOCOL.md).

## Code

- [`multiq/`](multiq/): calibration, 3D UNet++, paired data, losses and metrics.
- [`configs/common.yaml`](configs/common.yaml): shared experiment settings.
- [`tools/`](tools/): data preparation and experiment workflows.
- [`docs/USAGE.md`](docs/USAGE.md): data layout, commands and selection settings.

[MIT license](LICENSE)
