"""Construct the Multi-Q network from a checkpoint configuration."""

from copy import deepcopy


def build_mapping_model(model_config):
    from .backbone_unetpp import PairedResidualUNetPlusPlus3D

    kwargs = deepcopy(model_config)
    architecture = str(kwargs.pop("architecture", "unetpp")).lower()
    if architecture not in {"unetpp", "unet++"}:
        raise ValueError("Multi-Q supports the 3D UNet++ architecture")
    return PairedResidualUNetPlusPlus3D(**kwargs)
