"""Supervised dense regression losses for aligned MRI patches."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def ssim3d_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
    data_range: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    if window % 2 == 0 or window < 3:
        raise ValueError("SSIM window must be odd and >= 3")
    prediction = prediction.float()
    target = target.float()
    padding = window // 2
    mu_x = F.avg_pool3d(prediction, window, stride=1, padding=padding)
    mu_y = F.avg_pool3d(target, window, stride=1, padding=padding)
    sigma_x = F.avg_pool3d(prediction.square(), window, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool3d(target.square(), window, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool3d(prediction * target, window, 1, padding) - mu_x * mu_y
    if isinstance(data_range, torch.Tensor):
        scale = data_range.to(device=prediction.device, dtype=prediction.dtype)
        if scale.shape != (prediction.shape[0], 1, 1, 1, 1):
            raise ValueError("3D SSIM data_range must have shape (B, 1, 1, 1, 1)")
    else:
        scale = float(data_range)
    c1, c2 = ((0.01 * scale) ** 2, (0.03 * scale) ** 2)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)


def axial_ssim2d_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
    data_range: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return valid-window 2D SSIM maps for every original-grid axial slice."""
    if window % 2 == 0 or window < 3:
        raise ValueError("SSIM window must be odd and >= 3")
    prediction = prediction.float()
    target = target.float()
    is_slice_batch = prediction.ndim == 4
    if is_slice_batch:
        batch, channels, size_x, size_y = prediction.shape
        size_z = 1
        pred_2d, target_2d = (prediction, target)
    elif prediction.ndim == 5:
        batch, channels, size_x, size_y, size_z = prediction.shape
        pred_2d = prediction.permute(0, 4, 1, 2, 3).reshape(
            batch * size_z, channels, size_x, size_y
        )
        target_2d = target.permute(0, 4, 1, 2, 3).reshape_as(pred_2d)
    else:
        raise ValueError("Axial SSIM expects 4D slices or 5D volumes")
    mu_x = F.avg_pool2d(pred_2d, window, stride=1)
    mu_y = F.avg_pool2d(target_2d, window, stride=1)
    sigma_x = F.avg_pool2d(pred_2d.square(), window, 1) - mu_x.square()
    sigma_y = F.avg_pool2d(target_2d.square(), window, 1) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred_2d * target_2d, window, 1) - mu_x * mu_y
    if isinstance(data_range, torch.Tensor):
        scale = data_range.to(device=prediction.device, dtype=prediction.dtype)
        expected = (batch, 1, 1, 1) if is_slice_batch else (batch, 1, 1, 1, 1)
        if scale.shape != expected:
            raise ValueError(f"Axial SSIM data_range must have shape {expected}")
        scale = scale.reshape(batch).repeat_interleave(size_z).view(-1, 1, 1, 1)
    else:
        scale = float(data_range)
    c1, c2 = ((0.01 * scale) ** 2, (0.03 * scale) ** 2)
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    result = numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    out_x, out_y = result.shape[-2:]
    if is_slice_batch:
        return result
    return result.reshape(batch, size_z, channels, out_x, out_y).permute(0, 2, 3, 4, 1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def gradient_l1(
    prediction: torch.Tensor, target: torch.Tensor, foreground: torch.Tensor
) -> torch.Tensor:
    losses = []
    for dimension in range(2, prediction.ndim):
        pred_grad = torch.diff(prediction, dim=dimension)
        target_grad = torch.diff(target, dim=dimension)
        first = [slice(None)] * foreground.ndim
        second = [slice(None)] * foreground.ndim
        first[dimension] = slice(None, -1)
        second[dimension] = slice(1, None)
        weights = torch.maximum(foreground[tuple(first)], foreground[tuple(second)])
        losses.append(_weighted_mean((pred_grad - target_grad).abs(), weights))
    return torch.stack(losses).mean()


class MappingLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.0,
        ssim_weight: float = 0.2,
        gradient_weight: float = 0.1,
        high_frequency_weight: float = 0.0,
        foreground_weight: float = 0.0,
        foreground_threshold: float = 0.0,
        ssim_window: int = 7,
        ssim_spatial_dims: int = 3,
        score_region_weight: float = 0.0,
        ssim_data_range: float | str = 1.0,
        ssim_minimum_data_range: float = 0.05,
        perceptual_weight: float = 0.0,
        perceptual_network: str = "alex",
        perceptual_fake_3d_ratio: float = 0.5,
        perceptual_axis_scale: float = 1.0 / 3.0,
        perceptual_min_size: int = 32,
        mask_to_output_support: bool = False,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.ssim_weight = float(ssim_weight)
        self.gradient_weight = float(gradient_weight)
        self.high_frequency_weight = float(high_frequency_weight)
        self.foreground_weight = float(foreground_weight)
        self.foreground_threshold = float(foreground_threshold)
        self.ssim_window = int(ssim_window)
        self.ssim_spatial_dims = int(ssim_spatial_dims)
        self.score_region_weight = float(score_region_weight)
        self.ssim_data_range = ssim_data_range
        self.ssim_minimum_data_range = float(ssim_minimum_data_range)
        self.perceptual_weight = float(perceptual_weight)
        self.perceptual_network = str(perceptual_network)
        self.perceptual_fake_3d_ratio = float(perceptual_fake_3d_ratio)
        self.perceptual_axis_scale = float(perceptual_axis_scale)
        self.perceptual_min_size = int(perceptual_min_size)
        self.mask_to_output_support = bool(mask_to_output_support)
        self.perceptual_extractor: nn.Module | None = None
        if self.perceptual_weight:
            if self.perceptual_network != "alex":
                raise ValueError("perceptual_network must be 'alex'")
            if not 0.0 < self.perceptual_fake_3d_ratio <= 1.0:
                raise ValueError("perceptual_fake_3d_ratio must be in (0, 1]")
            if self.perceptual_axis_scale <= 0.0:
                raise ValueError("perceptual_axis_scale must be positive")
            if self.perceptual_min_size < 31:
                raise ValueError("perceptual_min_size must be at least 31")
            try:
                from monai.losses import PerceptualLoss
            except ImportError as error:
                raise ImportError(
                    "MONAI fake-3D perceptual loss requires monai and lpips"
                ) from error
            self.perceptual_extractor = PerceptualLoss(
                spatial_dims=3,
                network_type=self.perceptual_network,
                is_fake_3d=True,
                fake_3d_ratio=self.perceptual_fake_3d_ratio,
            )
            self.perceptual_extractor.requires_grad_(False).eval()
        if self.ssim_spatial_dims not in (2, 3):
            raise ValueError("ssim_spatial_dims must be 2 or 3")
        if isinstance(self.ssim_data_range, str):
            if self.ssim_data_range != "target":
                raise ValueError("ssim_data_range string must be 'target'")
        elif float(self.ssim_data_range) <= 0:
            raise ValueError("ssim_data_range must be positive")
        if self.ssim_minimum_data_range <= 0:
            raise ValueError("ssim_minimum_data_range must be positive")

    def _monai_fake_3d_perceptual(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the MONAI three-axis fake-3D AlexNet perceptual objective."""
        if self.perceptual_extractor is None:
            return prediction.new_zeros(())
        if prediction.ndim != 5:
            raise ValueError("MONAI fake-3D perceptual loss expects 5D BCDHW volumes")
        if prediction.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError(
                "MONAI fake-3D AlexNet perceptual loss expects one MRI channel"
            )
        self.perceptual_extractor.eval()

        def rgb_metric_range(volume: torch.Tensor) -> torch.Tensor:
            volume = volume.float().repeat(1, 3, 1, 1, 1).mul(2.0).sub(1.0)
            padding = []
            for size in reversed(volume.shape[2:]):
                missing = max(0, self.perceptual_min_size - size)
                padding.extend((missing // 2, missing - missing // 2))
            return F.pad(volume, padding, value=-1.0) if any(padding) else volume

        axis_sum = self.perceptual_extractor(
            rgb_metric_range(prediction), rgb_metric_range(target)
        )
        return axis_sum.float() * self.perceptual_axis_scale

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        score_mask: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.mask_to_output_support:
            if valid_mask is None:
                raise ValueError(
                    "valid_mask is required when mask_to_output_support is enabled"
                )
            if valid_mask.shape != target.shape:
                raise ValueError("valid_mask must have the same shape as target")
            valid_mask = (valid_mask > 0).to(dtype=prediction.dtype)
            prediction = prediction * valid_mask
            target = target * valid_mask
        foreground = (target > self.foreground_threshold).to(dtype=prediction.dtype)
        weights = 1.0 + self.foreground_weight * foreground
        if valid_mask is not None:
            weights = weights * valid_mask
        if score_mask is not None and self.score_region_weight:
            if score_mask.shape != target.shape:
                raise ValueError("score_mask must have the same shape as target")
            weights = weights * (
                1.0 + self.score_region_weight * score_mask.to(dtype=prediction.dtype)
            )
        l1 = _weighted_mean((prediction - target).abs(), weights)
        mse = _weighted_mean((prediction - target).square(), weights)
        if self.ssim_data_range == "target":
            data_range: float | torch.Tensor = (
                target.amax(dim=tuple(range(1, target.ndim)), keepdim=True)
                - target.amin(dim=tuple(range(1, target.ndim)), keepdim=True)
            ).clamp_min(self.ssim_minimum_data_range)
        else:
            data_range = float(self.ssim_data_range)
        if self.ssim_spatial_dims == 2:
            ssim_map = axial_ssim2d_map(
                prediction, target, self.ssim_window, data_range=data_range
            )
            padding = self.ssim_window // 2
            if weights.ndim == 4:
                ssim_weights = weights[:, :, padding:-padding, padding:-padding]
            else:
                ssim_weights = weights[:, :, padding:-padding, padding:-padding, :]
        else:
            ssim_map = ssim3d_map(
                prediction, target, self.ssim_window, data_range=data_range
            )
            ssim_weights = weights
        ssim_loss = _weighted_mean(1.0 - ssim_map, ssim_weights)
        gradient = gradient_l1(prediction, target, weights)
        average_pool = F.avg_pool2d if prediction.ndim == 4 else F.avg_pool3d
        prediction_high = prediction - average_pool(prediction, 3, 1, 1)
        target_high = target - average_pool(target, 3, 1, 1)
        high_frequency = _weighted_mean((prediction_high - target_high).abs(), weights)
        perceptual = self._monai_fake_3d_perceptual(prediction, target)
        total = (
            self.l1_weight * l1
            + self.mse_weight * mse
            + self.ssim_weight * ssim_loss
            + self.gradient_weight * gradient
            + self.high_frequency_weight * high_frequency
            + self.perceptual_weight * perceptual
        )
        return {
            "total": total,
            "l1": l1,
            "mse": mse,
            "ssim": ssim_loss,
            "gradient": gradient,
            "high_frequency": high_frequency,
            "perceptual": perceptual,
        }
