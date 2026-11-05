"""Directly authored, shared image features and symmetric template costs.

Author: native OpenAI subagent ``/root/architecture_a``.
No images, labels, fitted values, or prior model parameters informed this module.

``extract`` takes normalized grayscale images with shape ``(N, H, W)`` and
returns feature maps with shape ``(N, 5, H, W)``. ``compare`` takes two such
feature batches and returns an ``(N, M)`` cost matrix; lower costs are better.
Both sides must use the same parameters and spatial resolution. The caller
owns class reduction and must persist the complete parameter mapping with a
candidate. This module never reads files, writes weights, or updates parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage


# These patterns and scalars were explicitly authored, without examining data.
# Keep the rational kernel numerators and their scales separate so their exact
# mathematical meaning is visible in a serialized parameter record.
DEFAULT_PARAMETERS: dict[str, Any] = {
    "smoothing_kernel": ((1, 2, 1), (2, 4, 2), (1, 2, 1)),
    "smoothing_scale": 0.0625,
    "line_kernels": (
        ((-1, -1, -1), (2, 2, 2), (-1, -1, -1)),
        ((-1, 2, -1), (-1, 2, -1), (-1, 2, -1)),
        ((2, -1, -1), (-1, 2, -1), (-1, -1, 2)),
        ((-1, -1, 2), (-1, 2, -1), (2, -1, -1)),
    ),
    "line_scale": 1.0 / 6.0,
    "line_response_gain": 2.0,
    "feature_clip": (0.0, 1.0),
    "pool_radii": (0, 1, 2),
    "radius_weights": (0.20, 0.55, 0.25),
    "channel_weights": (0.60, 0.10, 0.10, 0.10, 0.10),
    "coverage_epsilon": 0.001,
    "symmetric_scale": 0.5,
    "coarse_grid": (6, 6),
    "coarse_weight": 0.35,
}


def _parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve explicit overrides without changing the caller's mapping."""
    if parameters is None:
        return dict(DEFAULT_PARAMETERS)
    unknown = set(parameters) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown shared-feature parameters: {sorted(unknown)}")
    return {**DEFAULT_PARAMETERS, **parameters}


def _array(values: Any, name: str, dimensions: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != dimensions or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {dimensions}-dimensional array")
    return array


def extract(
    normalized_batch: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply the same occupancy and four orientation channels to every image.

    Inputs are already normalized grayscale pixels in [0, 1]. Convolutions use
    zero padding and do not mix the batch axis. All outputs have dtype float32.
    The first channel is smoothed occupancy; the following channels are
    horizontal, vertical, descending-diagonal, and ascending-diagonal strokes.
    """
    p = _parameters(parameters)
    images = _array(normalized_batch, "normalized_batch", 3)
    if min(images.shape[1:]) < 1:
        raise ValueError("Images must have nonempty spatial dimensions")
    if images.size and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("normalized_batch pixels must lie in [0, 1]")

    smooth = _array(p["smoothing_kernel"], "smoothing_kernel", 2)
    lines = _array(p["line_kernels"], "line_kernels", 3)
    if smooth.shape != (3, 3) or lines.shape != (4, 3, 3):
        raise ValueError("Expected one 3x3 smoothing kernel and four 3x3 line kernels")
    lower, upper = tuple(p["feature_clip"])
    if (lower, upper) != (0.0, 1.0):
        raise ValueError("Coverage features require feature_clip=(0, 1)")

    output = np.empty((images.shape[0], 5, *images.shape[1:]), dtype=np.float32)
    if not images.shape[0]:
        return output
    occupancy = ndimage.correlate(
        images,
        (smooth * np.float32(p["smoothing_scale"]))[None],
        mode="constant",
        cval=0.0,
    )
    output[:, 0] = np.clip(occupancy, lower, upper)
    for channel, kernel in enumerate(lines, start=1):
        response = ndimage.correlate(
            occupancy,
            (kernel * np.float32(p["line_scale"]))[None],
            mode="constant",
            cval=0.0,
        )
        response *= np.float32(p["line_response_gain"])
        output[:, channel] = np.clip(response, lower, upper)
    return output


def _pool(features: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return features
    width = 2 * radius + 1
    return ndimage.maximum_filter(
        features,
        size=(1, 1, width, width),
        mode="constant",
        cval=0.0,
    )


def _coarse(occupancy: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Fixed nonoverlapping spatial means; no statistics cross samples."""
    rows, columns = grid
    height, width = occupancy.shape[1:]
    if rows < 1 or columns < 1 or height < rows or width < columns:
        raise ValueError("coarse_grid must fit within the feature spatial dimensions")
    y_edges = np.arange(rows + 1) * height // rows
    x_edges = np.arange(columns + 1) * width // columns
    result = np.empty((len(occupancy), rows * columns), dtype=np.float32)
    for y in range(rows):
        for x in range(columns):
            result[:, y * columns + x] = occupancy[
                :, y_edges[y] : y_edges[y + 1], x_edges[x] : x_edges[x + 1]
            ].mean(axis=(1, 2))
    return result


def compare(
    image_features: np.ndarray,
    template_features: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return symmetric multiradius discrepancy for every image/template pair.

    For each channel and radius, the directed discrepancy is
    ``sum(A * (1 - maxpool(B))) / (epsilon + sum(A))``. Matrix
    multiplication evaluates all pairs without an N-by-M-by-pixel tensor.
    The reverse direction penalizes extra strokes as well as missing strokes.
    A coarse occupancy L1 term preserves overall spatial arrangement.
    """
    p = _parameters(parameters)
    images = _array(image_features, "image_features", 4)
    templates = _array(template_features, "template_features", 4)
    if images.shape[1:] != templates.shape[1:] or images.shape[1] != 5:
        raise ValueError("Feature batches must have matching shapes (batch, 5, H, W)")
    if min(images.shape[2:]) < 1:
        raise ValueError("Features must have nonempty spatial dimensions")
    for values in (images, templates):
        if values.size and (values.min() < 0.0 or values.max() > 1.0):
            raise ValueError("Coverage features must lie in [0, 1]")

    radii = tuple(p["pool_radii"])
    radius_weights = tuple(p["radius_weights"])
    channel_weights = tuple(p["channel_weights"])
    if len(radii) != len(radius_weights) or len(channel_weights) != 5:
        raise ValueError("Pool/channel weights must match their corresponding dimensions")
    if any(not isinstance(r, (int, np.integer)) or r < 0 for r in radii):
        raise ValueError("pool_radii must contain nonnegative integers")
    weights = np.asarray((*radius_weights, *channel_weights), dtype=np.float32)
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("Comparison weights must be finite and nonnegative")
    epsilon = float(p["coverage_epsilon"])
    symmetric_scale = float(p["symmetric_scale"])
    coarse_weight = float(p["coarse_weight"])
    if not np.isfinite([epsilon, symmetric_scale, coarse_weight]).all():
        raise ValueError("Comparison scalar parameters must be finite")
    if epsilon <= 0.0 or symmetric_scale < 0.0 or coarse_weight < 0.0:
        raise ValueError("Invalid comparison scalar parameter sign")

    result = np.zeros((len(images), len(templates)), dtype=np.float32)
    if not result.size:
        return result

    image_flat = images.reshape(len(images), 5, -1)
    template_flat = templates.reshape(len(templates), 5, -1)
    image_sums = image_flat.sum(axis=2)
    template_sums = template_flat.sum(axis=2)
    for radius, radius_weight in zip(radii, radius_weights, strict=True):
        if radius_weight == 0.0:
            continue
        pooled_images = _pool(images, int(radius)).reshape(len(images), 5, -1)
        pooled_templates = _pool(templates, int(radius)).reshape(len(templates), 5, -1)
        for channel, channel_weight in enumerate(channel_weights):
            if channel_weight == 0.0:
                continue
            missing = image_sums[:, channel, None] - (
                image_flat[:, channel] @ pooled_templates[:, channel].T
            )
            extra = template_sums[None, :, channel] - (
                pooled_images[:, channel] @ template_flat[:, channel].T
            )
            # Exact arithmetic is nonnegative; clip only floating point noise.
            np.maximum(missing, 0.0, out=missing)
            np.maximum(extra, 0.0, out=extra)
            missing /= epsilon + image_sums[:, channel, None]
            extra /= epsilon + template_sums[None, :, channel]
            result += np.float32(radius_weight * channel_weight * symmetric_scale) * (
                missing + extra
            )

    if coarse_weight:
        grid = tuple(p["coarse_grid"])
        if len(grid) != 2 or any(not isinstance(v, (int, np.integer)) for v in grid):
            raise ValueError("coarse_grid must contain two integers")
        coarse_images = _coarse(images[:, 0], grid)
        coarse_templates = _coarse(templates[:, 0], grid)
        cell_scale = np.float32(coarse_weight / coarse_images.shape[1])
        for cell in range(coarse_images.shape[1]):
            result += cell_scale * np.abs(
                coarse_images[:, None, cell] - coarse_templates[None, :, cell]
            )
    return result
