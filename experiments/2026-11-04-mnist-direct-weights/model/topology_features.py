"""Directly authored shared Euler-characteristic convolution features.

Native OpenAI author: /root/architecture_b. All numerical patterns, thresholds,
and output coefficients are authored literals. This module reads only supplied
pixel/feature tensors; it has no file, label, sample-identity, or fitting access.

Conventions: strictly-greater thresholds; zero padding for the grayscale blur;
one zero-pixel border for binary maps; eight-connected foreground and its dual
four-connected background. Euler characteristic = components minus holes.
Every image and every authored template receives the identical fixed graph.

extract(normalized, parameters) -> float32 (N,6): three raw and three blurred
threshold-map Euler characteristics, in the declared threshold order.
compare(image_features, template_features, parameters) -> float32 (N,M):
coefficient times mean capped absolute feature discrepancy, lower is better.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage


VERSION = "shared-euler-convolution-v1"

DEFAULT_PARAMETERS: dict[str, Any] = {
    "thresholds": [0.30, 0.50, 0.70],
    "threshold_rule": "strictly_greater",
    "blur_kernel": [1, 2, 1],
    "blur_scale": 0.25,
    "zero_border": 1,
    "foreground_connectivity": 8,
    "background_connectivity": 4,
    "patterns": [
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [0, 1]],
        [[1, 1], [1, 0]],
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ],
    "output_weights": [0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, -0.50, -0.50],
    "absolute_cap": 1.0,
    "coefficient": 0.0015,
}


def _parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    unknown = set(parameters or ()) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown topology parameters: {sorted(unknown)}")
    p = {**DEFAULT_PARAMETERS, **(parameters or {})}
    if (p["threshold_rule"], p["zero_border"], p["foreground_connectivity"], p["background_connectivity"]) != ("strictly_greater", 1, 8, 4):
        raise ValueError("Euler convention requires strict thresholds, one zero border, foreground8/background4")
    thresholds = np.asarray(p["thresholds"], dtype=np.float32)
    if thresholds.shape != (3,) or not np.isfinite(thresholds).all() or np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Exactly three finite thresholds strictly inside (0,1) are required")
    kernel = np.asarray(p["blur_kernel"], dtype=np.float32)
    if kernel.ndim != 1 or not len(kernel) or len(kernel) % 2 != 1 or not np.isfinite(kernel).all():
        raise ValueError("blur_kernel must be a finite odd-length vector")
    patterns = np.asarray(p["patterns"], dtype=np.float32)
    weights = np.asarray(p["output_weights"], dtype=np.float32)
    if patterns.shape != (10, 2, 2) or not np.isin(patterns, (0, 1)).all() or weights.shape != (10,) or not np.isfinite(weights).all():
        raise ValueError("Expected ten binary2x2 patterns and ten finite output weights")
    if not np.isfinite(float(p["blur_scale"])):
        raise ValueError("blur_scale must be finite")
    if any(not np.isfinite(float(p[name])) or float(p[name]) < 0 for name in ("absolute_cap", "coefficient")):
        raise ValueError("absolute_cap and coefficient must be finite and nonnegative")
    return p


def _array(values: Any, name: str, dimensions: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != dimensions or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {dimensions}-dimensional array")
    return result


def _euler(binary: np.ndarray, parameters: Mapping[str, Any] | None = None) -> np.ndarray:
    """Ten literal2x2 convolution/ReLU pattern units and global sum pooling.

    For binary pattern P, W=2P-1 and b=1-sum(P). ReLU(W.patch+b)
    equals1 for an exact pattern match and0 for every other binary patch.
    These expansions depend only on authored patterns, never on examples.
    """
    p = _parameters(parameters)
    maps = np.asarray(binary)
    if maps.ndim != 3 or not np.isin(maps, (0, 1)).all():
        raise ValueError("binary must contain a batch of binary two-dimensional maps")
    maps = maps.astype(np.float32)
    height, width = maps.shape[-2:]
    if min(height, width) < 1:
        raise ValueError("Binary maps need nonempty spatial dimensions")
    padded = np.pad(maps, ((0, 0), (1, 1), (1, 1)))
    patches = [padded[:, y:y + height + 1, x:x + width + 1] for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))]
    patterns = np.asarray(p["patterns"], dtype=np.float32).reshape(10, 4)
    kernels = np.float32(2.0) * patterns - np.float32(1.0)
    biases = np.float32(1.0) - patterns.sum(axis=1)
    counts = np.empty((len(maps), 10), dtype=np.float32)
    for index, (kernel, bias) in enumerate(zip(kernels, biases, strict=True)):
        response = np.full((len(maps), height + 1, width + 1), bias, dtype=np.float32)
        for coefficient, patch in zip(kernel, patches, strict=True):
            response += coefficient * patch
        np.maximum(response, 0.0, out=response)
        counts[:, index] = response.sum(axis=(1, 2))
    return (counts * np.asarray(p["output_weights"], dtype=np.float32)[None]).sum(axis=1)


def extract(
    normalized_batch: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return three raw-map and three narrow-blur Euler characteristics."""
    p = _parameters(parameters)
    images = _array(normalized_batch, "normalized_batch", 3)
    if min(images.shape[1:]) < 1:
        raise ValueError("Images need nonempty spatial dimensions")
    if images.size and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("Normalized pixels must lie in [0,1]")
    if not len(images):
        return np.empty((0, 6), dtype=np.float32)
    kernel = np.asarray(p["blur_kernel"], dtype=np.float32) * np.float32(p["blur_scale"])
    blurred = ndimage.correlate1d(images, kernel, axis=1, mode="constant", cval=0.0)
    blurred = ndimage.correlate1d(blurred, kernel, axis=2, mode="constant", cval=0.0)
    return np.stack([
        _euler(source > np.float32(threshold), p)
        for source in (images, blurred) for threshold in p["thresholds"]
    ], axis=1)


def compare(
    image_features: np.ndarray,
    template_features: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Soft capped compatibility penalty; no class-specific decisions."""
    p = _parameters(parameters)
    images = _array(image_features, "image_features", 2)
    templates = _array(template_features, "template_features", 2)
    if images.shape[1] != 6 or templates.shape[1] != 6:
        raise ValueError("Both Euler feature batches must have six columns")
    difference = np.abs(images[:, None, :] - templates[None, :, :])
    np.minimum(difference, np.float32(p["absolute_cap"]), out=difference)
    return np.float32(p["coefficient"]) * difference.mean(axis=2)
