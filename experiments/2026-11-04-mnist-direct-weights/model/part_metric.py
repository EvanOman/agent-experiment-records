"""Authored fixed multiscale RBF features and a coherent part-deformation graph.

Native OpenAI author: /root/architecture_b. The graph and numerical constants
were directly authored after inspecting approved development error activations.
No values are fitted, no templates are extracted from examples, and this module
does not access files, labels, sample identifiers, or parameter-writing routines.

extract: normalized (N,H,W) pixels -> (N,4,H,W) shared feature maps.
compare: two extracted batches -> (N,M) nonnegative discrepancy, lower is better.
The same extraction and latent-state graph is applied to images and templates.
Finite-state minimum pooling selects activations, never updates any weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage


VERSION = "coherent-part-rbf-v1"

DEFAULT_PARAMETERS: dict[str, Any] = {
    "narrow_kernel": [1, 2, 1],
    "narrow_scale": 0.25,
    "wide_kernel": [1, 4, 6, 4, 1],
    "wide_scale": 0.0625,
    "derivative_kernel": [-1, 0, 1],
    "derivative_scale": 0.5,
    "derivative_gain": 2.0,
    "channel_weights": [0.35, 0.25, 0.20, 0.20],
    "grid": [3, 3],
    "parents": [3, 4, 5, 4, -1, 4, 3, 4, 5],
    "postorder": [0, 6, 3, 1, 2, 8, 5, 7, 4],
    "offsets_yx": [
        [-2, -4], [-2, -2], [-2, 0], [-2, 2], [-2, 4],
        [0, -4], [0, -2], [0, 0], [0, 2], [0, 4],
        [2, -4], [2, -2], [2, 0], [2, 2], [2, 4],
    ],
    "unary_coefficient": 0.00015,
    "transition_coefficient": 0.0005,
    "deformed_weight": 0.80,
    "undeformed_weight": 0.20,
    "symmetric_scale": 0.5,
}


def _parameters(parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    unknown = set(parameters or ()) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"Unknown part-metric parameters: {sorted(unknown)}")
    return {**DEFAULT_PARAMETERS, **(parameters or {})}


def _finite_array(values: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _kernel(parameters: dict[str, Any], name: str) -> np.ndarray:
    values = _finite_array(parameters[f"{name}_kernel"], f"{name}_kernel", 1)
    scale = float(parameters[f"{name}_scale"])
    if not len(values) or len(values) % 2 == 0 or not np.isfinite(scale):
        raise ValueError("Convolution kernels must have odd nonzero lengths and finite scales")
    return values * np.float32(scale)


def extract(
    normalized_batch: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Shared narrow/wide brightness and signed x/y gradient convolutions.

    Signed gradients retain boundary direction. Unlike one-sided coverage,
    squared feature mismatch penalizes both missing and additional strokes.
    All convolutions use zero padding and never mix examples across the batch.
    """
    p = _parameters(parameters)
    images = _finite_array(normalized_batch, "normalized_batch", 3)
    if min(images.shape[1:]) < 1:
        raise ValueError("Images must have nonempty spatial dimensions")
    if images.size and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("Normalized pixels must lie in [0,1]")
    output = np.empty((len(images), 4, *images.shape[1:]), dtype=np.float32)
    if not len(images):
        return output
    for channel, name in enumerate(("narrow", "wide")):
        kernel = _kernel(p, name)
        blurred = ndimage.correlate1d(images, kernel, axis=1, mode="constant", cval=0.0)
        output[:, channel] = ndimage.correlate1d(
            blurred, kernel, axis=2, mode="constant", cval=0.0
        )
    derivative = _kernel(p, "derivative")
    gain = float(p["derivative_gain"])
    if not np.isfinite(gain):
        raise ValueError("derivative_gain must be finite")
    for channel, axis in ((2, 2), (3, 1)):
        output[:, channel] = np.float32(gain) * ndimage.correlate1d(
            output[:, 0], derivative, axis=axis, mode="constant", cval=0.0
        )
    return output


def _graph(parameters: dict[str, Any], height: int, width: int):
    grid = np.asarray(parameters["grid"])
    offsets = np.asarray(parameters["offsets_yx"])
    parents = np.asarray(parameters["parents"])
    postorder = np.asarray(parameters["postorder"])
    for array in (grid, offsets, parents, postorder):
        if array.dtype.kind not in "iu":
            raise ValueError("Grid, offsets, parents, and postorder must contain integers")
    if grid.shape != (2,) or np.any(grid < 1) or height < grid[0] or width < grid[1]:
        raise ValueError("grid must fit the feature canvas")
    parts = int(np.prod(grid))
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets):
        raise ValueError("offsets_yx must contain a nonempty fixed list of two-dimensional offsets")
    if len(np.unique(offsets, axis=0)) != len(offsets):
        raise ValueError("Displacement states must be distinct")
    zero = np.flatnonzero((offsets == 0).all(axis=1))
    if len(zero) != 1:
        raise ValueError("Exactly one zero displacement state is required")
    if parents.shape != (parts,) or postorder.shape != (parts,):
        raise ValueError("parents and postorder must name every part")
    if sorted(postorder.tolist()) != list(range(parts)):
        raise ValueError("postorder must visit every part exactly once")
    roots = np.flatnonzero(parents == -1)
    if len(roots) != 1 or parents.min() < -1 or parents.max() >= parts:
        raise ValueError("The fixed part graph must have exactly one root")
    positions = np.empty(parts, dtype=np.int32)
    positions[postorder] = np.arange(parts)
    for child, parent in enumerate(parents):
        if parent >= 0 and positions[child] >= positions[parent]:
            raise ValueError("The fixed graph must be a tree in child-before-parent order")
    weights = _finite_array(parameters["channel_weights"], "channel_weights", 1)
    if weights.shape != (4,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Expected four nonnegative channel weights with positive sum")
    for name in ("unary_coefficient", "transition_coefficient", "deformed_weight", "undeformed_weight", "symmetric_scale"):
        value = float(parameters[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    # All arrays below are deterministic expansions of authored graph parameters.
    difference = offsets[:, None, :] - offsets[None, :, :]
    transitions = np.float32(parameters["transition_coefficient"] / max(parts - 1, 1)) * (difference**2).sum(axis=2)
    unary = np.float32(parameters["unary_coefficient"] / parts) * (offsets**2).sum(axis=1)
    edges_y = np.arange(grid[0] + 1) * height // grid[0]
    edges_x = np.arange(grid[1] + 1) * width // grid[1]
    regions = [
        (slice(edges_y[row], edges_y[row + 1]), slice(edges_x[col], edges_x[col + 1]))
        for row in range(grid[0]) for col in range(grid[1])
    ]
    return offsets, parents, postorder, int(roots[0]), int(zero[0]), weights, transitions.astype(np.float32), unary.astype(np.float32), regions


def _shifted(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Sample F(y+dy,x+dx), with true zero padding and no wraparound."""
    height, width = values.shape[-2:]
    result = np.zeros_like(values)
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y1 > y0 and x1 > x0:
        result[..., y0:y1, x0:x1] = values[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def _directed(source: np.ndarray, reference: np.ndarray, graph):
    offsets, parents, postorder, root, zero, weights, transitions, unary, regions = graph
    n, m = len(source), len(reference)
    area = source.shape[-2] * source.shape[-1]
    scale = np.sqrt(weights).astype(np.float32)[None, :, None, None]
    # Scaling by sqrt(weight) makes the RBF square-distance channel weights exact.
    weighted_source = source * scale
    weighted_reference = reference * scale
    moved = np.stack([_shifted(weighted_source, int(dy), int(dx)) for dy, dx in offsets], axis=1)
    emissions = []
    for ys, xs in regions:
        a = moved[:, :, :, ys, xs].reshape(n * len(offsets), -1)
        b = weighted_reference[:, :, ys, xs].reshape(m, -1)
        squares = (a * a).sum(axis=1)[:, None] + (b * b).sum(axis=1)[None, :]
        squares -= np.float32(2.0) * (a @ b.T)
        # This is a squared norm; clip only floating-point cancellation noise.
        np.maximum(squares, 0.0, out=squares)
        emissions.append((squares / np.float32(area)).reshape(n, len(offsets), m).transpose(0, 2, 1))
    undeformed = sum(emission[:, :, zero] for emission in emissions)
    accumulated = [emission + unary[None, None, :] for emission in emissions]
    for child in postorder:
        parent = parents[child]
        if parent >= 0:
            # Child-state and parent-state axes are fixed graph units; minimum
            # pooling computes latent activations without updating parameters.
            message = np.min(accumulated[child][:, :, :, None] + transitions[None, None, :, :], axis=2)
            accumulated[parent] += message
    return accumulated[root].min(axis=2), undeformed


def compare(
    image_features: np.ndarray,
    template_features: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return symmetric coherent-deformation RBF discrepancies.

    The tree has fixed parts, fixed displacement states, and fixed quadratic
    displacement/coherence biases. Both directed matches are evaluated, then
    an undeformed full-field term preserves missing/extra-stroke evidence.
    Internal chunks limit temporary memory only; they do not enter the graph.
    """
    p = _parameters(parameters)
    images = _finite_array(image_features, "image_features", 4)
    templates = _finite_array(template_features, "template_features", 4)
    if images.shape[1:] != templates.shape[1:] or images.shape[1] != 4:
        raise ValueError("Both feature batches must have shape (batch,4,H,W)")
    graph = _graph(p, images.shape[-2], images.shape[-1])
    result = np.empty((len(images), len(templates)), dtype=np.float32)
    if not result.size:
        return result
    # Template blocking also bounds temporary state-pair tensors when a style
    # library is expanded with directly authored shared width variants.
    for i in range(0, len(images), 64):
        for j in range(0, len(templates), 128):
            a, b = images[i:i + 64], templates[j:j + 128]
            forward, stationary = _directed(a, b, graph)
            backward, _ = _directed(b, a, graph)
            deformed = np.float32(p["symmetric_scale"]) * (forward + backward.T)
            result[i:i + len(a), j:j + len(b)] = (
                np.float32(p["deformed_weight"]) * deformed
                + np.float32(p["undeformed_weight"]) * stationary
            )
    return result
