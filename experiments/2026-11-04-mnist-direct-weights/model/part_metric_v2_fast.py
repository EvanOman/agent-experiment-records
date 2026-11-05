"""The v2 numerical graph with reusable, ephemeral part-matrix preparation.

Native OpenAI author: /root/architecture_b. The graph and numerical constants
were directly authored after inspecting approved development error activations.
No values are fitted, no templates are extracted from examples, and this module
does not access files, labels, sample identifiers, or parameter-writing routines.

extract: normalized (N,H,W) pixels -> (N,10,H,W) shared feature maps.
compare: two extracted batches -> (N,M) nonnegative discrepancy, lower is better.
The same extraction and latent-state graph is applied to images and templates.
Finite-state minimum pooling selects activations, never updates any weights.
This implementation preserves v2's numerical parameters and generic min-plus
messages. Only intermediate tensor layout and reuse across pair tiles differ.
No sample-identity cache or persistent image-derived storage is used.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage


VERSION = "oriented-constrained-part-rbf-v2-fast"

DEFAULT_PARAMETERS: dict[str, Any] = {
    "narrow_kernel": [1, 2, 1],
    "narrow_scale": 0.25,
    "wide_kernel": [1, 4, 6, 4, 1],
    "wide_scale": 0.0625,
    "derivative_kernel": [-1, 0, 1],
    "derivative_scale": 0.5,
    "derivative_gain": 2.0,
    "orientation_directions_xy": [
        [1.0, 0.0], [0.70710678, 0.70710678], [0.0, 1.0], [-0.70710678, 0.70710678],
        [-1.0, 0.0], [-0.70710678, -0.70710678], [0.0, -1.0], [0.70710678, -0.70710678],
    ],
    "orientation_pool_kernel": [1, 2, 1],
    "orientation_pool_scale": 0.25,
    "orientation_energy_floor": 0.0025,
    "channel_weights": [0.20, 0.10, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875],
    "grid": [4, 4],
    "parents": [1, 5, 1, 2, 5, -1, 5, 6, 9, 5, 9, 10, 13, 9, 13, 14],
    "postorder": [0, 3, 2, 1, 4, 7, 6, 8, 11, 10, 12, 15, 14, 13, 9, 5],
    "offsets_yx": [
        [-4, -4], [-4, -2], [-4, 0], [-4, 2], [-4, 4],
        [-2, -4], [-2, -2], [-2, 0], [-2, 2], [-2, 4],
        [0, -4], [0, -2], [0, 0], [0, 2], [0, 4],
        [2, -4], [2, -2], [2, 0], [2, 2], [2, 4],
        [4, -4], [4, -2], [4, 0], [4, 2], [4, 4],
    ],
    "neighbor_change_limit_yx": [2, 2],
    "unary_coefficient": 0.00015,
    "transition_coefficient": 0.0005,
    "deformed_weight": 0.80,
    "undeformed_weight": 0.05,
    "coarse_weight": 0.15,
    "coarse_grid": [4, 4],
    "coarse_channel_weights": [0.6666666666666666, 0.3333333333333333],
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
    """Shared brightness and eight locally normalized gradient directions.

    ReLU-projected signed gradients retain boundary direction. Shared local
    pooling and per-pixel orientation energy normalization reduce sensitivity
    to stroke width/contrast while retaining near-zero response in blank areas.
    All convolutions use zero padding and never mix examples across the batch.
    """
    p = _parameters(parameters)
    images = _finite_array(normalized_batch, "normalized_batch", 3)
    if min(images.shape[1:]) < 1:
        raise ValueError("Images must have nonempty spatial dimensions")
    if images.size and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("Normalized pixels must lie in [0,1]")
    output = np.empty((len(images), 10, *images.shape[1:]), dtype=np.float32)
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
    gx = np.float32(gain) * ndimage.correlate1d(output[:, 0], derivative, axis=2, mode="constant", cval=0.0)
    gy = np.float32(gain) * ndimage.correlate1d(output[:, 0], derivative, axis=1, mode="constant", cval=0.0)
    directions = _finite_array(p["orientation_directions_xy"], "orientation_directions_xy", 2)
    if directions.shape != (8, 2):
        raise ValueError("Eight explicit x/y orientation coefficient pairs are required")
    pool = _kernel(p, "orientation_pool")
    floor = float(p["orientation_energy_floor"])
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("orientation_energy_floor must be finite and positive")
    for channel, (cx, cy) in enumerate(directions, start=2):
        response = np.maximum(cx * gx + cy * gy, 0.0)
        response = ndimage.correlate1d(response, pool, axis=1, mode="constant", cval=0.0)
        output[:, channel] = ndimage.correlate1d(response, pool, axis=2, mode="constant", cval=0.0)
    norm = np.sqrt((output[:, 2:] ** 2).sum(axis=1) + np.float32(floor))
    output[:, 2:] /= norm[:, None]
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
    if weights.shape != (10,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Expected ten nonnegative channel weights with positive sum")
    for name in ("unary_coefficient", "transition_coefficient", "deformed_weight", "undeformed_weight", "coarse_weight", "symmetric_scale"):
        value = float(parameters[name])
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    # All arrays below are deterministic expansions of authored graph parameters.
    difference = offsets[:, None, :] - offsets[None, :, :]
    transitions = np.float32(parameters["transition_coefficient"] / max(parts - 1, 1)) * (difference**2).sum(axis=2)
    change_limit = np.asarray(parameters["neighbor_change_limit_yx"])
    if change_limit.shape != (2,) or change_limit.dtype.kind not in "iu" or np.any(change_limit < 0):
        raise ValueError("neighbor_change_limit_yx must contain two nonnegative integers")
    # Each parent-state unit receives only the statically allowed child-state
    # inputs. This literal graph constraint cannot be relaxed for any image.
    allowed = (np.abs(difference) <= change_limit[None, None]).all(axis=2)
    neighbors = [np.flatnonzero(allowed[:, parent_state]) for parent_state in range(len(offsets))]
    unary = np.float32(parameters["unary_coefficient"] / parts) * (offsets**2).sum(axis=1)
    edges_y = np.arange(grid[0] + 1) * height // grid[0]
    edges_x = np.arange(grid[1] + 1) * width // grid[1]
    regions = [
        (slice(edges_y[row], edges_y[row + 1]), slice(edges_x[col], edges_x[col + 1]))
        for row in range(grid[0]) for col in range(grid[1])
    ]
    return offsets, parents, postorder, int(roots[0]), int(zero[0]), weights, transitions.astype(np.float32), unary.astype(np.float32), regions, neighbors


def _shifted(values: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Sample F(y+dy,x+dx), with true zero padding and no wraparound."""
    height, width = values.shape[-2:]
    result = np.zeros_like(values)
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y1 > y0 and x1 > x0:
        result[..., y0:y1, x0:x1] = values[..., y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return result


def _prepare(features: np.ndarray, graph):
    """Prepare fixed shifted part matrices and squared norms once per batch.

    Each row retains v2's channel/row/column order, and each state retains the
    authored offset order. The zero-state rows also provide the unshifted
    reference matrix. All returned arrays are ephemeral forward activations.
    """
    offsets, _, _, _, zero, weights, _, _, regions, _ = graph
    count, states = len(features), len(offsets)
    scale = np.sqrt(weights).astype(np.float32)[None, :, None, None]
    weighted = features * scale
    pad_y = int(np.abs(offsets[:, 0]).max())
    pad_x = int(np.abs(offsets[:, 1]).max())
    padded = np.pad(weighted, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)))
    prepared = []
    for ys, xs in regions:
        # Small part patches are stacked directly. Avoid allocating a complete
        # shifted canvas for each state and copying strided part slices later.
        patches = [
            padded[:, :, ys.start + pad_y + dy:ys.stop + pad_y + dy,
                   xs.start + pad_x + dx:xs.stop + pad_x + dx]
            for dy, dx in offsets
        ]
        shifted = np.stack(patches, axis=1).reshape(count * states, -1)
        norms = (shifted * shifted).sum(axis=1)
        stationary = shifted.reshape(count, states, -1)[:, zero, :]
        stationary_norms = norms.reshape(count, states)[:, zero]
        prepared.append((shifted, norms, stationary, stationary_norms))
    return prepared


def _directed_prepared(source, reference, graph, area: int):
    offsets, parents, postorder, root, zero, weights, transitions, unary, regions, neighbors = graph
    n, m = len(source[0][2]), len(reference[0][2])
    emissions = []
    for source_part, reference_part in zip(source, reference, strict=True):
        a, a_norms = source_part[:2]
        b, b_norms = reference_part[2:]
        squares = a_norms[:, None] + b_norms[None, :]
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
            message = np.empty_like(accumulated[child])
            for parent_state, states in enumerate(neighbors):
                message[:, :, parent_state] = np.min(
                    accumulated[child][:, :, states] + transitions[states, parent_state][None, None], axis=2
                )
            accumulated[parent] += message
    return accumulated[root].min(axis=2), undeformed


def _directed(source: np.ndarray, reference: np.ndarray, graph):
    """Compatibility entry point for direct synthetic graph checks."""
    return _directed_prepared(
        _prepare(source, graph), _prepare(reference, graph), graph,
        source.shape[-2] * source.shape[-1],
    )


def _coarse(values: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Uniform fixed spatial averaging of the two brightness channels."""
    rows, columns = grid
    height, width = values.shape[-2:]
    if rows < 1 or columns < 1 or height < rows or width < columns:
        raise ValueError("coarse_grid must fit the image")
    y = np.arange(rows + 1) * height // rows
    x = np.arange(columns + 1) * width // columns
    return np.stack([
        values[:, :2, y[r]:y[r + 1], x[c]:x[c + 1]].mean(axis=(2, 3))
        for r in range(rows) for c in range(columns)
    ], axis=2)


def _coarse_cost(source: np.ndarray, reference: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    grid = np.asarray(parameters["coarse_grid"])
    if grid.shape != (2,) or grid.dtype.kind not in "iu":
        raise ValueError("coarse_grid must contain two integers")
    weights = _finite_array(parameters["coarse_channel_weights"], "coarse_channel_weights", 1)
    if weights.shape != (2,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Two nonnegative coarse-channel weights are required")
    scale = np.sqrt(weights)[None, :, None]
    a = (_coarse(source, tuple(grid)) * scale).reshape(len(source), -1)
    b = (_coarse(reference, tuple(grid)) * scale).reshape(len(reference), -1)
    result = (a*a).sum(axis=1)[:, None] + (b*b).sum(axis=1)[None, :] - np.float32(2.0) * (a @ b.T)
    return np.maximum(result, 0.0) / np.float32(np.prod(grid))


def compare(
    image_features: np.ndarray,
    template_features: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Return symmetric coherent-deformation RBF discrepancies.

    The tree has fixed parts/states, bounded neighbor changes, and quadratic
    displacement/coherence biases. Both directed matches are evaluated, with
    undeformed fine-field and coarse brightness terms retaining shape evidence.
    Internal chunks limit temporary memory only; they do not enter the graph.
    """
    p = _parameters(parameters)
    images = _finite_array(image_features, "image_features", 4)
    templates = _finite_array(template_features, "template_features", 4)
    if images.shape[1:] != templates.shape[1:] or images.shape[1] != 10:
        raise ValueError("Both feature batches must have shape (batch,10,H,W)")
    graph = _graph(p, images.shape[-2], images.shape[-1])
    result = np.empty((len(images), len(templates)), dtype=np.float32)
    if not result.size:
        return result
    # At most four image blocks and one template block are prepared together.
    # This bounds memory even when the public API receives thousands of images.
    # Preparations are reused only within this call, then discarded.
    area = images.shape[-2] * images.shape[-1]
    for group_start in range(0, len(images), 256):
        image_blocks = []
        for i in range(group_start, min(group_start + 256, len(images)), 64):
            a = images[i:i + 64]
            image_blocks.append((i, a, _prepare(a, graph)))
        for j in range(0, len(templates), 128):
            b = templates[j:j + 128]
            prepared_templates = _prepare(b, graph)
            for i, a, prepared_images in image_blocks:
                forward, stationary = _directed_prepared(prepared_images, prepared_templates, graph, area)
                backward, _ = _directed_prepared(prepared_templates, prepared_images, graph, area)
                deformed = np.float32(p["symmetric_scale"]) * (forward + backward.T)
                result[i:i + len(a), j:j + len(b)] = (
                    np.float32(p["deformed_weight"]) * deformed
                    + np.float32(p["undeformed_weight"]) * stationary
                    + np.float32(p["coarse_weight"]) * _coarse_cost(a, b, p)
                )
    return result
