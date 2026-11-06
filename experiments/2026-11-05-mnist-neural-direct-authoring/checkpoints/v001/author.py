"""Unaided analytical authoring. Never imports or opens examples or labels.

Only fixed numerical blocks, index connections, masks, and tied constants are
expanded here. No digit images, stroke drawings, prototypes, optimizer or search.
"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "candidates" / "v001"
W = {}
N = []
AUTHORED = {}


def weight(name, value, independent=None):
    W[name] = np.asarray(value, dtype=np.float32)
    AUTHORED[name] = W[name].size if independent is None else independent
    return name


def node(name, op, *inputs, **options):
    N.append(dict(name=name, op=op, inputs=list(inputs), **options))
    return name


def clip(name, source, scale=1):
    node(name + "_scale", "mul", source, weight(name + "_gain", scale))
    node(name + "_positive", "maximum", name + "_scale", "zero")
    return node(name, "minimum", name + "_positive", "one")


def summed(name, source, mask=None):
    if mask:
        source = node(name + "_masked", "mul", source, mask)
    return node(name, "sum", source, axes=[1, 2])


def mask(name, y0, y1, x0=0, x1=28):
    a = np.zeros((28, 28), dtype=np.float32)
    a[y0:y1, x0:x1] = 1
    return weight(name, a, independent=2)


def main():
    if DEST.exists(): raise RuntimeError("Frozen candidate already exists")
    DEST.mkdir(parents=True)
    weight("zero", 0)
    weight("one", 1)
    weight("pixel_scale", 1 / 255)
    weight("occupancy_threshold", .30)
    weight("area_epsilon", 1)
    weight("hole_gain", 1 / 8)
    weight("width_gain", 1 / 20)
    weight("narrow_bias", 13)
    weight("narrow_gain", 1 / 7)
    weight("hole_center_default", .5)
    weight("center_target", .5)
    weight("negative_one", -1)
    weight("quarter", .25)
    weight("two", 2)
    weight("three", 3)
    weight("four", 4)
    mask("top_mask", 0, 14)
    mask("bottom_mask", 14, 28)
    mask("cap_mask", 0, 10)
    mask("foot_mask", 19, 28)
    mask("middle_mask", 11, 18)
    border = np.zeros((28, 28), dtype=np.float32)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = 1
    weight("border_mask", border, independent=2)
    # Uniform coordinate ramp, not a digit template.
    weight("row_coordinate", np.arange(28, dtype=np.float32)[:, None] / 27, independent=28)
    node("unit_pixels", "mul", "pixels", "pixel_scale")
    node("threshold_shift", "sub", "unit_pixels", "occupancy_threshold")
    node("ink", "step", "threshold_shift")
    node("blank", "sub", "one", "ink")
    node("exterior_0", "mul", "blank", "border_mask")
    # A fixed feed-forward unrolling of local max/min pooling. 56 rounds can
    # cover ordinary 28x28 digit cavities; it is not an adaptive graph traversal.
    for i in range(56):
        node(f"spread_{i}", "maxpool", f"exterior_{i}", "zero", kernel=[3, 3])
        node(f"exterior_{i+1}", "minimum", f"spread_{i}", "blank")
    node("holes", "sub", "blank", "exterior_56")
    summed("hole_area", "holes")
    node("hole_scale", "mul", "hole_area", "hole_gain")
    node("hole_present", "minimum", "hole_scale", "one")
    summed("hole_y_sum", "holes", "row_coordinate")
    node("hole_y_numerator", "add", "hole_y_sum", "hole_center_default")
    node("hole_denominator", "add", "hole_area", "area_epsilon")
    node("hole_y", "div", "hole_y_numerator", "hole_denominator")
    node("hole_rows", "max", "holes", axes=[2])
    node("hole_height_raw", "sum", "hole_rows", axes=[1])
    node("hole_height", "mul", "hole_height_raw", "width_gain")
    node("hole_y_delta", "sub", "hole_y", "center_target")
    node("hole_y_neg_delta", "mul", "hole_y_delta", "negative_one")
    node("hole_y_abs", "maximum", "hole_y_delta", "hole_y_neg_delta")
    # Euler characteristic of a binary hole map from its 2x2 cells. Each
    # hole component has no interior hole, so its characteristic counts holes.
    node("holes_padded", "pad", "holes", "zero", padding=[[0, 0], [1, 1], [1, 1]])
    for name, ys, xs in [("a", [0, -1], [0, -1]), ("b", [0, -1], [1, None]),
                         ("c", [1, None], [0, -1]), ("d", [1, None], [1, None])]:
        node("cell_" + name, "slice", "holes_padded", slices=[[None, None], ys, xs])
    node("cell_ab", "add", "cell_a", "cell_b")
    node("cell_cd", "add", "cell_c", "cell_d")
    node("cell_count", "add", "cell_ab", "cell_cd")
    # Tent ReLU indicators at exactly 1 and 3 occupied vertices.
    for k in (1, 3):
        weight(f"k{k}_lo", k - 1)
        weight(f"k{k}_mid", k)
        weight(f"k{k}_hi", k + 1)
        for part in ("lo", "mid", "hi"):
            node(f"q{k}_{part}_shift", "sub", "cell_count", f"k{k}_{part}")
            node(f"q{k}_{part}", "relu", f"q{k}_{part}_shift")
        node(f"q{k}_twomid", "mul", f"q{k}_mid", "two")
        node(f"q{k}_diff", "sub", f"q{k}_lo", f"q{k}_twomid")
        node(f"q{k}", "add", f"q{k}_diff", f"q{k}_hi")
    node("cell_ad", "add", "cell_a", "cell_d")
    node("cell_bc", "add", "cell_b", "cell_c")
    node("diag_difference", "sub", "cell_ad", "cell_bc")
    node("diag_negative", "mul", "diag_difference", "negative_one")
    node("diag_abs", "maximum", "diag_difference", "diag_negative")
    node("diag_shift", "sub", "diag_abs", "one")
    node("diag_indicator", "relu", "diag_shift")
    node("diag_twice", "mul", "diag_indicator", "two")
    node("euler_diff", "sub", "q1", "q3")
    node("euler_cells", "add", "euler_diff", "diag_twice")
    summed("euler_total", "euler_cells")
    node("hole_count", "mul", "euler_total", "quarter")
    node("second_hole_raw", "sub", "hole_count", "one")
    clip("two_holes", "second_hole_raw")
    for name, axis, reverse in [("left", 2, False), ("right", 2, True),
                                 ("up", 1, False), ("down", 1, True)]:
        node("ray_" + name, "cummax", "ink", axis=axis, reverse=reverse)
    # Three-sided exterior gaps; these are reusable local geometric features.
    for direction, rays in [("L", ["right", "up", "down"]),
                            ("R", ["left", "up", "down"]),
                            ("U", ["left", "right", "down"]),
                            ("D", ["left", "right", "up"])]:
        s = "exterior_56"
        for j, ray in enumerate(rays):
            s = node(f"gap_{direction}_{j}", "minimum", s, "ray_" + ray)
        for band in (["top", "bottom"] if direction in ("L", "R") else ["all"]):
            name = f"gap_{direction}_{band}"
            summed(name + "_area", s, None if band == "all" else band + "_mask")
            clip(name, name + "_area", 1 / 30)
    node("column_occupied", "max", "ink", axes=[1])
    node("width", "sum", "column_occupied", axes=[1])
    node("narrow_raw", "sub", "narrow_bias", "width")
    clip("narrow", "narrow_raw", 1 / 7)
    for name in ("cap", "foot", "middle"):
        node(name + "_ink", "mul", "ink", name + "_mask")
        node(name + "_columns", "max", name + "_ink", axes=[1])
        node(name + "_width_raw", "sum", name + "_columns", axes=[1])
        node(name + "_width", "mul", name + "_width_raw", "width_gain")
    node("foot_narrow_raw", "sub", "cap_width", "foot_width")
    clip("foot_narrow", "foot_narrow_raw", 3)
    features = ["hole_present", "two_holes", "hole_y_delta", "hole_y_abs", "hole_height",
                "gap_L_top", "gap_R_top", "gap_L_bottom", "gap_R_bottom",
                "gap_U_all", "gap_D_all", "narrow", "cap_width", "foot_width",
                "middle_width", "foot_narrow"]
    node("features", "stack", *features, axis=1)
    # Each explicit row is a class score. Shared morphology features are not
    # image prototypes. These coefficients are specified before any data read.
    rows = [
        [4,-5, 0,-4, 4,  0,0,0,0, 0,0,-2, 0,0,0,0],
        [-4,-4,0,0,0, -1,-1,-1,-1,-1,-1,5,0,0,0,0],
        [-5,-5,0,0,0, 3,-2,-1.5,3,0,0,-2,0,0,0,0],
        [-4,-5,0,0,0, 3,-2,3,-2,0,0,-2,0,0,0,0],
        [2,-5,0,0,-1, 0,0,0,0,4,0,-2,-3,0,2,2],
        [-4,-5,0,0,0, -2,3,3,-2,0,0,-2,0,0,0,0],
        [4,-5,8,0,-2, 0,1.5,0,0,0,0,-2,0,0,0,0],
        [-4,-5,0,0,0, 0,0,-1,-2,0,0,-2,1.5,0,0,1.5],
        [3,6,0,0,0, 0,0,0,0,0,0,-2,0,0,0,0],
        [4,-5,-8,0,-2, 0,0,1.5,0,0,0,-2,0,0,0,0],
    ]
    weight("class_weights", np.asarray(rows, dtype=np.float32).T)
    weight("class_bias", [0, 1, .5, .5, 0, .5, 0, 2, 0, 0])
    node("logits", "affine", "features", "class_weights", "class_bias")
    graph = dict(schema="authored-neural-dag-v1", input="pixels", nodes=N,
                 outputs=["features", "logits"], feature_names=features,
                 prediction_logits="logits", input_shape=[None, 28, 28],
                 activations="affine, ReLU, hard threshold, max/min pooling, fixed normalization")
    (DEST / "graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    np.savez_compressed(DEST / "weights.npz", **W)
    shutil.copy2(ROOT / "runner.py", DEST / "runner.py")
    shutil.copy2(__file__, DEST / "author.py")
    provenance = dict(candidate="v001", stage="unaided before all development access",
                      description="Fixed binary morphology with 56 max/min exterior propagation layers, local Euler hole count, directional ray concavities, extent features and hand-authored affine ten-class readout.",
                      sources=["ASSIGNMENT.md", "data/manifest.json", "author's general mathematical and digit knowledge"],
                      independent_numerical_specifications=sum(AUTHORED.values()),
                      expanded_tensor_elements=sum(a.size for a in W.values()),
                      tensor_elements={k: a.size for k, a in W.items()},
                      authored_specifications=AUTHORED,
                      count_note="Mask literals 0 and 1 counted once per authored rectangular/border mask; their placement is structural. Coordinate ramp counts 28 values. Each class-matrix entry counts independently, including explicit zeros. Scalar tied constants counted once per tensor.",
                      training_or_search=False, development_access_before_freeze=False,
                      limitations="No guarantee of threshold accuracy. Topology and gap measures may be sensitive to slant, open loops and handwriting variants. Canonical data may have been in model pretraining.")
    (DEST / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.iterdir()) if p.is_file()}
    (DEST / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(dict(candidate="v001", hashes=hashes, nodes=len(N),
                          independent=sum(AUTHORED.values()), expanded=sum(a.size for a in W.values())), indent=2))


if __name__ == "__main__":
    main()
