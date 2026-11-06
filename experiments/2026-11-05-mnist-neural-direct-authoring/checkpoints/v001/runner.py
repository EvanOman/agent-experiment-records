"""Generic, deterministic feed-forward tensor-graph runner. No fitting code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


class Network:
    def __init__(self, candidate):
        root = Path(candidate)
        self.graph = json.loads((root / "graph.json").read_text())
        with np.load(root / "weights.npz", allow_pickle=False) as archive:
            self.weights = {k: archive[k] for k in archive.files}

    def forward(self, images, diagnostics=False):
        values = {"pixels": np.asarray(images, dtype=np.float32).reshape(-1, 28, 28)}
        values.update(self.weights)
        uses = {}
        for node in self.graph["nodes"]:
            for key in node["inputs"]:
                uses[key] = uses.get(key, 0) + 1
        keep = set(self.graph["outputs"])
        for node in self.graph["nodes"]:
            args = [values[k] for k in node["inputs"]]
            op = node["op"]
            if op == "add": out = args[0] + args[1]
            elif op == "sub": out = args[0] - args[1]
            elif op == "mul": out = args[0] * args[1]
            elif op == "div": out = args[0] / args[1]
            elif op == "minimum": out = np.minimum(args[0], args[1])
            elif op == "maximum": out = np.maximum(args[0], args[1])
            elif op == "relu": out = np.maximum(args[0], 0)
            elif op == "step": out = (args[0] >= 0).astype(np.float32)
            elif op == "sum": out = np.sum(args[0], axis=tuple(node["axes"]))
            elif op == "max": out = np.max(args[0], axis=tuple(node["axes"]))
            elif op == "slice":
                index = tuple(slice(*v) for v in node["slices"])
                out = args[0][index]
            elif op == "pad": out = np.pad(args[0], node["padding"], constant_values=float(args[1]))
            elif op in ("maxpool", "minpool"):
                fn = maximum_filter if op == "maxpool" else minimum_filter
                out = fn(args[0], size=(1, *node["kernel"]), mode="constant", cval=float(args[1]))
            elif op == "cummax":
                axis = node["axis"]
                source = np.flip(args[0], axis) if node.get("reverse") else args[0]
                out = np.maximum.accumulate(source, axis=axis)
                if node.get("reverse"): out = np.flip(out, axis)
            elif op == "stack": out = np.stack(args, axis=node["axis"])
            elif op == "affine": out = args[0] @ args[1] + args[2]
            else: raise ValueError(f"Unsupported graph operation {op}")
            values[node["name"]] = out
            for key in node["inputs"]:
                uses[key] -= 1
                if uses[key] == 0 and key not in keep and key not in self.weights:
                    del values[key]
        if diagnostics:
            return {key: values[key] for key in self.graph["outputs"]}
        return values[self.graph["prediction_logits"]]

    def predict(self, images):
        return np.argmax(self.forward(images), axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    parser.add_argument("images", help="NPY array of image pixels; no labels are needed")
    parser.add_argument("output", help="NPY predictions output")
    args = parser.parse_args()
    network = Network(args.candidate)
    images = np.load(args.images, allow_pickle=False)
    predictions = np.concatenate([network.predict(images[i:i+128]) for i in range(0, len(images), 128)])
    np.save(args.output, predictions, allow_pickle=False)


if __name__ == "__main__":
    main()
