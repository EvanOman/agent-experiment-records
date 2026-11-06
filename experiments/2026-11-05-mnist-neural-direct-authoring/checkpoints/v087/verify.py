"""Local artifact/inference verification; never accesses sealed data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", help="e.g. candidates/v002")
    args = parser.parse_args()
    root = Path(args.candidate)
    hashes = json.loads((root / "SHA256SUMS.json").read_text())
    for name, digest in hashes.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    graph = json.loads((root / "graph.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    with np.load(root / "weights.npz", allow_pickle=False) as archive:
        weights = {k: archive[k] for k in archive.files}
    assert all(v.dtype == np.float32 and np.all(np.isfinite(v)) for v in weights.values())
    assert sum(v.size for v in weights.values()) == provenance["expanded_tensor_elements"]
    known = {"pixels", *weights}
    for node in graph["nodes"]:
        assert node["name"] not in known, node["name"]
        assert all(k in known for k in node["inputs"]), node["name"]
        known.add(node["name"])
    assert all(k in known for k in graph["outputs"])
    spec = importlib.util.spec_from_file_location("verified_frozen_runner", root / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.Network(root)
    # Generic numerical edge cases, not glyphs or an exemplar/template bank.
    images = np.zeros((4, 28, 28), dtype=np.uint8)
    images[1] = 255
    images[2, 14, 14] = 255
    images[3] = np.arange(784, dtype=np.uint16).reshape(28, 28) % 256
    batched = network.forward(images)
    singles = np.concatenate([network.forward(image[None]) for image in images])
    assert batched.shape == (4, 10)
    assert np.all(np.isfinite(batched))
    np.testing.assert_allclose(batched, singles, atol=1e-5, rtol=1e-6)
    np.testing.assert_array_equal(network.predict(images), np.argmax(batched, axis=1))
    if "sh_resampler_plus" in weights:
        plus, minus = weights["sh_resampler_plus"], weights["sh_resampler_minus"]
        for matrix in (plus, minus):
            assert matrix.shape == (784, 784)
            assert np.count_nonzero(matrix) == 1448
            assert np.all(np.sum(matrix != 0, axis=0) <= 2)
            source, destination = np.nonzero(matrix)
            np.testing.assert_array_equal(source // 28, destination // 28)
            assert np.all(matrix >= 0)
        # Independent impulse expectations at top, pivot-adjacent, and bottom
        # rows; these are arithmetic checks, not digit exemplars.
        assert plus[5, 10] == np.float32(.05)
        assert plus[5, 9] == np.float32(.95)
        assert plus[13*28+13, 13*28+14] == np.float32(.15)
        assert plus[27*28+14, 27*28+10] == np.float32(.95)
        np.testing.assert_array_equal(minus, plus.reshape(28,28,28,28)[::-1,:,::-1,:].reshape(784,784))
        diagnostics = network.forward(images, diagnostics=True)["features"]
        names = graph["feature_names"]
        np.testing.assert_array_equal(diagnostics[:, names.index("sh_gate_sum")], np.ones(4))
        assert np.all(diagnostics[:, names.index("sh_selected_loss")] <= .005)
    print(json.dumps(dict(verified=str(root), nodes=len(graph["nodes"]),
                          tensor_elements=sum(v.size for v in weights.values()),
                          artifact_hashes=hashes), indent=2))


if __name__ == "__main__": main()
