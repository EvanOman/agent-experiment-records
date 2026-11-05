# Data and model provenance

The MNIST replay downloads the public TensorFlow/Keras dataset archive from:

`https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`

Expected SHA-256:

`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`

MNIST and its distribution remain subject to their own terms. Dataset images and labels are not redistributed in this repository. The replay creates a local ignored cache.

The checkpoint contains features expanded from agent-authored geometric digit descriptions and shared parameters. It is not a downloaded pretrained MNIST classifier. The original numerical archive is retained byte-for-byte, with SHA-256:

`298c01852a29ce46f8b41f339a2adf47b4982e6d6b4ff5b23a7934bfcb946123`

Portable packaging preserves the inference equations. The package manifest records original and public file hashes and identifies the loader/construction adaptation. Historical evidence hashes and public sanitized-file hashes serve different purposes; the publication manifest explains any transformation.
