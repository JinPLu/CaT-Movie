"""DOVER named-head adapter and shot-level media access for the block pipeline's quality scoring.

Importing this package does not import torch, onnxruntime or transformers: the
CPU orchestrator stays free of model frameworks, and each backend imports its
own stack only when it is first asked to score a shot.
"""
