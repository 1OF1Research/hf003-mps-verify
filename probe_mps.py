"""Probe: is this runner really Apple Silicon with a live MPS device?"""
import platform

import torch

print("platform:", platform.platform())
print("machine:", platform.machine())
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
