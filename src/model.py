"""Compatibility wrapper for model architectures.

The actual model files are organized under src/models/.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.baseline import ResNet18Baseline  # noqa: F401
from src.models.cnn_se import DeepfakeCNNSE, SEBlock  # noqa: F401
