"""Compatibility wrapper for the audio preprocessing code.

The main preprocessing implementation lives in src/data/preprocess.py. This
file is included so the project also matches the simple class repository
structure.
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import *  # noqa: F401,F403
