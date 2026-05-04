"""Class-structure evaluation entry point.

The main evaluation script is src/eval.py. This wrapper lets the project also
support the conventional command:

    python src/evaluate.py ...
"""

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval import main


if __name__ == "__main__":
    main()
