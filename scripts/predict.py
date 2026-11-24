"""Classify a local MRI image from the command line: `python scripts/predict.py scan.jpg`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroengine.config import Settings  # noqa: E402
from neuroengine.predictor import KerasPredictor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    predictor = KerasPredictor(Settings().model_path)
    for p in predictor.predict(Image.open(args.image)):
        print(f"{p.label:<22} {p.probability:6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
