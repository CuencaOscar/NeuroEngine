"""Download the trained Keras model from the GitHub release."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://github.com/CuencaOscar/NeuroEngine/releases/download/v1.0.0/alzheimer-classifier.h5"


def main() -> int:
    url = os.getenv("MODEL_URL", DEFAULT_URL)
    target = Path(os.getenv("MODEL_PATH", "models/alzheimer-classifier.h5"))
    if target.exists():
        print(f"Model already present at {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {target}")
    urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed HTTPS release URL
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
