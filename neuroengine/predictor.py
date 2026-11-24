"""Inference wrapper around the trained Keras CNN (4 Alzheimer stages)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

INPUT_SIZE = (128, 128)
# Alphabetical order, as produced by the folder-based training data.
CLASS_NAMES: tuple[str, ...] = ("Mild_Demented", "Moderate_Demented", "Non_Demented", "Very_Mild_Demented")


@dataclass(frozen=True)
class Prediction:
    label: str
    probability: float  # percentage, 0-100


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> list[Prediction]: ...


def preprocess(image: Image.Image) -> np.ndarray:
    """RGB, 128x128, scaled to [0, 1], with a batch dimension."""
    array = np.asarray(image.convert("RGB").resize(INPUT_SIZE), dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


class KerasPredictor:
    """Loads the model once. Its last layer already applies softmax."""

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found at {model_path}. Run `python scripts/download_model.py`.")
        from tensorflow import keras

        self._model = keras.models.load_model(model_path, compile=False)

    def predict(self, image: Image.Image) -> list[Prediction]:
        probs = self._model.predict(preprocess(image), verbose=0)[0]
        ranked = sorted(zip(CLASS_NAMES, probs, strict=True), key=lambda item: item[1], reverse=True)
        return [Prediction(label=name, probability=float(p) * 100) for name, p in ranked]
