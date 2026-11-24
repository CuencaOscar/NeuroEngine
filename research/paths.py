"""Filesystem locations used by the research experiments.

Set NEUROENGINE_DATASET_DIR to the folder that contains the four class sub-folders
(Mild_Demented, Moderate_Demented, Non_Demented, Very_Mild_Demented).
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# `ROOT.joinpath("Dataset", ...)` resolves to the dataset folder.
_DATASET = Path(os.getenv("NEUROENGINE_DATASET_DIR", HERE.parent / "Dataset"))
ROOT = _DATASET.parent
DATASET_DIR_NAME = _DATASET.name
RESULTS_ROOT = HERE / "results"
