from PIL import Image

from neuroengine.predictor import CLASS_NAMES, preprocess


def test_preprocess_shape_and_range():
    batch = preprocess(Image.new("L", (200, 180), color=255))
    assert batch.shape == (1, 128, 128, 3)
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_four_classes():
    assert len(CLASS_NAMES) == 4
