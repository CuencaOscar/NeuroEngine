import io

import pytest
from PIL import Image

from neuroengine import create_app
from neuroengine.config import Settings
from neuroengine.predictor import Prediction


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return [
            Prediction("Non_Demented", 87.654),
            Prediction("Very_Mild_Demented", 10.0),
            Prediction("Mild_Demented", 2.0),
            Prediction("Moderate_Demented", 0.346),
        ]


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def client(predictor):
    app = create_app(Settings(model_path="unused", allowed_image_hosts=("firebasestorage.googleapis.com",)), predictor)
    app.testing = True
    return app.test_client()


@pytest.fixture
def png_bytes():
    buf = io.BytesIO()
    Image.new("L", (64, 64), color=120).save(buf, format="PNG")
    return buf.getvalue()
